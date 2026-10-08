from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .matcher import HungarianMatcher
from .point_sampling import point_sample, sample_valid_point_coords


def _matched_dice_loss(inputs: Tensor, targets: Tensor) -> Tensor:
    probabilities = inputs.sigmoid()
    numerator = 2.0 * (probabilities * targets).sum(-1)
    denominator = probabilities.sum(-1) + targets.sum(-1)
    return (1.0 - (numerator + 1.0) / (denominator + 1.0)).mean()


class SetCriterion(nn.Module):
    def __init__(
        self,
        *,
        num_classes: int,
        matcher: HungarianMatcher,
        class_weight: float,
        mask_weight: float,
        dice_weight: float,
        no_object_weight: float,
        num_points: int,
        oversample_ratio: float,
        importance_sample_ratio: float,
        semantic_class_weights: Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.matcher = matcher
        self.class_weight = float(class_weight)
        self.mask_weight = float(mask_weight)
        self.dice_weight = float(dice_weight)
        self.num_points = int(num_points)
        self.oversample_ratio = float(oversample_ratio)
        self.importance_sample_ratio = float(importance_sample_ratio)
        empty_weight = torch.ones(self.num_classes + 1, dtype=torch.float32)
        if semantic_class_weights is not None:
            if len(semantic_class_weights) != self.num_classes:
                raise ValueError(
                    f"Expected {self.num_classes} semantic class weights, "
                    f"got {len(semantic_class_weights)}"
                )
            semantic_weights = torch.tensor(semantic_class_weights, dtype=torch.float32)
            if not torch.isfinite(semantic_weights).all() or torch.any(semantic_weights <= 0):
                raise ValueError("Semantic class weights must be finite and positive")
            empty_weight[: self.num_classes] = semantic_weights
        empty_weight[-1] = float(no_object_weight)
        self.register_buffer("empty_weight", empty_weight)

    def _single_layer(
        self,
        outputs: dict[str, object],
        targets: list[dict[str, object]],
        *,
        generator: torch.Generator | None,
    ) -> dict[str, Tensor]:
        logits = outputs["pred_logits"]
        masks = outputs["pred_masks"]
        if not isinstance(logits, Tensor) or not isinstance(masks, Tensor):
            raise TypeError("Criterion outputs must contain tensors")
        zero = logits.sum() * 0.0 + masks.sum() * 0.0
        assignments = self.matcher(outputs, targets, generator=generator)

        class_losses = []
        mask_losses = []
        dice_losses = []
        for batch_index, (source_indices, target_indices) in enumerate(assignments):
            target = targets[batch_index]
            if bool(target.get("skip_supervision", False)):
                continue
            labels = target["labels"]
            target_masks = target["masks"]
            valid_mask = target["valid_mask"]
            if not isinstance(labels, Tensor) or not isinstance(target_masks, Tensor) or not isinstance(valid_mask, Tensor):
                raise TypeError("Target tensors are required")

            query_targets = torch.full(
                (logits.shape[1],),
                self.num_classes,
                dtype=torch.long,
                device=logits.device,
            )
            query_targets[source_indices] = labels.to(logits.device)[target_indices]
            class_losses.append(
                F.cross_entropy(logits[batch_index].float(), query_targets, weight=self.empty_weight.float())
            )

            if source_indices.numel() == 0:
                continue
            selected_predictions = masks[batch_index, source_indices].float()
            sample = sample_valid_point_coords(
                selected_predictions,
                valid_mask,
                num_points=self.num_points,
                oversample_ratio=self.oversample_ratio,
                importance_sample_ratio=self.importance_sample_ratio,
                generator=generator,
            )
            if sample.flat_indices.numel() == 0:
                continue
            pred_points = point_sample(selected_predictions, sample.coords).float()
            target_points = (
                target_masks.to(device=masks.device, dtype=torch.float32)
                .flatten(1)[target_indices][:, sample.flat_indices]
            )
            mask_losses.append(F.binary_cross_entropy_with_logits(pred_points, target_points))
            dice_losses.append(_matched_dice_loss(pred_points, target_points))

        loss_ce = torch.stack(class_losses).mean() if class_losses else zero
        loss_mask = torch.stack(mask_losses).mean() if mask_losses else zero
        loss_dice = torch.stack(dice_losses).mean() if dice_losses else zero
        return {"loss_ce": loss_ce, "loss_mask": loss_mask, "loss_dice": loss_dice}

    def forward(
        self,
        outputs: dict[str, object],
        targets: list[dict[str, object]],
        *,
        generator: torch.Generator | None = None,
    ) -> dict[str, Tensor]:
        main = self._single_layer(outputs, targets, generator=generator)
        loss_ce_weighted = self.class_weight * main["loss_ce"]
        loss_mask_weighted = self.mask_weight * main["loss_mask"]
        loss_dice_weighted = self.dice_weight * main["loss_dice"]
        loss_aux = loss_ce_weighted.new_zeros(())
        auxiliary = outputs.get("aux_outputs", [])
        if not isinstance(auxiliary, list):
            raise TypeError("aux_outputs must be a list")
        for auxiliary_output in auxiliary:
            layer = self._single_layer(auxiliary_output, targets, generator=generator)
            loss_aux = loss_aux + (
                self.class_weight * layer["loss_ce"]
                + self.mask_weight * layer["loss_mask"]
                + self.dice_weight * layer["loss_dice"]
            )
        loss_total = loss_ce_weighted + loss_mask_weighted + loss_dice_weighted + loss_aux
        return {
            **main,
            "loss_ce_weighted": loss_ce_weighted,
            "loss_mask_weighted": loss_mask_weighted,
            "loss_dice_weighted": loss_dice_weighted,
            "loss_aux": loss_aux,
            "loss_total": loss_total,
        }
