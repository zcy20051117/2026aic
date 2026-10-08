from __future__ import annotations

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn

from .point_sampling import point_sample, sample_valid_point_coords


def _pairwise_sigmoid_bce_cost(inputs: Tensor, targets: Tensor) -> Tensor:
    point_count = max(inputs.shape[1], 1)
    positive = F.binary_cross_entropy_with_logits(inputs, torch.ones_like(inputs), reduction="none")
    negative = F.binary_cross_entropy_with_logits(inputs, torch.zeros_like(inputs), reduction="none")
    return (positive @ targets.t() + negative @ (1.0 - targets).t()) / point_count


def _pairwise_dice_cost(inputs: Tensor, targets: Tensor) -> Tensor:
    probabilities = inputs.sigmoid()
    numerator = 2.0 * (probabilities @ targets.t())
    denominator = probabilities.sum(-1, keepdim=True) + targets.sum(-1).unsqueeze(0)
    return 1.0 - (numerator + 1.0) / (denominator + 1.0)


class HungarianMatcher(nn.Module):
    def __init__(
        self,
        *,
        class_cost: float,
        mask_cost: float,
        dice_cost: float,
        num_points: int,
        oversample_ratio: float,
        importance_sample_ratio: float,
    ) -> None:
        super().__init__()
        if class_cost == mask_cost == dice_cost == 0:
            raise ValueError("At least one matcher cost must be non-zero")
        self.class_cost = float(class_cost)
        self.mask_cost = float(mask_cost)
        self.dice_cost = float(dice_cost)
        self.num_points = int(num_points)
        self.oversample_ratio = float(oversample_ratio)
        self.importance_sample_ratio = float(importance_sample_ratio)

    @torch.no_grad()
    def forward(
        self,
        outputs: dict[str, object],
        targets: list[dict[str, object]],
        *,
        generator: torch.Generator | None = None,
    ) -> list[tuple[Tensor, Tensor]]:
        logits = outputs["pred_logits"]
        masks = outputs["pred_masks"]
        if not isinstance(logits, Tensor) or not isinstance(masks, Tensor):
            raise TypeError("Matcher outputs must contain tensor pred_logits and pred_masks")
        if logits.shape[0] != len(targets):
            raise ValueError("Target batch length does not match outputs")
        probabilities = logits.float().softmax(-1)
        assignments: list[tuple[Tensor, Tensor]] = []
        for batch_index, target in enumerate(targets):
            labels = target["labels"]
            target_masks = target["masks"]
            valid_mask = target["valid_mask"]
            if not isinstance(labels, Tensor) or not isinstance(target_masks, Tensor) or not isinstance(valid_mask, Tensor):
                raise TypeError("Target labels, masks, and valid_mask must be tensors")
            if bool(target.get("skip_supervision", False)) or labels.numel() == 0:
                empty = torch.empty(0, dtype=torch.int64, device=logits.device)
                assignments.append((empty, empty.clone()))
                continue

            sample = sample_valid_point_coords(
                masks[batch_index],
                valid_mask,
                num_points=self.num_points,
                oversample_ratio=self.oversample_ratio,
                importance_sample_ratio=self.importance_sample_ratio,
                generator=generator,
            )
            pred_points = point_sample(masks[batch_index].float(), sample.coords).float()
            target_points = target_masks.to(device=masks.device, dtype=torch.float32).flatten(1)[:, sample.flat_indices]
            class_cost = -probabilities[batch_index][:, labels.to(logits.device)]
            mask_cost = _pairwise_sigmoid_bce_cost(pred_points, target_points)
            dice_cost = _pairwise_dice_cost(pred_points, target_points)
            total_cost = (
                self.class_cost * class_cost
                + self.mask_cost * mask_cost
                + self.dice_cost * dice_cost
            )
            source, destination = linear_sum_assignment(total_cost.cpu().numpy())
            assignments.append(
                (
                    torch.as_tensor(source, dtype=torch.int64, device=logits.device),
                    torch.as_tensor(destination, dtype=torch.int64, device=logits.device),
                )
            )
        return assignments
