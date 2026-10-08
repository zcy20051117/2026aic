from __future__ import annotations

import torch
from torch import nn

from aic_dino_m2f.engine.validate import validate_fold0


class _PerfectEightClassModel(nn.Module):
    def forward(self, images: torch.Tensor) -> dict[str, object]:
        logits = torch.full((1, 8, 9), -12.0, device=images.device)
        masks = torch.full((1, 8, 2, 4), -12.0, device=images.device)
        for class_index in range(8):
            logits[0, class_index, class_index] = 12.0
            masks[0, class_index].flatten()[class_index] = 12.0
        return {"pred_logits": logits, "pred_masks": masks, "aux_outputs": []}


class _SeededCriterion(nn.Module):
    def forward(
        self,
        outputs: dict[str, object],
        targets: list[dict[str, object]],
        *,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        del outputs, targets
        sampled = torch.rand((), generator=generator)
        return {
            "loss_ce": sampled,
            "loss_mask": sampled + 1.0,
            "loss_dice": sampled + 2.0,
            "loss_ce_weighted": sampled + 3.0,
            "loss_mask_weighted": sampled + 4.0,
            "loss_dice_weighted": sampled + 5.0,
            "loss_aux": sampled + 6.0,
            "loss_total": sampled + 7.0,
        }


def _loader() -> list[dict[str, object]]:
    return [
        {
            "image": torch.zeros((1, 3, 2, 4)),
            "semantic": torch.arange(8).reshape(1, 2, 4),
            "target": [{}],
        }
    ]


def test_fold0_validation_reports_reproducible_loss_components() -> None:
    first, first_confusion = validate_fold0(
        model=_PerfectEightClassModel(),
        criterion=_SeededCriterion(),
        loader=_loader(),
        device=torch.device("cpu"),
        expected_samples=1,
        query_chunk=8,
        loss_seed=20260918,
    )
    second, second_confusion = validate_fold0(
        model=_PerfectEightClassModel(),
        criterion=_SeededCriterion(),
        loader=_loader(),
        device=torch.device("cpu"),
        expected_samples=1,
        query_chunk=8,
        loss_seed=20260918,
    )

    assert first_confusion.tolist() == torch.eye(8, dtype=torch.int64).tolist()
    assert second_confusion.tolist() == first_confusion.tolist()
    for key in (
        "loss_ce",
        "loss_mask",
        "loss_dice",
        "loss_ce_weighted",
        "loss_mask_weighted",
        "loss_dice_weighted",
        "loss_aux",
        "loss_total",
    ):
        assert first[key] == second[key]
