from __future__ import annotations

import copy

import torch

from aic_dino_m2f.losses.criterion import SetCriterion
from aic_dino_m2f.losses.matcher import HungarianMatcher


def _target() -> dict[str, object]:
    valid = torch.ones((4, 4), dtype=torch.bool)
    valid[0, 0] = False
    masks = torch.zeros((2, 4, 4), dtype=torch.bool)
    masks[0, :, :2] = True
    masks[1, :, 2:] = True
    return {
        "labels": torch.tensor([0, 1]),
        "masks": masks,
        "valid_mask": valid,
        "skip_supervision": False,
    }


def _outputs() -> dict[str, object]:
    class_logits = torch.full((1, 3, 9), -2.0)
    class_logits[0, 0, 0] = 4.0
    class_logits[0, 1, 1] = 4.0
    class_logits[0, 2, 8] = 4.0
    masks = torch.zeros((1, 3, 4, 4))
    masks[0, 0, :, :2] = 3.0
    masks[0, 0, :, 2:] = -3.0
    masks[0, 1, :, :2] = -3.0
    masks[0, 1, :, 2:] = 3.0
    return {"pred_logits": class_logits, "pred_masks": masks, "aux_outputs": []}


def _matcher() -> HungarianMatcher:
    return HungarianMatcher(
        class_cost=2.0,
        mask_cost=5.0,
        dice_cost=5.0,
        num_points=128,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
    )


def test_ignore_prediction_changes_do_not_change_matching_or_mask_losses() -> None:
    first = _outputs()
    second = copy.deepcopy(first)
    second["pred_masks"][:, :, 0, 0] = torch.tensor([100.0, -100.0, 50.0])
    target = _target()
    matcher = _matcher()
    criterion = SetCriterion(
        num_classes=8,
        matcher=matcher,
        class_weight=2.0,
        mask_weight=5.0,
        dice_weight=5.0,
        no_object_weight=0.1,
        num_points=128,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
    )

    seed = 19
    match_a = matcher(first, [target], generator=torch.Generator().manual_seed(seed))
    match_b = matcher(second, [target], generator=torch.Generator().manual_seed(seed))
    assert torch.equal(match_a[0][0], match_b[0][0])
    assert torch.equal(match_a[0][1], match_b[0][1])

    loss_a = criterion(first, [target], generator=torch.Generator().manual_seed(seed))
    loss_b = criterion(second, [target], generator=torch.Generator().manual_seed(seed))
    torch.testing.assert_close(loss_a["loss_mask"], loss_b["loss_mask"])
    torch.testing.assert_close(loss_a["loss_dice"], loss_b["loss_dice"])


def test_all_ignore_image_skips_query_class_and_mask_supervision() -> None:
    target = {
        "labels": torch.empty(0, dtype=torch.long),
        "masks": torch.empty((0, 4, 4), dtype=torch.bool),
        "valid_mask": torch.zeros((4, 4), dtype=torch.bool),
        "skip_supervision": True,
    }
    criterion = SetCriterion(
        num_classes=8,
        matcher=_matcher(),
        class_weight=2.0,
        mask_weight=5.0,
        dice_weight=5.0,
        no_object_weight=0.1,
        num_points=32,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
    )
    losses = criterion(_outputs(), [target], generator=torch.Generator().manual_seed(5))
    for name in ("loss_ce", "loss_mask", "loss_dice", "loss_total"):
        assert losses[name].item() == 0.0


def test_auxiliary_outputs_use_same_valid_filter_and_contribute_loss() -> None:
    outputs = _outputs()
    outputs["aux_outputs"] = [
        {
            "pred_logits": outputs["pred_logits"].clone(),
            "pred_masks": outputs["pred_masks"].clone(),
        }
    ]
    criterion = SetCriterion(
        num_classes=8,
        matcher=_matcher(),
        class_weight=2.0,
        mask_weight=5.0,
        dice_weight=5.0,
        no_object_weight=0.1,
        num_points=64,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
    )
    losses = criterion(outputs, [_target()], generator=torch.Generator().manual_seed(3))
    assert losses["loss_aux"].isfinite()
    assert losses["loss_aux"].item() > 0
    torch.testing.assert_close(
        losses["loss_total"],
        losses["loss_ce_weighted"]
        + losses["loss_mask_weighted"]
        + losses["loss_dice_weighted"]
        + losses["loss_aux"],
    )


def test_semantic_class_weights_raise_only_barren_classification_penalty() -> None:
    semantic_weights = [1.0, 1.0, 1.0, 1.0, 1.5, 1.0, 1.0, 1.0]
    criterion = SetCriterion(
        num_classes=8,
        matcher=_matcher(),
        class_weight=2.0,
        mask_weight=5.0,
        dice_weight=5.0,
        semantic_class_weights=semantic_weights,
        no_object_weight=0.1,
        num_points=32,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
    )

    torch.testing.assert_close(
        criterion.empty_weight,
        torch.tensor([1.0, 1.0, 1.0, 1.0, 1.5, 1.0, 1.0, 1.0, 0.1]),
    )
