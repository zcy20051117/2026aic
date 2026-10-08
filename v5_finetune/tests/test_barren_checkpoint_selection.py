from __future__ import annotations

from aic_dino_m2f.train import BarrenCheckpointState


def _metrics(miou: float, barren_iou: float) -> dict[str, object]:
    return {
        "miou": miou,
        "per_class_iou_by_name": {"Barren": barren_iou},
    }


def test_barren_checkpoint_requires_miou_floor_and_strict_iou_improvement() -> None:
    state = BarrenCheckpointState(min_miou=0.792)

    assert not state.update(_metrics(0.7919, 0.60), epoch=40)
    assert state.best_epoch == -1
    assert state.update(_metrics(0.8070, 0.5472), epoch=40)
    assert state.best_epoch == 40
    assert state.best_iou == 0.5472
    assert not state.update(_metrics(0.8100, 0.5472), epoch=41)
    assert not state.update(_metrics(0.7910, 0.60), epoch=42)
    assert state.update(_metrics(0.7920, 0.56), epoch=43)
    assert state.best_epoch == 43
