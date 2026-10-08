from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from aic_dino_m2f.engine.checkpoint import (
    capture_rng_state,
    load_initialization_checkpoint,
    load_training_checkpoint,
    restore_rng_state,
    save_training_checkpoint,
)
from aic_dino_m2f.engine.optim import PolynomialWarmupScheduler, build_adamw
from aic_dino_m2f.engine.state import EarlyStoppingState


class _CheckpointModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Linear(3, 3)
        self.backbone.requires_grad_(False)
        self.adapter = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.adapter(self.backbone(values))


def _objects():
    model = _CheckpointModel()
    optimizer = build_adamw(model, learning_rate=1.0e-3, weight_decay=0.05)
    scheduler = PolynomialWarmupScheduler(
        optimizer,
        total_steps=5,
        warmup_steps=0,
        warmup_start_factor=0.001,
        power=0.9,
        min_learning_rate=1.0e-6,
    )
    return model, optimizer, scheduler


def _step(model, optimizer, scheduler, values):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = model(values).square().mean()
    loss.backward()
    optimizer.step()
    scheduler.step()


def test_early_stopping_is_strict_and_equality_consumes_patience() -> None:
    state = EarlyStoppingState(patience=2)
    assert state.update(0.4, epoch=1)
    assert not state.update(0.4, epoch=2)
    assert state.bad_epochs == 1 and not state.should_stop
    assert not state.update(0.3, epoch=3)
    assert state.should_stop
    assert state.best_score == 0.4 and state.best_epoch == 1


def test_checkpoint_round_trip_preserves_buffers_and_exact_next_step(tmp_path: Path) -> None:
    torch.manual_seed(11)
    model, optimizer, scheduler = _objects()
    values = torch.randn(4, 3)
    _step(model, optimizer, scheduler, values)
    model.eval()
    expected_prediction = model(values).detach().clone()
    metadata = {"backbone_sha256": "abc", "split_sha256": "def", "effective_batch": 8}
    checkpoint = tmp_path / "last.pth"
    save_training_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        epoch=1,
        global_step=1,
        early_stopping=EarlyStoppingState(patience=20, best_score=0.2, best_epoch=1),
        metadata=metadata,
        include_optimizer=True,
    )
    assert checkpoint.is_file()
    restored, restored_optimizer, restored_scheduler = _objects()
    restored.backbone.load_state_dict(model.backbone.state_dict())
    payload = load_training_checkpoint(
        checkpoint,
        model=restored,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        scaler=None,
        expected_metadata=metadata,
        restore_rng=True,
    )
    restored.eval()
    torch.testing.assert_close(restored(values), expected_prediction)
    assert payload["epoch"] == 1 and payload["global_step"] == 1

    next_values = torch.full((4, 3), 0.25)
    _step(model, optimizer, scheduler, next_values)
    _step(restored, restored_optimizer, restored_scheduler, next_values)
    for original, resumed in zip(model.adapter.state_dict().values(), restored.adapter.state_dict().values()):
        torch.testing.assert_close(original, resumed)


def test_resume_rejects_incompatible_metadata(tmp_path: Path) -> None:
    model, optimizer, scheduler = _objects()
    checkpoint = tmp_path / "last.pth"
    save_training_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        epoch=0,
        global_step=0,
        early_stopping=EarlyStoppingState(patience=20),
        metadata={"split_sha256": "expected"},
        include_optimizer=True,
    )
    with pytest.raises(ValueError, match="metadata"):
        load_training_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            expected_metadata={"split_sha256": "different"},
            restore_rng=False,
        )


def test_initialization_checkpoint_allows_config_change_but_checks_structure(tmp_path: Path) -> None:
    model, _, _ = _objects()
    checkpoint = tmp_path / "best.pth"
    source_metadata = {
        "config_sha256": "baseline",
        "backbone_identity": "dinov3",
        "backbone_sha256": "abc",
        "train_split_sha256": "train",
        "val_split_sha256": "val",
        "label_mapping": {"0": 255},
        "input_size": 1024,
        "effective_batch": 8,
        "architecture": "m2f",
    }
    save_training_checkpoint(
        checkpoint,
        model=model,
        optimizer=None,
        scheduler=None,
        scaler=None,
        epoch=34,
        global_step=23_800,
        early_stopping=EarlyStoppingState(
            patience=20, best_score=0.8013101193614892, best_epoch=34
        ),
        metadata=source_metadata,
        include_optimizer=False,
    )
    restored, _, _ = _objects()
    target_metadata = {**source_metadata, "config_sha256": "barren-v2"}

    payload = load_initialization_checkpoint(
        checkpoint,
        model=restored,
        expected_metadata=target_metadata,
    )

    assert payload["epoch"] == 34
    assert payload["global_step"] == 23_800
    mismatched = {**target_metadata, "input_size": 512}
    with pytest.raises(ValueError, match="input_size"):
        load_initialization_checkpoint(
            checkpoint,
            model=restored,
            expected_metadata=mismatched,
        )


def test_rng_capture_restore_covers_python_numpy_and_torch() -> None:
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    state = capture_rng_state()
    expected = (random.random(), float(np.random.rand()), float(torch.rand(())))
    restore_rng_state(state)
    actual = (random.random(), float(np.random.rand()), float(torch.rand(())))
    assert actual == pytest.approx(expected)
