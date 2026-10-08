from __future__ import annotations

import pytest
import torch
from torch import nn

from aic_dino_m2f.engine import optim
from aic_dino_m2f.engine.optim import (
    AccumulationController,
    PolynomialWarmupScheduler,
    build_adamw,
)


class _ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Linear(3, 3)
        self.backbone.requires_grad_(False)
        self.proj = nn.Linear(3, 4)
        self.norm = nn.LayerNorm(4)
        self.query_embed = nn.Embedding(5, 4)


def test_optimizer_contains_each_trainable_parameter_once_with_explicit_decay() -> None:
    model = _ToyModel()
    optimizer = build_adamw(model, learning_rate=1.0e-4, weight_decay=0.05)
    grouped = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    expected = [parameter for parameter in model.parameters() if parameter.requires_grad]
    assert len(grouped) == len({id(parameter) for parameter in grouped}) == len(expected)
    assert {id(parameter) for parameter in grouped} == {id(parameter) for parameter in expected}
    decay_by_id = {
        id(parameter): float(group["weight_decay"])
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert decay_by_id[id(model.proj.weight)] == pytest.approx(0.05)
    assert decay_by_id[id(model.proj.bias)] == 0.0
    assert decay_by_id[id(model.norm.weight)] == 0.0
    assert decay_by_id[id(model.query_embed.weight)] == 0.0
    assert all(id(parameter) not in decay_by_id for parameter in model.backbone.parameters())


def test_step_scheduler_uses_warmup_then_polynomial_floor() -> None:
    model = nn.Linear(2, 1)
    optimizer = build_adamw(model, learning_rate=1.0, weight_decay=0.0)
    scheduler = PolynomialWarmupScheduler(
        optimizer,
        total_steps=10,
        warmup_steps=2,
        warmup_start_factor=0.1,
        power=1.0,
        min_learning_rate=0.2,
    )
    assert scheduler.get_last_lr()[0] == pytest.approx(0.1)
    scheduler.step()
    assert scheduler.get_last_lr()[0] == pytest.approx(0.55)
    scheduler.step()
    assert scheduler.get_last_lr()[0] == pytest.approx(1.0)
    for _ in range(8):
        scheduler.step()
    assert scheduler.get_last_lr()[0] == pytest.approx(0.2)


def test_scheduler_can_continue_from_best_checkpoint_global_step() -> None:
    model = nn.Linear(2, 1)
    optimizer = build_adamw(model, learning_rate=1.0e-4, weight_decay=0.0)
    scheduler = PolynomialWarmupScheduler(
        optimizer,
        total_steps=700 * 150,
        warmup_steps=700 * 3,
        warmup_start_factor=0.001,
        power=0.9,
        min_learning_rate=1.0e-6,
    )

    scheduler.set_step_count(23_800)

    assert scheduler.step_count == 23_800
    assert scheduler.get_last_lr()[0] == pytest.approx(8.080283324963005e-5)


def test_formal_scheduler_steps_use_configured_dataset_size_not_smoke_subset() -> None:
    assert hasattr(optim, "formal_optimizer_steps_per_epoch")
    assert optim.formal_optimizer_steps_per_epoch(
        sample_count=5597,
        physical_batch_size=2,
        accumulation_steps=4,
    ) == 700


def test_tail_accumulation_window_is_scaled_by_its_actual_size() -> None:
    controller = AccumulationController(accumulation_steps=2, total_microbatches=5)
    assert [controller.window_size(index) for index in range(5)] == [2, 2, 2, 2, 1]
    assert [controller.should_step(index) for index in range(5)] == [False, True, False, True, True]
    losses = [torch.tensor(float(index + 1)) for index in range(5)]
    scaled = [controller.scale_loss(loss, index).item() for index, loss in enumerate(losses)]
    assert scaled == pytest.approx([0.5, 1.0, 1.5, 2.0, 5.0])
