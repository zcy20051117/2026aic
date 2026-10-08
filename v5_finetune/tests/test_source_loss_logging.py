from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

from aic_dino_m2f.engine.train_epoch import collate_samples, train_one_epoch


class _Loss(nn.Module):
    def forward(self, output, targets):
        return {"loss_total": output.square().mean()}


class _Scheduler:
    def step(self):
        pass


def test_training_metrics_report_actual_weighted_hard_draw_fraction() -> None:
    samples = [
        {"image": torch.ones(1, 1, 1), "semantic": torch.zeros(1, 1),
         "filename": "a.png", "target": {}, "source": "aic"},
        {"image": torch.ones(1, 1, 1), "semantic": torch.zeros(1, 1),
         "filename": "a.png", "target": {}, "source": "hard"},
        {"image": torch.ones(1, 1, 1), "semantic": torch.zeros(1, 1),
         "filename": "0.png", "target": {}, "source": "loveda_rural"},
    ]
    loader = DataLoader(samples, batch_size=1, collate_fn=collate_samples)
    model = nn.Conv2d(1, 1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    metrics, steps = train_one_epoch(
        model=model, criterion=_Loss(), loader=loader, optimizer=optimizer,
        scheduler=_Scheduler(), device=torch.device("cpu"), accumulation_steps=1,
        epoch=1, max_epochs=1, global_step=0, use_bf16=False, gradient_clip_norm=0.0,
    )

    assert steps == 3
    assert metrics["samples"] == 3
    assert metrics["hard_samples"] == 1
    assert metrics["hard_fraction"] == 1 / 3
    assert metrics["loveda_samples"] == 1
    assert metrics["loveda_fraction"] == 1 / 3
    assert metrics["aic_samples"] == 1
    assert metrics["aic_fraction"] == 1 / 3
    assert metrics["loss_total"] >= 0
