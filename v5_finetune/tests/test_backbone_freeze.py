from __future__ import annotations

import torch
from torch import nn

from aic_dino_m2f.models.backbone import FrozenBackboneGuard


class _TrackingBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(4, 4)
        self.last_grad_enabled: bool | None = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.last_grad_enabled = torch.is_grad_enabled()
        return self.projection(inputs)


def test_parent_train_keeps_backbone_eval_no_grad_and_out_of_optimizer() -> None:
    backbone = _TrackingBackbone()
    head = nn.Linear(4, 2)
    model = FrozenBackboneGuard(backbone, head)
    model.train()

    assert model.training
    assert model.head.training
    assert not model.backbone.training
    assert all(not parameter.requires_grad for parameter in model.backbone.parameters())

    output = model(torch.randn(3, 4)).sum()
    output.backward()
    assert backbone.last_grad_enabled is False
    assert all(parameter.grad is None for parameter in backbone.parameters())
    assert any(parameter.grad is not None for parameter in head.parameters())

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad), lr=1e-4
    )
    optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    assert not optimizer_ids & {id(parameter) for parameter in backbone.parameters()}
