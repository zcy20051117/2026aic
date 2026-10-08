from __future__ import annotations

import torch
from torch import nn

from aic_dino_m2f.models.model import DINOv3Mask2Former


class _Adapter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
        self.projection = nn.Linear(2, 2)
        self.backbone.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, inputs):
        with torch.no_grad():
            frozen = self.backbone(inputs)
        return self.projection(frozen)


def test_combined_model_train_keeps_only_backbone_frozen() -> None:
    adapter = _Adapter()
    decoder = nn.Linear(2, 1)
    model = DINOv3Mask2Former(adapter, decoder)
    model.train()
    assert not adapter.backbone.training
    assert adapter.projection.training
    assert decoder.training
    assert all(not parameter.requires_grad for parameter in adapter.backbone.parameters())
    assert all(parameter.requires_grad for parameter in adapter.projection.parameters())
    assert all(parameter.requires_grad for parameter in decoder.parameters())

    model(torch.randn((4, 2))).sum().backward()
    assert all(parameter.grad is None for parameter in adapter.backbone.parameters())
    assert all(parameter.grad is not None for parameter in adapter.projection.parameters())
    assert all(parameter.grad is not None for parameter in decoder.parameters())
