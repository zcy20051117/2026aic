from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn


class FrozenBackboneGuard(nn.Module):
    """Minimal composition that proves and enforces frozen-backbone semantics."""

    def __init__(self, backbone: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.backbone.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, inputs: Tensor) -> Tensor:
        with torch.no_grad():
            features = self.backbone(inputs)
        return self.head(features)


def load_frozen_dinov3_vitl16(weights: Path) -> nn.Module:
    from dinov3.hub.backbones import dinov3_vitl16

    resolved = Path(weights).resolve(strict=True)
    backbone = dinov3_vitl16(pretrained=True, weights=str(resolved), check_hash=True)
    backbone.requires_grad_(False)
    backbone.eval()
    return backbone
