from __future__ import annotations

from torch import nn


class DINOv3Mask2Former(nn.Module):
    def __init__(self, adapter: nn.Module, decoder: nn.Module) -> None:
        super().__init__()
        self.adapter = adapter
        self.decoder = decoder

    def train(self, mode: bool = True):
        super().train(mode)
        backbone = getattr(self.adapter, "backbone", None)
        if backbone is None:
            raise AttributeError("Adapter must expose its frozen backbone")
        backbone.eval()
        return self

    def forward(self, images):
        features = self.adapter(images)
        return self.decoder(features)
