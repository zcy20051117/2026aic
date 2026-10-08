from __future__ import annotations

import torch
from torch import nn

from aic_dino_m2f.models.adapter import TrainableDINOv3Adapter


class _FakeDINO(nn.Module):
    def __init__(self, embed_dim: int = 32, patch_size: int = 16) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.patch_embed = nn.Conv2d(3, embed_dim, patch_size, patch_size)

    def get_intermediate_layers(self, images, n, return_class_token):
        features = self.patch_embed(images).flatten(2).transpose(1, 2)
        class_token = features.mean(dim=1, keepdim=True)
        return [(features + float(index) / 100.0, class_token) for index in range(len(n))]


def test_adapter_outputs_true_stride_pyramid_and_backpropagates_only_downstream() -> None:
    backbone = _FakeDINO()
    adapter = TrainableDINOv3Adapter(
        backbone,
        interaction_indexes=[4, 11, 17, 23],
        pretrain_size=64,
        conv_inplane=8,
        deform_num_heads=4,
        drop_path_rate=0.0,
        with_cp=False,
        convert_sync_batchnorm=True,
    )
    adapter.train()
    outputs = adapter(torch.randn((1, 3, 64, 64)))
    assert {key: tuple(value.shape) for key, value in outputs.items()} == {
        "1": (1, 32, 16, 16),
        "2": (1, 32, 8, 8),
        "3": (1, 32, 4, 4),
        "4": (1, 32, 2, 2),
    }
    assert not adapter.backbone.training
    assert not any(isinstance(module, nn.SyncBatchNorm) for module in adapter.modules())

    sum(value.float().mean() for value in outputs.values()).backward()
    assert all(parameter.grad is None for parameter in backbone.parameters())
    downstream_gradients = [
        parameter.grad
        for name, parameter in adapter.named_parameters()
        if not name.startswith("backbone.") and parameter.requires_grad
    ]
    assert any(gradient is not None and torch.isfinite(gradient).all() for gradient in downstream_gradients)
