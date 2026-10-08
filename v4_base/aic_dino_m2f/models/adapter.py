from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from dinov3.eval.segmentation.models.backbone.dinov3_adapter import (
    DINOv3_Adapter,
)

from .deformable_attention import patch_ms_deform_attn_modules


def _reference_points(spatial_shapes: list[tuple[int, int]], device: torch.device) -> Tensor:
    levels = []
    for height, width in spatial_shapes:
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, height - 0.5, height, dtype=torch.float32, device=device),
            torch.linspace(0.5, width - 0.5, width, dtype=torch.float32, device=device),
            indexing="ij",
        )
        levels.append(
            torch.stack((ref_x.reshape(-1)[None] / width, ref_y.reshape(-1)[None] / height), dim=-1)
        )
    return torch.cat(levels, dim=1)[:, :, None]


def _deform_inputs(images: Tensor, patch_size: int):
    height, width = images.shape[-2:]
    spatial_shapes = torch.as_tensor(
        [(height // 8, width // 8), (height // 16, width // 16), (height // 32, width // 32)],
        dtype=torch.long,
        device=images.device,
    )
    starts = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    first = [
        _reference_points([(height // patch_size, width // patch_size)], images.device),
        spatial_shapes,
        starts,
    ]
    patch_shapes = torch.as_tensor(
        [(height // patch_size, width // patch_size)], dtype=torch.long, device=images.device
    )
    patch_starts = torch.cat((patch_shapes.new_zeros((1,)), patch_shapes.prod(1).cumsum(0)[:-1]))
    second = [
        _reference_points(
            [(height // 8, width // 8), (height // 16, width // 16), (height // 32, width // 32)],
            images.device,
        ),
        patch_shapes,
        patch_starts,
    ]
    return first, second


def _sync_batchnorm_to_batchnorm(module: nn.Module) -> nn.Module:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.SyncBatchNorm):
            replacement = nn.BatchNorm2d(
                child.num_features,
                eps=child.eps,
                momentum=child.momentum,
                affine=child.affine,
                track_running_stats=child.track_running_stats,
            )
            if child.affine:
                replacement.weight.data.copy_(child.weight.data)
                replacement.bias.data.copy_(child.bias.data)
            if child.track_running_stats:
                replacement.running_mean.data.copy_(child.running_mean.data)
                replacement.running_var.data.copy_(child.running_var.data)
                replacement.num_batches_tracked.data.copy_(child.num_batches_tracked.data)
            setattr(module, name, replacement)
        else:
            _sync_batchnorm_to_batchnorm(child)
    return module


class TrainableDINOv3Adapter(DINOv3_Adapter):
    """Hash-pinned official adapter with safe mode and differentiable attention."""

    def __init__(self, backbone: nn.Module, *, convert_sync_batchnorm: bool = True, **kwargs) -> None:
        super().__init__(backbone, **kwargs)
        self.deformable_modules_patched = patch_ms_deform_attn_modules(self)
        if convert_sync_batchnorm:
            _sync_batchnorm_to_batchnorm(self)
        self.backbone.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, images: Tensor) -> dict[str, Tensor]:
        deform_inputs1, deform_inputs2 = _deform_inputs(images, self.patch_size)
        c1, c2, c3, c4 = self.spm(images)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        combined = torch.cat((c2, c3, c4), dim=1)

        patch_height = images.shape[2] // self.patch_size
        patch_width = images.shape[3] // self.patch_size
        batch = images.shape[0]
        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if images.is_cuda
            else nullcontext()
        )
        with autocast_context:
            with torch.no_grad():
                all_layers = self.backbone.get_intermediate_layers(
                    images,
                    n=self.interaction_indexes,
                    return_class_token=True,
                )
        if len(all_layers) != len(self.interactions):
            raise RuntimeError("Backbone intermediate layer count does not match adapter interactions")
        embed_dim = all_layers[0][0].shape[-1]

        outputs = []
        for index, interaction in enumerate(self.interactions):
            tokens, class_token = all_layers[index]
            _, combined, _ = interaction(
                tokens,
                combined,
                class_token,
                deform_inputs1,
                deform_inputs2,
                patch_height,
                patch_width,
                patch_height,
                patch_width,
            )
            outputs.append(
                tokens.transpose(1, 2)
                .reshape(batch, embed_dim, patch_height, patch_width)
                .contiguous()
            )

        c2_length, c3_length = c2.shape[1], c3.shape[1]
        c2 = combined[:, :c2_length]
        c3 = combined[:, c2_length : c2_length + c3_length]
        c4 = combined[:, c2_length + c3_length :]
        c2 = c2.transpose(1, 2).reshape(batch, embed_dim, patch_height * 2, patch_width * 2)
        c3 = c3.transpose(1, 2).reshape(batch, embed_dim, patch_height, patch_width)
        c4 = c4.transpose(1, 2).reshape(batch, embed_dim, patch_height // 2, patch_width // 2)
        c1 = self.up(c2) + c1

        if self.add_vit_feature:
            sizes = (
                (patch_height * 4, patch_width * 4),
                (patch_height * 2, patch_width * 2),
                (patch_height, patch_width),
                (patch_height // 2, patch_width // 2),
            )
            resized = [
                F.interpolate(feature, size=size, mode="bilinear", align_corners=False)
                for feature, size in zip(outputs, sizes, strict=True)
            ]
            c1, c2, c3, c4 = (
                c1 + resized[0],
                c2 + resized[1],
                c3 + resized[2],
                c4 + resized[3],
            )
        return {
            "1": self.norm1(c1),
            "2": self.norm2(c2),
            "3": self.norm3(c3),
            "4": self.norm4(c4),
        }
