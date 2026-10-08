from __future__ import annotations

from typing import Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from dinov3.eval.segmentation.models.heads.pixel_decoder import (
    MSDeformAttnTransformerEncoderOnly,
)
from dinov3.eval.segmentation.models.utils.position_encoding import PositionEmbeddingSine

from .deformable_attention import patch_ms_deform_attn_modules


class CorrectedMSDeformAttnPixelDecoder(nn.Module):
    """Three-level deformable encoder plus one stride-4 FPN fusion branch."""

    def __init__(
        self,
        *,
        input_channels: int,
        conv_dim: int,
        mask_dim: int,
        encoder_layers: int,
        attention_heads: int,
        ffn_dim: int,
        dropout: float,
        common_stride: int,
    ) -> None:
        super().__init__()
        if common_stride != 4:
            raise ValueError("Baseline v1 requires common_stride=4")
        self.in_features = ["1", "2", "3", "4"]
        self.feature_strides = [4, 8, 16, 32]
        self.transformer_in_features = ["2", "3", "4"]
        self.transformer_feature_strides = [8, 16, 32]
        self.transformer_num_feature_levels = 3
        self.maskformer_num_feature_levels = 3
        self.common_stride = common_stride

        self.input_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(input_channels, conv_dim, kernel_size=1),
                    nn.GroupNorm(32, conv_dim),
                )
                for _ in range(self.transformer_num_feature_levels)
            ]
        )
        self.encoder = MSDeformAttnTransformerEncoderOnly(
            d_model=conv_dim,
            dropout=dropout,
            nhead=attention_heads,
            dim_feedforward=ffn_dim,
            num_encoder_layers=encoder_layers,
            num_feature_levels=self.transformer_num_feature_levels,
        )
        self.pe_layer = PositionEmbeddingSine(conv_dim // 2, normalize=True)
        self.lateral_stride4 = nn.Sequential(
            nn.Conv2d(input_channels, conv_dim, kernel_size=1),
            nn.GroupNorm(32, conv_dim),
        )
        self.output_stride4 = nn.Sequential(
            nn.Conv2d(conv_dim, conv_dim, kernel_size=3, padding=1),
            nn.GroupNorm(32, conv_dim),
            nn.ReLU(inplace=True),
        )
        self.mask_feature = nn.Conv2d(conv_dim, mask_dim, kernel_size=1)
        self._reset_parameters()
        self.deformable_modules_patched = patch_ms_deform_attn_modules(self.encoder)

    def _reset_parameters(self) -> None:
        for module in (*self.input_convs, self.lateral_stride4, self.output_stride4):
            for child in module.modules():
                if isinstance(child, nn.Conv2d):
                    nn.init.xavier_uniform_(child.weight)
                    if child.bias is not None:
                        nn.init.zeros_(child.bias)
        nn.init.xavier_uniform_(self.mask_feature.weight)
        if self.mask_feature.bias is not None:
            nn.init.zeros_(self.mask_feature.bias)

    def forward_features(
        self, features: Mapping[str, Tensor]
    ) -> tuple[Tensor, Tensor, list[Tensor]]:
        missing = [name for name in self.in_features if name not in features]
        if missing:
            raise KeyError(f"Missing adapter feature levels: {missing}")
        sources = []
        positions = []
        for index, name in enumerate(reversed(self.transformer_in_features)):
            feature = features[name].float()
            sources.append(self.input_convs[index](feature))
            positions.append(self.pe_layer(feature))
        encoded, spatial_shapes, level_start_index = self.encoder(sources, positions)
        split_sizes = []
        for level in range(self.transformer_num_feature_levels):
            if level + 1 < self.transformer_num_feature_levels:
                split_sizes.append(int((level_start_index[level + 1] - level_start_index[level]).item()))
            else:
                split_sizes.append(int(encoded.shape[1] - level_start_index[level].item()))
        chunks = torch.split(encoded, split_sizes, dim=1)
        batch = encoded.shape[0]
        multi_scale = [
            chunk.transpose(1, 2)
            .reshape(batch, -1, int(spatial_shapes[index, 0]), int(spatial_shapes[index, 1]))
            .contiguous()
            for index, chunk in enumerate(chunks)
        ]
        stride4 = self.lateral_stride4(features["1"].float())
        stride4 = stride4 + F.interpolate(
            multi_scale[-1], size=stride4.shape[-2:], mode="bilinear", align_corners=False
        )
        stride4 = self.output_stride4(stride4)
        return self.mask_feature(stride4), multi_scale[0], multi_scale
