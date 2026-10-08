from __future__ import annotations

from typing import Mapping

from torch import Tensor, nn

from dinov3.eval.segmentation.models.heads.mask2former_transformer_decoder import (
    MultiScaleMaskedTransformerDecoder,
)

from .pixel_decoder import CorrectedMSDeformAttnPixelDecoder


class TrainableMask2FormerHead(nn.Module):
    def __init__(
        self,
        *,
        input_channels: int,
        num_classes: int,
        hidden_dim: int,
        mask_dim: int,
        conv_dim: int,
        pixel_decoder_encoder_layers: int,
        pixel_decoder_attention_heads: int,
        pixel_decoder_ffn_dim: int,
        transformer_decoder_layers: int,
        transformer_decoder_attention_heads: int,
        transformer_decoder_ffn_dim: int,
        num_queries: int,
        dropout: float,
        common_stride: int,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.pixel_decoder = CorrectedMSDeformAttnPixelDecoder(
            input_channels=input_channels,
            conv_dim=conv_dim,
            mask_dim=mask_dim,
            encoder_layers=pixel_decoder_encoder_layers,
            attention_heads=pixel_decoder_attention_heads,
            ffn_dim=pixel_decoder_ffn_dim,
            dropout=dropout,
            common_stride=common_stride,
        )
        self.predictor = MultiScaleMaskedTransformerDecoder(
            in_channels=conv_dim,
            mask_classification=True,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            nheads=transformer_decoder_attention_heads,
            dim_feedforward=transformer_decoder_ffn_dim,
            dec_layers=transformer_decoder_layers,
            pre_norm=False,
            mask_dim=mask_dim,
            enforce_input_project=conv_dim != hidden_dim,
        )

    def forward(self, features: Mapping[str, Tensor]) -> dict[str, object]:
        mask_features, _, multi_scale = self.pixel_decoder.forward_features(features)
        return self.predictor(multi_scale, mask_features, None)
