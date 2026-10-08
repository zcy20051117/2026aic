from __future__ import annotations

import torch

from aic_dino_m2f.models.mask2former import TrainableMask2FormerHead


def test_true_256d_mask2former_shapes_layers_strides_and_aux_outputs() -> None:
    head = TrainableMask2FormerHead(
        input_channels=64,
        num_classes=8,
        hidden_dim=256,
        mask_dim=256,
        conv_dim=256,
        pixel_decoder_encoder_layers=6,
        pixel_decoder_attention_heads=8,
        pixel_decoder_ffn_dim=1024,
        transformer_decoder_layers=9,
        transformer_decoder_attention_heads=8,
        transformer_decoder_ffn_dim=2048,
        num_queries=100,
        dropout=0.0,
        common_stride=4,
    )
    features = {
        "1": torch.randn((1, 64, 16, 16)),
        "2": torch.randn((1, 64, 8, 8)),
        "3": torch.randn((1, 64, 4, 4)),
        "4": torch.randn((1, 64, 2, 2)),
    }
    output = head(features)

    assert output["pred_logits"].shape == (1, 100, 9)
    assert output["pred_masks"].shape == (1, 100, 16, 16)
    assert len(output["aux_outputs"]) == 9
    assert head.pixel_decoder.feature_strides == [4, 8, 16, 32]
    assert head.pixel_decoder.transformer_feature_strides == [8, 16, 32]
    assert head.pixel_decoder.transformer_num_feature_levels == 3
    assert head.predictor.num_layers == 9
    assert head.predictor.num_queries == 100
    assert head.hidden_dim == 256

    loss = output["pred_logits"].float().mean() + output["pred_masks"].float().mean()
    loss.backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in head.parameters()
        if parameter.requires_grad
    )
