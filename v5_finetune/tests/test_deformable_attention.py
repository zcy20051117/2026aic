from __future__ import annotations

import torch

from aic_dino_m2f.models.deformable_attention import differentiable_ms_deform_attn_core
from dinov3.eval.segmentation.models.utils.ms_deform_attn import ms_deform_attn_core_pytorch


def test_project_deformable_core_matches_official_forward_and_has_gradients() -> None:
    generator = torch.Generator().manual_seed(42)
    spatial_shapes = torch.tensor([[2, 2], [1, 2]], dtype=torch.long)
    value = torch.randn((1, 6, 2, 4), generator=generator, requires_grad=True)
    locations = torch.rand((1, 3, 2, 2, 2, 2), generator=generator, requires_grad=True)
    raw_weights = torch.randn((1, 3, 2, 2, 2), generator=generator, requires_grad=True)
    weights = raw_weights.softmax(dim=-1)

    expected = ms_deform_attn_core_pytorch(value, spatial_shapes, locations, weights)
    actual = differentiable_ms_deform_attn_core(value, spatial_shapes, locations, weights)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

    actual.square().mean().backward()
    assert value.grad is not None and torch.isfinite(value.grad).all()
    assert locations.grad is not None and torch.isfinite(locations.grad).all()
    assert raw_weights.grad is not None and torch.isfinite(raw_weights.grad).all()
