from __future__ import annotations

import torch

from aic_dino_m2f.losses.point_sampling import sample_valid_point_coords


def test_sampled_points_never_select_ignore_pixels() -> None:
    valid = torch.tensor(
        [[True, False, True], [False, True, True], [True, True, False]],
        dtype=torch.bool,
    )
    logits = torch.randn((5, 3, 3), generator=torch.Generator().manual_seed(7))
    sample = sample_valid_point_coords(
        logits,
        valid,
        num_points=100,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
        generator=torch.Generator().manual_seed(11),
    )
    assert sample.coords.shape == (100, 2)
    assert sample.flat_indices.shape == (100,)
    assert valid.flatten()[sample.flat_indices].all()


def test_all_ignore_returns_no_points_without_nan() -> None:
    sample = sample_valid_point_coords(
        torch.zeros((2, 4, 5)),
        torch.zeros((4, 5), dtype=torch.bool),
        num_points=32,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
        generator=torch.Generator().manual_seed(1),
    )
    assert sample.coords.shape == (0, 2)
    assert sample.flat_indices.numel() == 0
