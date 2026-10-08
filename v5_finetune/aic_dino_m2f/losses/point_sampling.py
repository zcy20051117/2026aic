from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class ValidPointSample:
    coords: Tensor
    flat_indices: Tensor


def _indices_to_coords(indices: Tensor, height: int, width: int) -> Tensor:
    y = torch.div(indices, width, rounding_mode="floor")
    x = indices.remainder(width)
    return torch.stack(((x.float() + 0.5) / width, (y.float() + 0.5) / height), dim=-1)


def point_sample(mask_logits: Tensor, coords: Tensor) -> Tensor:
    """Bilinearly sample `[N,H,W]` logits at shared normalized `[P,2]` coordinates."""
    if mask_logits.ndim != 3:
        raise ValueError(f"Expected [N,H,W] logits, got {tuple(mask_logits.shape)}")
    if coords.ndim != 2 or coords.shape[-1] != 2:
        raise ValueError(f"Expected [P,2] coordinates, got {tuple(coords.shape)}")
    if coords.shape[0] == 0:
        return mask_logits.new_empty((mask_logits.shape[0], 0))
    grid = coords.to(device=mask_logits.device, dtype=mask_logits.dtype)
    grid = grid.mul(2.0).sub(1.0).view(1, 1, -1, 2).expand(mask_logits.shape[0], -1, -1, -1)
    sampled = F.grid_sample(
        mask_logits.unsqueeze(1),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    return sampled[:, 0, 0, :]


def sample_valid_point_coords(
    prediction_logits: Tensor,
    valid_mask: Tensor,
    *,
    num_points: int,
    oversample_ratio: float,
    importance_sample_ratio: float,
    generator: torch.Generator | None = None,
) -> ValidPointSample:
    """Choose shared uncertainty-biased points strictly from valid original pixels."""
    if prediction_logits.ndim != 3:
        raise ValueError(f"Expected prediction logits [Q,H,W], got {tuple(prediction_logits.shape)}")
    if valid_mask.ndim != 2:
        raise ValueError(f"Expected valid mask [H,W], got {tuple(valid_mask.shape)}")
    if num_points <= 0 or oversample_ratio < 1.0:
        raise ValueError("num_points must be positive and oversample_ratio must be at least one")
    if not 0.0 <= importance_sample_ratio <= 1.0:
        raise ValueError("importance_sample_ratio must be in [0,1]")

    valid = valid_mask.to(device=prediction_logits.device, dtype=torch.bool)
    valid_indices = valid.flatten().nonzero(as_tuple=False).flatten()
    if valid_indices.numel() == 0:
        return ValidPointSample(
            coords=prediction_logits.new_empty((0, 2), dtype=torch.float32),
            flat_indices=valid_indices,
        )

    candidate_count = max(num_points, int(math.ceil(num_points * oversample_ratio)))
    candidate_choices = torch.randint(
        valid_indices.numel(),
        (candidate_count,),
        device=valid_indices.device,
        generator=generator,
    )
    candidate_indices = valid_indices[candidate_choices]
    height, width = valid.shape
    candidate_coords = _indices_to_coords(candidate_indices, height, width)
    with torch.no_grad():
        candidate_logits = point_sample(prediction_logits.detach().float(), candidate_coords)
        uncertainty = -candidate_logits.abs().mean(dim=0)

    importance_count = min(num_points, int(num_points * importance_sample_ratio))
    if importance_count:
        important_positions = uncertainty.topk(importance_count, sorted=False).indices
        selected_indices = candidate_indices[important_positions]
    else:
        selected_indices = valid_indices.new_empty((0,))
    random_count = num_points - importance_count
    if random_count:
        random_choices = torch.randint(
            valid_indices.numel(),
            (random_count,),
            device=valid_indices.device,
            generator=generator,
        )
        selected_indices = torch.cat((selected_indices, valid_indices[random_choices]))
    return ValidPointSample(
        coords=_indices_to_coords(selected_indices, height, width),
        flat_indices=selected_indices,
    )
