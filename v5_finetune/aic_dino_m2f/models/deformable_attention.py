from __future__ import annotations

import types

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def differentiable_ms_deform_attn_core(
    value: Tensor,
    value_spatial_shapes: Tensor,
    sampling_locations: Tensor,
    attention_weights: Tensor,
) -> Tensor:
    """Official PyTorch MSDeformAttn math without a custom backward barrier."""
    batch, _, num_heads, channels_per_head = value.shape
    _, query_length, _, num_levels, num_points, _ = sampling_locations.shape
    split_sizes = [int(height * width) for height, width in value_spatial_shapes.tolist()]
    value_list = value.split(split_sizes, dim=1)
    sampling_grids = sampling_locations.mul(2.0).sub(1.0)
    sampled_per_level = []
    for level, (height_tensor, width_tensor) in enumerate(value_spatial_shapes):
        height = int(height_tensor.item())
        width = int(width_tensor.item())
        level_value = (
            value_list[level]
            .flatten(2)
            .transpose(1, 2)
            .reshape(batch * num_heads, channels_per_head, height, width)
        )
        level_grid = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        sampled_per_level.append(
            F.grid_sample(
                level_value,
                level_grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
        )
    weights = attention_weights.transpose(1, 2).reshape(
        batch * num_heads, 1, query_length, num_levels * num_points
    )
    output = (
        torch.stack(sampled_per_level, dim=-2).flatten(-2) * weights
    ).sum(-1).view(batch, num_heads * channels_per_head, query_length)
    return output.transpose(1, 2).contiguous()


def differentiable_ms_deform_forward(
    self: nn.Module,
    query: Tensor,
    reference_points: Tensor,
    input_flatten: Tensor,
    input_spatial_shapes: Tensor,
    input_level_start_index: Tensor,
    input_padding_mask: Tensor | None = None,
) -> Tensor:
    del input_level_start_index
    batch, input_length, _ = input_flatten.shape
    if int((input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum()) != input_length:
        raise ValueError("Flattened input length does not match spatial shapes")
    value = self.value_proj(input_flatten)
    if input_padding_mask is not None:
        value = value.masked_fill(input_padding_mask[..., None], 0.0)
    value = value.view(
        batch,
        input_length,
        self.n_heads,
        int(self.ratio * self.d_model) // self.n_heads,
    )
    sampling_offsets = self.sampling_offsets(query).view(
        batch, query.shape[1], self.n_heads, self.n_levels, self.n_points, 2
    )
    attention_weights = self.attention_weights(query).view(
        batch, query.shape[1], self.n_heads, self.n_levels * self.n_points
    )
    attention_weights = attention_weights.softmax(-1).view(
        batch, query.shape[1], self.n_heads, self.n_levels, self.n_points
    )
    if reference_points.shape[-1] == 2:
        normalizer = torch.stack(
            (input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]), dim=-1
        )
        sampling_locations = (
            reference_points[:, :, None, :, None, :]
            + sampling_offsets / normalizer[None, None, None, :, None, :]
        )
    elif reference_points.shape[-1] == 4:
        sampling_locations = (
            reference_points[:, :, None, :, None, :2]
            + sampling_offsets
            / self.n_points
            * reference_points[:, :, None, :, None, 2:]
            * 0.5
        )
    else:
        raise ValueError("reference_points must end in two coordinates or four box values")
    output = differentiable_ms_deform_attn_core(
        value,
        input_spatial_shapes,
        sampling_locations,
        attention_weights,
    )
    return self.output_proj(output)


def patch_ms_deform_attn_modules(root: nn.Module) -> int:
    from dinov3.eval.segmentation.models.utils.ms_deform_attn import MSDeformAttn

    patched = 0
    for module in root.modules():
        if isinstance(module, MSDeformAttn):
            module.forward = types.MethodType(differentiable_ms_deform_forward, module)
            patched += 1
    if patched == 0:
        raise RuntimeError("No MSDeformAttn modules were found to patch")
    return patched
