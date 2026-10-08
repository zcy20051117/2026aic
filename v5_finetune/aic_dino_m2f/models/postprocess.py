from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from aic_dino_m2f.data.constants import NUM_CLASSES


def semantic_scores(
    pred_logits: Tensor,
    pred_masks: Tensor,
    *,
    output_size: tuple[int, int],
    query_chunk: int | None = None,
) -> Tensor:
    if pred_logits.ndim != 3 or pred_logits.shape[-1] != NUM_CLASSES + 1:
        raise ValueError(f"Expected pred_logits [B,Q,9], got {tuple(pred_logits.shape)}")
    if pred_masks.ndim != 4 or pred_masks.shape[:2] != pred_logits.shape[:2]:
        raise ValueError(
            f"Expected pred_masks [B,Q,H,W] aligned with logits, got {tuple(pred_masks.shape)}"
        )
    query_count = pred_logits.shape[1]
    chunk_size = query_count if query_chunk is None else int(query_chunk)
    if chunk_size <= 0:
        raise ValueError("query_chunk must be positive")

    class_prob = pred_logits.float().softmax(dim=-1)[..., :NUM_CLASSES]
    scores = torch.zeros(
        (pred_logits.shape[0], NUM_CLASSES, *output_size),
        dtype=torch.float32,
        device=pred_masks.device,
    )
    for start in range(0, query_count, chunk_size):
        stop = min(start + chunk_size, query_count)
        resized_logits = F.interpolate(
            pred_masks[:, start:stop].float(),
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
        mask_prob = resized_logits.sigmoid()
        scores += torch.einsum("bqc,bqhw->bchw", class_prob[:, start:stop], mask_prob)
    return scores
