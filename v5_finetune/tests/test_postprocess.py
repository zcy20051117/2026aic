from __future__ import annotations

import torch

from aic_dino_m2f.models.postprocess import semantic_scores


def test_semantic_composition_keeps_background_and_drops_no_object_after_softmax() -> None:
    logits = torch.full((1, 2, 9), -20.0)
    logits[0, 0, 0] = 4.0
    logits[0, 0, 8] = 3.0
    logits[0, 1, 1] = 4.0
    masks = torch.tensor([[[[2.0]], [[-2.0]]]])
    scores = semantic_scores(logits, masks, output_size=(2, 2), query_chunk=None)
    probs = logits.softmax(-1)[..., :8]
    expected = torch.einsum("bqc,bqhw->bchw", probs, masks.sigmoid().expand(-1, -1, 2, 2))
    torch.testing.assert_close(scores, expected.float())
    assert scores.shape == (1, 8, 2, 2)
    assert scores[:, 0].sum() > scores[:, 1].sum()


def test_query_chunk_matches_non_chunked_fp32_accumulation() -> None:
    generator = torch.Generator().manual_seed(42)
    logits = torch.randn((2, 7, 9), generator=generator)
    masks = torch.randn((2, 7, 3, 5), generator=generator)
    direct = semantic_scores(logits, masks, output_size=(6, 10), query_chunk=None)
    chunked = semantic_scores(logits, masks, output_size=(6, 10), query_chunk=3)
    torch.testing.assert_close(chunked, direct, rtol=1e-5, atol=1e-6)
