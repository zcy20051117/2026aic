from __future__ import annotations

import numpy as np

from aic_dino_m2f.data.constants import (
    IGNORE_INTERNAL_ID,
    map_internal_to_official,
    map_official_to_internal,
)
from aic_dino_m2f.data.targets import semantic_to_m2f_target


def test_official_internal_mapping_round_trip_without_uint8_underflow() -> None:
    official = np.arange(9, dtype=np.uint8).reshape(3, 3)
    internal = map_official_to_internal(official)
    assert internal.dtype == np.uint8
    assert internal.tolist() == [[255, 0, 1], [2, 3, 4], [5, 6, 7]]
    assert map_internal_to_official(internal).tolist() == official.tolist()


def test_target_keeps_background_and_merges_disconnected_same_class() -> None:
    semantic = np.array(
        [[0, 255, 0], [1, 1, 255], [0, 2, 2]],
        dtype=np.uint8,
    )
    target = semantic_to_m2f_target(semantic)
    assert target["labels"].tolist() == [0, 1, 2]
    assert target["masks"].shape == (3, 3, 3)
    assert target["masks"][0].sum().item() == 3
    assert target["valid_mask"].sum().item() == 7
    assert not target["skip_supervision"]


def test_all_ignore_target_skips_all_supervision() -> None:
    semantic = np.full((4, 5), IGNORE_INTERNAL_ID, dtype=np.uint8)
    target = semantic_to_m2f_target(semantic)
    assert target["labels"].numel() == 0
    assert target["masks"].shape == (0, 4, 5)
    assert not target["valid_mask"].any()
    assert target["skip_supervision"]
