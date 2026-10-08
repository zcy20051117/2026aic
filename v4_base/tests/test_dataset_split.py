from __future__ import annotations

from pathlib import Path

import pytest

from aic_dino_m2f.data.dataset import validate_split_lists


def test_split_validation_requires_disjoint_complete_image_mask_pairs(tmp_path: Path) -> None:
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    for name in ("a.png", "b.png", "c.png"):
        (images / name).write_bytes(b"image")
        (masks / name).write_bytes(b"mask")
    train = tmp_path / "train.txt"
    val = tmp_path / "val.txt"
    train.write_text("a.png\nb.png\n", encoding="utf-8")
    val.write_text("c.png\n", encoding="utf-8")

    summary = validate_split_lists(train, val, images, masks, expected_total=3)
    assert summary == {"train": 2, "val": 1, "total": 3}

    val.write_text("b.png\nc.png\n", encoding="utf-8")
    with pytest.raises(ValueError, match="overlap"):
        validate_split_lists(train, val, images, masks, expected_total=3)
