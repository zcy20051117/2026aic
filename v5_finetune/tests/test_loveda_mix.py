from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from aic_dino_m2f.data.loveda_mix import (
    MultiSourceDataset,
    discover_loveda_pairs,
    select_vehicle_top_names,
)
from aic_dino_m2f.data.dataset import AICSegmentationDataset


def _mask(path: Path, vehicle_pixels: int, barren_pixels: int = 0) -> None:
    values = np.ones((4, 5), dtype=np.uint8)
    values.flat[:vehicle_pixels] = 8
    values.flat[vehicle_pixels:vehicle_pixels + barren_pixels] = 5
    Image.fromarray(values).save(path)


def test_vehicle_ranking_uses_pixel_share_and_excludes_validation(tmp_path: Path) -> None:
    masks = tmp_path / "masks"
    masks.mkdir()
    for name, pixels in (("low.png", 1), ("high.png", 8), ("mid.png", 4)):
        _mask(masks / name, pixels)
    train = tmp_path / "train.txt"
    val = tmp_path / "val.txt"
    train.write_text("low.png\nhigh.png\nmid.png\n", encoding="utf-8")
    val.write_text("heldout.png\n", encoding="utf-8")

    names, stats = select_vehicle_top_names(train, val, masks, limit=2, seed=42, expected_size=(4, 5), show_progress=False)

    assert names == ["high.png", "mid.png"]
    assert stats["vehicle_positive_total"] == 3
    assert stats["vehicle_positive_selected"] == 2


def test_vehicle_ranking_rejects_train_validation_overlap(tmp_path: Path) -> None:
    masks = tmp_path / "masks"
    masks.mkdir()
    _mask(masks / "same.png", 1)
    train = tmp_path / "train.txt"
    val = tmp_path / "val.txt"
    train.write_text("same.png\n", encoding="utf-8")
    val.write_text("same.png\n", encoding="utf-8")

    with pytest.raises(ValueError, match="overlap"):
        select_vehicle_top_names(train, val, masks, limit=1, seed=42, expected_size=(4, 5), show_progress=False)


def test_loveda_pair_discovery_requires_exact_image_mask_match(tmp_path: Path) -> None:
    images = tmp_path / "images_png"
    masks = tmp_path / "masks_png"
    images.mkdir()
    masks.mkdir()
    Image.new("RGB", (2, 2)).save(images / "0.png")
    Image.new("L", (2, 2)).save(masks / "0.png")
    assert discover_loveda_pairs(images, masks) == ["0.png"]
    Image.new("RGB", (2, 2)).save(images / "1.png")
    with pytest.raises(ValueError, match="pair"):
        discover_loveda_pairs(images, masks)


def test_multi_source_dataset_preserves_loveda_label_mapping(tmp_path: Path) -> None:
    items = []
    for source, label in (("aic", 8), ("loveda_rural", 6)):
        folder = tmp_path / source
        images = folder / "images"
        masks = folder / "masks"
        images.mkdir(parents=True)
        masks.mkdir()
        Image.new("RGB", (4, 5)).save(images / "same.png")
        mask = np.full((5, 4), label, dtype=np.uint8)
        mask[0, 0] = 0
        Image.fromarray(mask).save(masks / "same.png")
        split = folder / "split.txt"
        split.write_text("same.png\n", encoding="utf-8")
        items.append(AICSegmentationDataset(split, images, masks, expected_size=(5, 4), source=source))
    mixed = MultiSourceDataset(items)

    assert len(mixed) == 2
    assert mixed[0]["source"] == "aic"
    assert mixed[0]["semantic"][1, 1].item() == 7
    assert mixed[1]["source"] == "loveda_rural"
    assert mixed[1]["semantic"][1, 1].item() == 5
    assert mixed[1]["semantic"][0, 0].item() == 255
    assert int(mixed.official_mask(1)[1, 1]) == 6
