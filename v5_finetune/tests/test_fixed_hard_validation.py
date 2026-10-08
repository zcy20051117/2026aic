from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
from PIL import Image

from aic_dino_m2f.data.dataset import AICSegmentationDataset
from aic_dino_m2f.data.hard_validation import degrade_fixed_v2_like


def test_fixed_degradation_is_deterministic_and_keeps_geometry() -> None:
    gradient = np.arange(64, dtype=np.uint8).reshape(8, 8)
    image = Image.fromarray(np.stack([gradient * 2, gradient, gradient], axis=-1))

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        first = degrade_fixed_v2_like(image, "scene.png")
        second = degrade_fixed_v2_like(image, "scene.png")

    assert first.size == image.size
    np.testing.assert_array_equal(np.asarray(first), np.asarray(second))
    assert not np.array_equal(np.asarray(first), np.asarray(image))


def test_hard_validation_transforms_image_but_not_mask(tmp_path: Path) -> None:
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    original = np.full((8, 8, 3), 100, dtype=np.uint8)
    original[:4] = 150
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[:4] = 5
    Image.fromarray(original).save(images / "a.png")
    Image.fromarray(mask).save(masks / "a.png")
    split = tmp_path / "val.txt"
    split.write_text("a.png\n", encoding="utf-8")

    clean = AICSegmentationDataset(split, images, masks, expected_size=(8, 8))
    degraded = AICSegmentationDataset(
        split, images, masks, expected_size=(8, 8), image_transform=degrade_fixed_v2_like,
    )

    assert not np.array_equal(clean[0]["image"].numpy(), degraded[0]["image"].numpy())
    np.testing.assert_array_equal(clean[0]["semantic"].numpy(), degraded[0]["semantic"].numpy())
