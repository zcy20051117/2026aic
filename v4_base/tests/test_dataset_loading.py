from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from aic_dino_m2f.data.dataset import AICSegmentationDataset


def test_dataset_loads_original_rgb_and_builds_internal_target(tmp_path: Path) -> None:
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    name = "sample.png"
    rgb = np.zeros((4, 5, 3), dtype=np.uint8)
    rgb[..., 0] = 255
    official = np.array(
        [[0, 1, 1, 2, 2], [0, 1, 3, 3, 2], [1, 1, 3, 2, 2], [1, 1, 1, 1, 1]],
        dtype=np.uint8,
    )
    Image.fromarray(rgb).save(images / name)
    Image.fromarray(official).save(masks / name)
    split = tmp_path / "train.txt"
    split.write_text(name + "\n", encoding="utf-8")

    dataset = AICSegmentationDataset(split, images, masks, expected_size=(4, 5))
    sample = dataset[0]

    assert sample["filename"] == name
    assert sample["image"].shape == (3, 4, 5)
    torch.testing.assert_close(
        sample["image"][:, 0, 0],
        torch.tensor([(1.0 - 0.485) / 0.229, -0.456 / 0.224, -0.406 / 0.225]),
    )
    assert sample["semantic"].shape == (4, 5)
    assert sample["semantic"][0, 0].item() == 255
    assert sample["semantic"][0, 1].item() == 0
    assert sample["target"]["labels"].tolist() == [0, 1, 2]
