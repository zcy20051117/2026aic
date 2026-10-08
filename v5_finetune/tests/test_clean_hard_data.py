from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from aic_dino_m2f.data.dataset import (
    AICSegmentationDataset,
    MixedTrainingDataset,
    build_clean_hard_training_dataset,
    select_clean_hard_names,
)
from aic_dino_m2f.data.sampling import compute_barren_ratios


def _list(path: Path, *names: str) -> Path:
    path.write_text("\n".join(names) + "\n", encoding="utf-8")
    return path


def test_select_clean_hard_names_excludes_removed_originals(tmp_path: Path) -> None:
    train = _list(tmp_path / "train.txt", "a.png", "b.png")
    val = _list(tmp_path / "val.txt", "c.png")
    hard = _list(tmp_path / "hard.txt", "b.png", "removed.png")

    assert select_clean_hard_names(train, val, hard) == ["b.png"]


def test_select_clean_hard_names_rejects_validation_leakage(tmp_path: Path) -> None:
    train = _list(tmp_path / "train.txt", "a.png")
    val = _list(tmp_path / "val.txt", "c.png")
    hard = _list(tmp_path / "hard.txt", "a.png", "c.png")

    with pytest.raises(ValueError, match="validation"):
        select_clean_hard_names(train, val, hard)


def test_mixed_dataset_uses_hard_image_and_shared_original_mask(tmp_path: Path) -> None:
    original = tmp_path / "original"
    hard = tmp_path / "hard"
    masks = tmp_path / "masks"
    for directory in (original, hard, masks):
        directory.mkdir()
    Image.fromarray(np.full((4, 4, 3), 10, dtype=np.uint8)).save(original / "a.png")
    Image.fromarray(np.full((4, 4, 3), 20, dtype=np.uint8)).save(hard / "a.png")
    mask = np.full((4, 4), 5, dtype=np.uint8)
    Image.fromarray(mask).save(masks / "a.png")
    split = _list(tmp_path / "train.txt", "a.png")
    base = AICSegmentationDataset(split, original, masks, expected_size=(4, 4))
    difficult = AICSegmentationDataset(
        split, hard, masks, expected_size=(4, 4), source="hard"
    )

    mixed = MixedTrainingDataset(base, difficult)

    assert len(mixed) == 2
    assert mixed[0]["source"] == "original"
    assert mixed[1]["source"] == "hard"
    assert float(mixed[1]["image"].mean()) > float(mixed[0]["image"].mean())
    assert compute_barren_ratios(mixed, show_progress=False) == [1.0, 1.0]


def test_builder_rejects_missing_hard_image_before_training(tmp_path: Path) -> None:
    original = tmp_path / "original"
    hard = tmp_path / "hard"
    masks = tmp_path / "masks"
    for directory in (original, hard, masks):
        directory.mkdir()
    Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(original / "a.png")
    Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(masks / "a.png")
    train = _list(tmp_path / "train.txt", "a.png")
    val = _list(tmp_path / "val.txt", "b.png")
    hard_list = _list(tmp_path / "hard.txt", "a.png")

    with pytest.raises(FileNotFoundError, match="hard image"):
        build_clean_hard_training_dataset(
            train, val, hard_list, original, hard, masks,
            expected_size=(4, 4), expected_original=1, expected_hard=1,
        )
