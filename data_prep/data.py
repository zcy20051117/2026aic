"""Fixed AIC split loading with no training augmentation."""

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
IGNORE_INDEX = 255
CLASS_NAMES = (
    "background", "building", "road", "water", "barren land",
    "vegetation", "agricultural land", "vehicle",
)


def raw_mask_to_target(raw: np.ndarray) -> np.ndarray:
    if raw.ndim != 2:
        raise ValueError(f"mask must be single-channel, got {raw.shape}")
    values = set(np.unique(raw).tolist())
    if values - set(range(9)):
        raise ValueError(f"unexpected mask values: {sorted(values - set(range(9)))}")
    target = raw.astype(np.uint8, copy=True)
    ignore = target == 0
    target[~ignore] -= 1
    target[ignore] = IGNORE_INDEX
    return target


class AICDataset(Dataset):
    def __init__(self, image_dir: Path, mask_dir: Path, split_file: Path):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.names = [
            line.strip()
            for line in Path(split_file).read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
        if not self.names or len(set(self.names)) != len(self.names):
            raise ValueError(f"empty or duplicate split entries: {split_file}")
        missing = [
            name for name in self.names
            if not (self.image_dir / name).is_file() or not (self.mask_dir / name).is_file()
        ]
        if missing:
            raise FileNotFoundError(f"{len(missing)} missing image/mask pairs; first: {missing[0]}")

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        image = cv2.imread(str(self.image_dir / name), cv2.IMREAD_COLOR)
        raw_mask = cv2.imread(str(self.mask_dir / name), cv2.IMREAD_UNCHANGED)
        if image is None or raw_mask is None:
            raise RuntimeError(f"failed to read image or mask: {name}")
        if image.shape != (1024, 1024, 3) or raw_mask.shape != (1024, 1024):
            raise ValueError(f"unexpected size for {name}: {image.shape}, {raw_mask.shape}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = (image - MEAN) / STD
        target = raw_mask_to_target(raw_mask)
        return (
            torch.from_numpy(image).permute(2, 0, 1),
            torch.from_numpy(target.astype(np.int64)).unsqueeze(0),
            name,
        )
