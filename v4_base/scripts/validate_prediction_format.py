from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
from tqdm.auto import tqdm


def validate_prediction_directory(
    prediction_dir: Path,
    expected_names: Iterable[str],
    *,
    expected_size: tuple[int, int] = (1024, 1024),
) -> dict[str, int]:
    root = Path(prediction_dir).resolve(strict=True)
    expected = list(expected_names)
    actual = sorted(path.name for path in root.iterdir() if path.is_file())
    if actual != sorted(expected):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise ValueError(f"Prediction filenames mismatch; missing={missing[:5]}, extra={extra[:5]}")
    for name in tqdm(expected, desc="validate predictions", dynamic_ncols=True):
        with Image.open(root / name) as image:
            if image.mode != "L":
                raise ValueError(f"Prediction {name} must use Pillow mode L, got {image.mode}")
            array = np.asarray(image)
        if array.shape != expected_size:
            raise ValueError(f"Prediction {name} has size {array.shape}, expected {expected_size}")
        unique = np.unique(array)
        if np.any((unique < 1) | (unique > 8)):
            raise ValueError(f"Prediction {name} contains invalid official IDs: {unique.tolist()}")
    return {"files": len(expected), "width": expected_size[1], "height": expected_size[0]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prediction_dir", type=Path)
    parser.add_argument("expected_list", type=Path)
    args = parser.parse_args()
    names = [line.strip() for line in args.expected_list.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    print(json.dumps(validate_prediction_directory(args.prediction_dir, names), indent=2), flush=True)


if __name__ == "__main__":
    main()
