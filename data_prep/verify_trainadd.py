"""Verify the uploaded trainadd split, images, masks, and data loader."""

import json

import torch

from .train import ROOT, dataset_paths
from .trainadd_data import build_combined_train_data
from .weighted_train import make_loaders


def main():
    original_paths, val_paths = dataset_paths()
    original_names = original_paths[2].read_text(encoding="utf-8-sig").splitlines()
    val_names = val_paths[2].read_text(encoding="utf-8-sig").splitlines()
    add_split = ROOT / "splits/v1/trainadd.txt"
    added_names = add_split.read_text(encoding="utf-8-sig").splitlines()
    image_names = {p.name for p in (ROOT / "trainadd").glob("*.png")}
    if (len(original_names), len(added_names), len(val_names)) != (5597, 1027, 1399):
        raise ValueError("unexpected train/trainadd/val split counts")
    if len(set(added_names)) != 1027 or set(added_names) != image_names:
        raise ValueError("trainadd split and uploaded image names differ")
    if not set(added_names) <= set(original_names) or set(added_names) & set(val_names):
        raise ValueError("trainadd images must be original training images only")
    train = build_combined_train_data(
        original_paths, (ROOT / "trainadd", ROOT / "train/masks", add_split)
    )
    image, target, sample_name = train[5597]
    original_index = original_names.index(sample_name)
    original_image, original_target, _ = train[original_index]
    if torch.equal(image, original_image) or not torch.equal(target, original_target):
        raise ValueError("processed image or shared original mask does not match expectation")
    train_loader, val_loader, _ = make_loaders(2, 0, False, True)
    if len(train) != 6624 or len(train_loader.dataset) != 6624 or len(val_loader.dataset) != 1399:
        raise ValueError("combined data loader counts differ")
    print(json.dumps({"status": "TRAINADD_OK", "original": 5597, "added": 1027,
                      "train_total": len(train), "val": len(val_loader.dataset),
                      "sample_name": sample_name, "image_shape": list(image.shape),
                      "mask_shape": list(target.shape)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
