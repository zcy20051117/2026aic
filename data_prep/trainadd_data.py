"""Load original and processed training images with their shared masks."""

from torch.utils.data import ConcatDataset

from .data import AICDataset


def build_combined_train_data(original_paths, added_paths):
    original = AICDataset(*original_paths)
    added = AICDataset(*added_paths)
    unexpected = set(added.names) - set(original.names)
    if unexpected:
        raise ValueError(f"trainadd contains images outside train.txt: {sorted(unexpected)[0]}")
    if original.mask_dir.resolve() != added.mask_dir.resolve():
        raise ValueError("trainadd must use the original training mask directory")
    return ConcatDataset((original, added))
