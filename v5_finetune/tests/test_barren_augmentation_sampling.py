from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from aic_dino_m2f.data.augmentation import (
    apply_d4,
    build_train_barren_augmenter,
    compute_guided_crop_bounds,
    guided_crop_size_for_barren_ratio,
    guided_crop_and_resize,
)
from aic_dino_m2f.data.dataset import AICSegmentationDataset
from aic_dino_m2f.data.sampling import (
    barren_sampling_weights,
    build_barren_weighted_sampler,
    compute_barren_ratios,
)


def test_d4_keeps_rgb_mask_pairs_aligned_and_preserves_labels() -> None:
    official = np.array(
        [[0, 1, 2, 3], [4, 5, 6, 7], [8, 1, 2, 3]], dtype=np.uint8
    )
    rgb = np.repeat((official * 20)[..., None], 3, axis=2)

    image, transformed = apply_d4(Image.fromarray(rgb), official, transform_index=5)

    transformed_rgb = np.asarray(image)
    np.testing.assert_array_equal(transformed_rgb[..., 0] // 20, transformed)
    assert set(np.unique(transformed)) == set(np.unique(official))


def test_guided_crop_bounds_include_chosen_barren_pixel_at_edges() -> None:
    official = np.zeros((12, 12), dtype=np.uint8)
    official[11, 0] = 5

    top, left, selected_y, selected_x = compute_guided_crop_bounds(
        official, crop_size=8, rng=random.Random(7)
    )

    assert top <= selected_y < top + 8
    assert left <= selected_x < left + 8
    assert (selected_y, selected_x) == (11, 0)
    assert (top, left) == (4, 0)


def test_guided_crop_resizes_mask_with_nearest_and_keeps_barren() -> None:
    official = np.zeros((12, 12), dtype=np.uint8)
    official[9:12, 0:3] = 5
    rgb = np.repeat((official * 20)[..., None], 3, axis=2)

    image, transformed = guided_crop_and_resize(
        Image.fromarray(rgb), official, crop_size=8, output_size=(12, 12), rng=random.Random(3)
    )

    assert image.size == (12, 12)
    assert transformed.shape == (12, 12)
    assert 5 in np.unique(transformed)
    assert set(np.unique(transformed)).issubset(set(np.unique(official)))


def test_guided_crop_size_follows_small_and_medium_barren_ratio_rules() -> None:
    assert guided_crop_size_for_barren_ratio(0.0) is None
    assert guided_crop_size_for_barren_ratio(0.01479625702) == 512
    assert guided_crop_size_for_barren_ratio(0.01479625703) == 768
    assert guided_crop_size_for_barren_ratio(0.04709005356) == 768
    assert guided_crop_size_for_barren_ratio(0.04709005357) is None


def test_train_augmenter_reads_only_v3_augmentation_parameters() -> None:
    augmenter = build_train_barren_augmenter(
        {
            "small_barren_ratio_max": 0.01,
            "medium_barren_ratio_max": 0.02,
            "crop_probability": 0.5,
            "small_crop_size": 256,
            "medium_crop_size": 512,
        }
    )

    assert augmenter.small_ratio_max == 0.01
    assert augmenter.medium_ratio_max == 0.02
    assert augmenter.crop_probability == 0.5
    assert augmenter.small_crop_size == 256
    assert augmenter.medium_crop_size == 512


def test_barren_sampling_weights_obey_all_ratio_boundaries() -> None:
    ratios = [0.0, 0.01479625702, 0.01479625703, 0.04709005356, 0.04709005357, 0.1418712139, 0.141871214]

    weights = barren_sampling_weights(ratios)

    torch.testing.assert_close(
        weights,
        torch.tensor([1.0, 2.0, 1.75, 1.75, 1.25, 1.25, 1.0], dtype=torch.double),
    )


def test_weighted_sampler_is_seeded_replacement_and_keeps_epoch_length() -> None:
    sampler = build_barren_weighted_sampler([0.0, 0.01, 0.03], seed=42)

    assert sampler.replacement is True
    assert sampler.num_samples == 3
    assert sampler.generator.initial_seed() == 42
    assert len(sampler) == 3


def test_precomputed_barren_ratios_and_validation_dataset_without_transform(tmp_path: Path) -> None:
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    names = ["empty.png", "barren.png"]
    for name in names:
        Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(images / name)
    Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(masks / names[0])
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[:2, :2] = 5
    Image.fromarray(mask).save(masks / names[1])
    split = tmp_path / "validation.txt"
    split.write_text("\n".join(names) + "\n", encoding="utf-8")

    dataset = AICSegmentationDataset(split, images, masks, expected_size=(4, 4))
    ratios = compute_barren_ratios(dataset, show_progress=False)
    sample = dataset[1]

    assert ratios == [0.0, 0.25]
    assert sample["semantic"][0, 0].item() == 4
    assert sample["semantic"][3, 3].item() == 255


def test_precomputed_barren_ratios_can_limit_mask_reads_for_smoke() -> None:
    class _Masks:
        def __init__(self) -> None:
            self.read_indices: list[int] = []

        def __len__(self) -> int:
            return 4

        def official_mask(self, index: int) -> np.ndarray:
            self.read_indices.append(index)
            mask = np.zeros((2, 2), dtype=np.uint8)
            mask.flat[:index] = 5
            return mask

    dataset = _Masks()
    ratios = compute_barren_ratios(dataset, indices=[1, 3], show_progress=False)

    assert dataset.read_indices == [1, 3]
    assert ratios == [0.25, 0.75]
