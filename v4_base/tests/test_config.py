from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from aic_dino_m2f.config import load_and_validate_config, validate_config


def test_baseline_config_locks_architecture_data_and_training_protocol() -> None:
    path = Path(__file__).resolve().parents[1] / "configs" / "baseline_v1.yaml"
    config = load_and_validate_config(path)
    assert config["data"]["image_size"] == 1024
    assert config["data"]["train_count"] == 6488
    assert config["data"]["original_count"] == 5464
    assert config["data"]["hard_count"] == 1024
    assert config["data"]["val_count"] == 1368
    assert config["model"]["backbone"] == "facebook/dinov3-vitl16-pretrain-lvd1689m"
    assert config["model"]["interaction_indexes"] == [4, 11, 17, 23]
    assert config["model"]["hidden_dim"] == 256
    assert config["model"]["num_queries"] == 100
    assert config["model"]["transformer_decoder_layers"] == 9
    assert config["training"]["max_epochs"] == 50
    assert config["training"]["patience"] == 8
    assert config["training"]["target_effective_batch"] == 8
    assert config["training"]["physical_batch_size"] == 4
    assert config["loss"]["semantic_class_weights"] == [
        1.0,
        1.0,
        1.0,
        1.0,
        1.5,
        1.0,
        1.0,
        1.0,
    ]
    assert config["validation"]["loss_seed"] == 20260918
    assert config["augmentation"] == {
        "enabled": True,
        "small_barren_ratio_max": 0.01479625702,
        "medium_barren_ratio_max": 0.04709005356,
        "crop_probability": 0.70,
        "small_crop_size": 512,
        "medium_crop_size": 768,
    }
    assert config["sampling"] == {"barren_weighted": True}
    assert config["selection"] == {"barren_min_miou": 0.792}
    assert "test_1" not in str(config).lower()

    changed = deepcopy(config)
    changed["data"]["image_size"] = 512
    with pytest.raises(ValueError, match="locked"):
        validate_config(changed)

    changed = deepcopy(config)
    changed["augmentation"]["crop_probability"] = 0.5
    with pytest.raises(ValueError, match="locked"):
        validate_config(changed)
