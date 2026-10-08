from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Mapping

import yaml


_LOCKED = {
    "schema_version": 1,
    "data.image_size": 1024,
    "data.original_count": 5464,
    "data.aic_selected_count": 2500,
    "data.loveda_count": 2522,
    "data.train_count": 5022,
    "data.val_count": 1368,
    "model.backbone": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "model.interaction_indexes": [4, 11, 17, 23],
    "model.hidden_dim": 256,
    "model.mask_dim": 256,
    "model.conv_dim": 256,
    "model.pixel_decoder_encoder_layers": 6,
    "model.transformer_decoder_layers": 9,
    "model.num_queries": 100,
    "model.num_classes": 8,
    "model.common_stride": 4,
    "loss.num_points": 12544,
    "loss.semantic_class_weights": [1.0, 1.0, 1.0, 1.0, 1.5, 1.0, 1.0, 1.0],
    "augmentation.enabled": True,
    "augmentation.small_barren_ratio_max": 0.01479625702,
    "augmentation.medium_barren_ratio_max": 0.04709005356,
    "augmentation.crop_probability": 0.70,
    "augmentation.small_crop_size": 512,
    "augmentation.medium_crop_size": 768,
    "sampling.barren_weighted": True,
    "selection.barren_min_miou": 0.792,
    "training.max_epochs": 50,
    "training.optimizer": "AdamW",
    "training.learning_rate": 0.0001,
    "training.weight_decay": 0.05,
    "training.drop_last": False,
}


def _nested(config: Mapping[str, object], dotted: str) -> object:
    value: object = config
    for part in dotted.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise ValueError(f"Missing required config key: {dotted}")
        value = value[part]
    return value


def validate_config(config: Mapping[str, object]) -> dict[str, object]:
    materialized = dict(config)
    for dotted, expected in _LOCKED.items():
        actual = _nested(materialized, dotted)
        if actual != expected:
            raise ValueError(f"Structural setting is locked: {dotted}={expected!r}, got {actual!r}")
    serialized = json.dumps(materialized, sort_keys=True, default=str).lower()
    if "test_1" in serialized or "test1" in serialized:
        raise ValueError("Test1 paths are prohibited for baseline v1")
    physical = int(_nested(materialized, "training.physical_batch_size"))
    effective = int(_nested(materialized, "training.target_effective_batch"))
    if physical <= 0 or effective <= 0:
        raise ValueError("Batch sizes must be positive")
    if effective != physical:
        raise ValueError("Without gradient accumulation, physical and effective batch must be the same")
    if int(_nested(materialized, "training.patience")) <= 0:
        raise ValueError("Early-stopping patience must be positive")
    return materialized


def load_and_validate_config(path: Path) -> dict[str, object]:
    resolved = Path(path).resolve(strict=True)
    loaded = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("Configuration root must be a mapping")
    return validate_config(_expand_environment(loaded))


def _expand_environment(value: object) -> object:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    return value


def config_sha256(config: Mapping[str, object]) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
