from __future__ import annotations

import os
from pathlib import Path
import random
import tempfile
from typing import Mapping

import numpy as np
import torch
from torch import nn

from .state import EarlyStoppingState


def capture_rng_state() -> dict[str, object]:
    state: dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, object]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state:
        if not torch.cuda.is_available():
            raise RuntimeError("Checkpoint contains CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _is_backbone_key(key: str) -> bool:
    return key.startswith("backbone.") or ".backbone." in key


def downstream_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value for key, value in model.state_dict().items() if not _is_backbone_key(key)}


def _load_downstream_state(model: nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    expected = {key for key in model.state_dict() if not _is_backbone_key(key)}
    actual = set(state)
    if expected != actual:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(f"Downstream state mismatch; missing={missing}, unexpected={unexpected}")
    result = model.load_state_dict(state, strict=False)
    unexpected = [key for key in result.unexpected_keys if not _is_backbone_key(key)]
    missing = [key for key in result.missing_keys if not _is_backbone_key(key)]
    if missing or unexpected:
        raise ValueError(f"Failed strict downstream load; missing={missing}, unexpected={unexpected}")


def _atomic_torch_save(payload: dict[str, object], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary_name = handle.name
        torch.save(payload, temporary_name)
        with open(temporary_name, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def save_training_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: object | None,
    scaler: object | None,
    epoch: int,
    global_step: int,
    early_stopping: EarlyStoppingState,
    metadata: Mapping[str, object],
    include_optimizer: bool,
) -> None:
    payload: dict[str, object] = {
        "schema_version": 1,
        "downstream_state_dict": downstream_state_dict(model),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "early_stopping": early_stopping.state_dict(),
        "metadata": dict(metadata),
    }
    if include_optimizer:
        if optimizer is None or scheduler is None:
            raise ValueError("Resume checkpoint requires optimizer and scheduler")
        payload.update(
            {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": None if scaler is None else scaler.state_dict(),
                "rng_state": capture_rng_state(),
            }
        )
    _atomic_torch_save(payload, Path(path))


def load_training_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: object | None,
    scaler: object | None,
    expected_metadata: Mapping[str, object],
    restore_rng: bool,
) -> dict[str, object]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported checkpoint schema")
    if payload.get("metadata") != dict(expected_metadata):
        raise ValueError("Checkpoint metadata does not match the current run metadata")
    _load_downstream_state(model, payload["downstream_state_dict"])
    if optimizer is not None:
        if "optimizer" not in payload:
            raise ValueError("Checkpoint has no optimizer state")
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None:
        if payload.get("scaler") is None:
            raise ValueError("Checkpoint has no AMP scaler state")
        scaler.load_state_dict(payload["scaler"])
    if restore_rng:
        if "rng_state" not in payload:
            raise ValueError("Checkpoint has no RNG state")
        restore_rng_state(payload["rng_state"])
    return payload


def load_initialization_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    expected_metadata: Mapping[str, object],
) -> dict[str, object]:
    """Load model-only best weights while allowing a new loss/output configuration."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported checkpoint schema")
    source_metadata = payload.get("metadata")
    if not isinstance(source_metadata, Mapping):
        raise ValueError("Initialization checkpoint has no metadata mapping")
    for key, expected in expected_metadata.items():
        if key == "config_sha256":
            continue
        actual = source_metadata.get(key)
        if actual != expected:
            raise ValueError(
                f"Initialization checkpoint metadata mismatch for {key}: "
                f"expected={expected!r}, actual={actual!r}"
            )
    _load_downstream_state(model, payload["downstream_state_dict"])
    return payload


def load_finetune_weights(
    path: Path,
    *,
    model: nn.Module,
    expected_metadata: Mapping[str, object],
) -> dict[str, object]:
    """Start a new run from trusted model weights while permitting new training data."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported checkpoint schema")
    source_metadata = payload.get("metadata")
    if not isinstance(source_metadata, Mapping):
        raise ValueError("Finetune checkpoint has no metadata mapping")
    for key in ("backbone_identity", "backbone_sha256", "label_mapping", "input_size", "architecture"):
        if source_metadata.get(key) != expected_metadata.get(key):
            raise ValueError(f"Finetune checkpoint metadata mismatch for {key}")
    _load_downstream_state(model, payload["downstream_state_dict"])
    return payload
