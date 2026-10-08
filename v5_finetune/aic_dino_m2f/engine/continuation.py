from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping

import torch


def validate_continuation_configs(source: Mapping[str, object], target: Mapping[str, object]) -> None:
    """Permit only a new output directory and early-stopping patience."""
    adjusted = deepcopy(dict(target))
    source_training = source["training"]
    target_training = adjusted["training"]
    if not isinstance(source_training, Mapping) or not isinstance(target_training, dict):
        raise ValueError("Training configurations must be mappings")
    target_training["patience"] = source_training["patience"]
    target_training["output_dir"] = source_training["output_dir"]
    if adjusted != dict(source):
        raise ValueError("Continuation may change only patience and output directory")
    if target["training"]["output_dir"] == source_training["output_dir"]:
        raise ValueError("Continuation output directory must differ from the source")


def prepare_continuation_output(
    source_output: Path,
    target_output: Path,
    source_metadata: Mapping[str, object],
    target_metadata: Mapping[str, object],
    expected_last_epoch: int,
    new_patience: int,
) -> None:
    """Copy a stopped run into a new output, preserving optimizer/scheduler/RNG state."""
    source = Path(source_output).resolve(strict=True)
    target = Path(target_output).resolve(strict=False)
    if target == source or source in target.parents:
        raise ValueError("Continuation output must be separate from the source output")
    if target.exists():
        raise FileExistsError(f"Continuation output already exists: {target}")
    if int(new_patience) <= 0:
        raise ValueError("New early-stopping patience must be positive")
    without_config = lambda metadata: {key: value for key, value in metadata.items() if key != "config_sha256"}
    if without_config(source_metadata) != without_config(target_metadata):
        raise ValueError("Non-config checkpoint metadata differs between source and target")

    names = ("last.pth", "best.pth", "best_barren.pth")
    required = ("best_metrics.json", "best_barren_metrics.json", "epoch_metrics.csv")
    for name in names:
        if not (source / "checkpoints" / name).is_file():
            raise FileNotFoundError(source / "checkpoints" / name)
    for name in required:
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)

    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{target.name}.prepare.", dir=target.parent) as temporary:
        staged = Path(temporary)
        (staged / "checkpoints").mkdir()
        last_details: dict[str, object] = {}
        for name in names:
            original = torch.load(source / "checkpoints" / name, map_location="cpu", weights_only=False)
            if original.get("schema_version") != 1 or original.get("metadata") != dict(source_metadata):
                raise ValueError(f"Source checkpoint metadata mismatch: {name}")
            payload = dict(original)
            early = dict(payload["early_stopping"])
            early["patience"] = int(new_patience)
            early["bad_epochs"] = 0
            payload["early_stopping"] = early
            payload["metadata"] = dict(target_metadata)
            if name == "last.pth":
                if int(payload["epoch"]) != int(expected_last_epoch):
                    raise ValueError(f"Unexpected last epoch: {payload['epoch']}")
                if not all(key in payload for key in ("optimizer", "scheduler", "rng_state")):
                    raise ValueError("Last checkpoint lacks resumable training state")
                last_details = {"source_last_epoch": int(payload["epoch"]),
                                "source_global_step": int(payload["global_step"]),
                                "source_best_epoch": int(early["best_epoch"]),
                                "source_best_miou": float(early["best_score"])}
            torch.save(payload, staged / "checkpoints" / name)
            del payload, original

        for name in (*required, "initial_metrics.json"):
            source_file = source / name
            if source_file.is_file():
                shutil.copy2(source_file, staged / name)
        for source_file in source.glob("confusion_epoch_*.csv"):
            shutil.copy2(source_file, staged / source_file.name)
        (staged / "continuation_manifest.json").write_text(
            json.dumps({"source_output": str(source), "new_patience": int(new_patience),
                        **last_details}, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(staged, target)
