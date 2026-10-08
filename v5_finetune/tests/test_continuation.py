from copy import deepcopy
from pathlib import Path

import pytest
import torch

from aic_dino_m2f.config import load_and_validate_config, validate_config
from aic_dino_m2f.engine.continuation import (
    prepare_continuation_output,
    validate_continuation_configs,
)


def test_resume_config_changes_only_patience_and_output() -> None:
    root = Path(__file__).resolve().parents[1]
    source = load_and_validate_config(root / "configs/baseline_v1.yaml")
    configured_resume = load_and_validate_config(root / "configs/resume15.yaml")
    validate_continuation_configs(source, configured_resume)
    assert configured_resume["training"]["patience"] == 10
    target = deepcopy(source)
    target["training"]["patience"] = 10
    target["training"]["output_dir"] = "/project/outputs/resume15"
    assert validate_config(target)["training"]["patience"] == 10
    validate_continuation_configs(source, target)
    target["training"]["learning_rate"] = 2.0e-4
    with pytest.raises(ValueError, match="only patience and output"):
        validate_continuation_configs(source, target)


def test_continuation_preserves_optimizer_and_original_output(tmp_path: Path) -> None:
    source = tmp_path / "original"
    checkpoints = source / "checkpoints"
    checkpoints.mkdir(parents=True)
    old_metadata = {"config_sha256": "old", "train_split_sha256": "same"}
    new_metadata = {"config_sha256": "new", "train_split_sha256": "same"}
    state = {"patience": 4, "best_score": 0.807, "best_epoch": 2, "bad_epochs": 4}
    common = {"schema_version": 1, "downstream_state_dict": {"head.weight": torch.tensor([1.0])},
              "metadata": old_metadata, "early_stopping": state}
    torch.save({**common, "epoch": 6, "global_step": 10044,
                "optimizer": {"marker": 123}, "scheduler": {"step": 10044},
                "rng_state": {"marker": 456}}, checkpoints / "last.pth")
    torch.save({**common, "epoch": 2, "global_step": 3348}, checkpoints / "best.pth")
    torch.save({**common, "epoch": 3, "global_step": 5022}, checkpoints / "best_barren.pth")
    for name in ("best_metrics.json", "best_barren_metrics.json", "epoch_metrics.csv", "confusion_epoch_006.csv"):
        (source / name).write_text(name, encoding="utf-8")

    target = tmp_path / "resume15"
    prepare_continuation_output(
        source_output=source, target_output=target,
        source_metadata=old_metadata, target_metadata=new_metadata,
        expected_last_epoch=6, new_patience=10,
    )

    original = torch.load(checkpoints / "last.pth", weights_only=False)
    resumed = torch.load(target / "checkpoints" / "last.pth", weights_only=False)
    assert original["early_stopping"]["bad_epochs"] == 4
    assert resumed["early_stopping"] == {"patience": 10, "best_score": 0.807, "best_epoch": 2, "bad_epochs": 0}
    assert resumed["optimizer"] == original["optimizer"]
    assert resumed["scheduler"] == original["scheduler"]
    assert resumed["rng_state"] == original["rng_state"]
    assert resumed["epoch"] == 6 and resumed["global_step"] == 10044
    for name in ("best.pth", "best_barren.pth"):
        payload = torch.load(target / "checkpoints" / name, weights_only=False)
        assert payload["metadata"] == new_metadata
    assert (target / "epoch_metrics.csv").read_text(encoding="utf-8") == "epoch_metrics.csv"
    with pytest.raises(FileExistsError):
        prepare_continuation_output(source, target, old_metadata, new_metadata, 6, 10)
