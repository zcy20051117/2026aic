from pathlib import Path

import pytest
import torch
from torch import nn

from aic_dino_m2f.engine.checkpoint import load_finetune_weights, downstream_state_dict


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Linear(2, 2)
        self.adapter = nn.Linear(2, 2)


def test_finetune_loads_weights_when_training_manifest_changes(tmp_path: Path) -> None:
    source = TinyModel()
    checkpoint = tmp_path / "best.pth"
    metadata = {
        "config_sha256": "old", "backbone_identity": "dinov3", "backbone_sha256": "abc",
        "train_split_sha256": "oldtrain", "val_split_sha256": "val", "label_mapping": {"0": 255},
        "input_size": 1024, "effective_batch": 3, "architecture": "m2f",
    }
    torch.save({"schema_version": 1, "downstream_state_dict": downstream_state_dict(source),
                "epoch": 34, "global_step": 1000, "metadata": metadata}, checkpoint)
    restored = TinyModel()
    target = {**metadata, "config_sha256": "new", "train_split_sha256": "newtrain", "loveda_split_sha256": "mix"}

    payload = load_finetune_weights(checkpoint, model=restored, expected_metadata=target)

    assert payload["epoch"] == 34
    for name, value in source.adapter.state_dict().items():
        torch.testing.assert_close(restored.adapter.state_dict()[name], value)
    with pytest.raises(ValueError, match="architecture"):
        load_finetune_weights(checkpoint, model=TinyModel(), expected_metadata={**target, "architecture": "other"})
