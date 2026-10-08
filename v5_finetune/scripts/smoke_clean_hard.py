from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from aic_dino_m2f.config import load_and_validate_config
from aic_dino_m2f.engine.train_epoch import collate_samples
from aic_dino_m2f.runtime import build_training_runtime


def _targets_to_device(targets: list[dict[str, object]], device: torch.device):
    return [
        {key: value.to(device) if isinstance(value, torch.Tensor) else value
         for key, value in target.items()}
        for target in targets
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean + hard V4 CUDA training/validation smoke")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_and_validate_config(args.config)
    runtime = build_training_runtime(config, train_limit=8)
    full_training = runtime["train_loader"].dataset.dataset
    assert len(full_training) == int(config["data"]["train_count"])
    batch = collate_samples([full_training[0], full_training[len(full_training.original)]])
    assert batch["source"] == ["original", "hard"]
    device = runtime["device"]
    model = runtime["model"]
    criterion = runtime["criterion"]
    images = batch["image"].to(device)
    targets = _targets_to_device(batch["target"], device)
    runtime["optimizer"].zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bool(config["training"]["bf16"])):
        outputs = model(images)
        train_losses = criterion(outputs, targets)
    trainloss = float(train_losses["loss_total"].detach())
    train_losses["loss_total"].backward()
    runtime["optimizer"].zero_grad(set_to_none=True)
    del outputs, train_losses, images, targets
    model.eval()
    criterion.eval()
    val_batch = next(iter(runtime["val_loader"]))
    with torch.inference_mode():
        outputs = model(val_batch["image"].to(device))
        val_losses = criterion(
            outputs, _targets_to_device(val_batch["target"], device),
            generator=torch.Generator(device=device).manual_seed(int(config["validation"]["loss_seed"])),
        )
    report = {
        "status": "passed", "original_count": len(full_training.original),
        "hard_count": len(full_training.hard), "val_count": len(runtime["val_loader"].dataset),
        "sources": batch["source"], "trainloss": trainloss,
        "valloss": float(val_losses["loss_total"]),
        "barren_class_weight": float(criterion.empty_weight[4]),
        "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
