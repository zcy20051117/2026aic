from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from aic_dino_m2f.config import load_and_validate_config, validate_config
from aic_dino_m2f.engine.checkpoint import load_finetune_weights
from aic_dino_m2f.engine.train_epoch import collate_samples, train_one_epoch
from aic_dino_m2f.runtime import _datasets, build_training_runtime


def main() -> None:
    parser = argparse.ArgumentParser(description="One real mixed-source optimizer step at a requested batch size")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch", type=int, required=True)
    args = parser.parse_args()
    if args.batch <= 0:
        raise ValueError("Batch must be positive")
    config = load_and_validate_config(args.config)
    config["training"] = dict(config["training"])
    config["training"]["physical_batch_size"] = args.batch
    config["training"]["target_effective_batch"] = args.batch
    config = validate_config(config)

    runtime = build_training_runtime(config, train_limit=1)
    load_finetune_weights(args.checkpoint, model=runtime["model"], expected_metadata=runtime["checkpoint_metadata"])
    mixed, _ = _datasets(config)
    aic_count = int(config["data"]["aic_selected_count"])
    rural_count = len(mixed.sources[1])
    bases = (0, aic_count, aic_count + rural_count)
    indices = [bases[i % 3] + i // 3 for i in range(args.batch)]
    loader = DataLoader(Subset(mixed, indices), batch_size=args.batch, collate_fn=collate_samples, num_workers=0)
    torch.cuda.reset_peak_memory_stats(runtime["device"])
    metrics, steps = train_one_epoch(
        model=runtime["model"], criterion=runtime["criterion"], loader=loader,
        optimizer=runtime["optimizer"], scheduler=runtime["scheduler"],
        device=runtime["device"], accumulation_steps=1, epoch=1, max_epochs=10,
        global_step=0, use_bf16=bool(config["training"]["bf16"]),
        gradient_clip_norm=float(config["training"]["gradient_clip_norm"]),
    )
    torch.cuda.synchronize(runtime["device"])
    print("BATCH_SMOKE_OK " + json.dumps({
        "batch": args.batch,
        "sources": [mixed[index]["source"] for index in indices],
        "optimizer_steps": steps,
        "train_loss": metrics["loss_total"],
        "max_allocated_gib": round(torch.cuda.max_memory_allocated(runtime["device"]) / 2**30, 2),
        "max_reserved_gib": round(torch.cuda.max_memory_reserved(runtime["device"]) / 2**30, 2),
    }), flush=True)


if __name__ == "__main__":
    main()
