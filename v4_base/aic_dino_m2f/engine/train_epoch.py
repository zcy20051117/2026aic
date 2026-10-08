from __future__ import annotations

import time
from typing import Iterable

import torch
from torch import Tensor, nn
from tqdm.auto import tqdm

from .optim import AccumulationController


def collate_samples(samples: list[dict[str, object]]) -> dict[str, object]:
    return {
        "image": torch.stack([sample["image"] for sample in samples]),
        "semantic": torch.stack([sample["semantic"] for sample in samples]),
        "filename": [str(sample["filename"]) for sample in samples],
        "source": [str(sample.get("source", "original")) for sample in samples],
        "target": [sample["target"] for sample in samples],
    }


def _target_to_device(target: dict[str, object], device: torch.device) -> dict[str, object]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, Tensor) else value
        for key, value in target.items()
    }


def train_one_epoch(
    *,
    model: nn.Module,
    criterion: nn.Module,
    loader: Iterable[dict[str, object]],
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    device: torch.device,
    accumulation_steps: int,
    epoch: int,
    max_epochs: int,
    global_step: int,
    use_bf16: bool,
    gradient_clip_norm: float,
) -> tuple[dict[str, float], int]:
    model.train()
    total_microbatches = len(loader)
    controller = AccumulationController(accumulation_steps, total_microbatches)
    optimizer.zero_grad(set_to_none=True)
    totals: dict[str, float] = {}
    sample_count = 0
    hard_samples = 0
    started = time.perf_counter()
    progress = tqdm(loader, total=total_microbatches, desc=f"train {epoch}/{max_epochs}", dynamic_ncols=True)
    for microbatch_index, batch in enumerate(progress):
        images = batch["image"].to(device, non_blocking=True)
        targets = [_target_to_device(target, device) for target in batch["target"]]
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=bool(use_bf16 and device.type == "cuda"),
        ):
            outputs = model(images)
            losses = criterion(outputs, targets)
            loss = losses["loss_total"]
        controller.scale_loss(loss, microbatch_index).backward()
        if controller.should_step(microbatch_index):
            if gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    gradient_clip_norm,
                )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1
        batch_size = int(images.shape[0])
        sample_count += batch_size
        hard_samples += sum(source == "hard" for source in batch["source"])
        for name, value in losses.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach()) * batch_size
        elapsed = max(time.perf_counter() - started, 1.0e-9)
        memory_gib = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0
        progress.set_postfix(
            trainloss=f"{totals['loss_total'] / sample_count:.4f}",
            hard=f"{hard_samples / sample_count:.1%}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            samples=sample_count,
            ips=f"{sample_count / elapsed:.2f}",
            mem=f"{memory_gib:.2f}G",
            refresh=True,
        )
    if sample_count == 0:
        raise RuntimeError("Training loader produced no samples")
    metrics = {name: value / sample_count for name, value in totals.items()}
    metrics.update(
        samples=float(sample_count), hard_samples=float(hard_samples),
        hard_fraction=hard_samples / sample_count,
        seconds=time.perf_counter() - started,
    )
    return metrics, global_step
