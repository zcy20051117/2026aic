from __future__ import annotations

import time
from typing import Iterable

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm

from aic_dino_m2f.metrics.confusion import ConfusionMatrix8
from aic_dino_m2f.models.postprocess import semantic_scores


def _target_to_device(target: dict[str, object], device: torch.device) -> dict[str, object]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in target.items()
    }


@torch.inference_mode()
def validate_fold0(
    *,
    model: nn.Module,
    loader: Iterable[dict[str, object]],
    device: torch.device,
    expected_samples: int,
    query_chunk: int,
    description: str = "val Fold0",
    criterion: nn.Module | None = None,
    loss_seed: int = 0,
) -> tuple[dict[str, object], np.ndarray]:
    model.eval()
    if criterion is not None:
        criterion.eval()
    confusion = ConfusionMatrix8()
    loss_totals: dict[str, float] = {}
    loss_generator = torch.Generator(device=device).manual_seed(int(loss_seed))
    sample_count = 0
    started = time.perf_counter()
    progress = tqdm(loader, total=len(loader), desc=description, dynamic_ncols=True)
    for batch in progress:
        images = batch["image"].to(device, non_blocking=True)
        outputs = model(images)
        if criterion is not None:
            targets = [_target_to_device(target, device) for target in batch["target"]]
            losses = criterion(outputs, targets, generator=loss_generator)
            for name, value in losses.items():
                loss_totals[name] = loss_totals.get(name, 0.0) + float(value) * int(images.shape[0])
        scores = semantic_scores(
            outputs["pred_logits"],
            outputs["pred_masks"],
            output_size=tuple(batch["semantic"].shape[-2:]),
            query_chunk=query_chunk,
        )
        predictions = scores.argmax(dim=1).cpu().numpy()
        ground_truth = batch["semantic"].cpu().numpy()
        for target, prediction in zip(ground_truth, predictions, strict=True):
            confusion.update(target, prediction)
        sample_count += int(images.shape[0])
        partial = confusion.compute(require_all_classes=False)
        elapsed = max(time.perf_counter() - started, 1.0e-9)
        memory_gib = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0
        postfix = {
            "mIoU": f"{partial['miou']:.5f}",
            "samples": sample_count,
            "ips": f"{sample_count / elapsed:.2f}",
            "mem": f"{memory_gib:.2f}G",
        }
        if "loss_total" in loss_totals:
            postfix["valloss"] = f"{loss_totals['loss_total'] / sample_count:.4f}"
        progress.set_postfix(**postfix, refresh=True)
    if sample_count != expected_samples:
        raise RuntimeError(f"Fold0 validation saw {sample_count} samples, expected {expected_samples}")
    metrics = confusion.compute(require_all_classes=True)
    metrics.update({name: total / sample_count for name, total in loss_totals.items()})
    metrics.update(samples=sample_count, seconds=time.perf_counter() - started)
    return metrics, confusion.matrix.copy()
