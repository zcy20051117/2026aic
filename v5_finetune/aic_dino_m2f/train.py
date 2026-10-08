from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np

from aic_dino_m2f.engine.checkpoint import (
    load_finetune_weights,
    load_initialization_checkpoint,
    load_training_checkpoint,
    save_training_checkpoint,
)
from aic_dino_m2f.engine.safety import prepare_run_directory
from aic_dino_m2f.engine.state import EarlyStoppingState
from aic_dino_m2f.engine.train_epoch import train_one_epoch
from aic_dino_m2f.engine.validate import validate_fold0


@dataclass
class BarrenCheckpointState:
    min_miou: float
    best_iou: float = -math.inf
    best_epoch: int = -1

    def update(self, metrics: dict[str, object], *, epoch: int) -> bool:
        miou = float(metrics["miou"])
        per_class = metrics["per_class_iou_by_name"]
        if not isinstance(per_class, dict) or "Barren" not in per_class:
            raise ValueError("Validation metrics do not contain Barren IoU")
        barren_iou = float(per_class["Barren"])
        if not math.isfinite(miou) or not math.isfinite(barren_iou):
            raise ValueError("Barren checkpoint metrics must be finite")
        if miou < float(self.min_miou) or barren_iou <= self.best_iou:
            return False
        self.best_iou = barren_iou
        self.best_epoch = int(epoch)
        return True


def _json_metrics(metrics: dict[str, object]) -> dict[str, object]:
    return {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in metrics.items()
    }


def _append_epoch_csv(path: Path, row: dict[str, object]) -> None:
    flat = {key: value for key, value in row.items() if not isinstance(value, (dict, list, np.ndarray))}
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat))
        if write_header:
            writer.writeheader()
        writer.writerow(flat)
        handle.flush()


def _write_barren_metrics(
    path: Path,
    metrics: dict[str, object],
    *,
    epoch: int,
    min_miou: float,
) -> None:
    payload = {
        **_json_metrics(metrics),
        "selection_epoch": int(epoch),
        "selection_min_miou": float(min_miou),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _assert_metric_match(recorded: dict[str, object], current: dict[str, object], tolerance: float = 1.0e-4) -> None:
    comparisons = [("miou", float(recorded["miou"]), float(current["miou"]))]
    recorded_classes = recorded["per_class_iou_by_name"]
    current_classes = current["per_class_iou_by_name"]
    comparisons.extend(
        (f"iou_{name}", float(recorded_classes[name]), float(current_classes[name]))
        for name in recorded_classes
    )
    mismatches = [
        f"{name}: recorded={expected:.8f}, current={actual:.8f}"
        for name, expected, actual in comparisons
        if abs(expected - actual) > tolerance
    ]
    if mismatches:
        raise RuntimeError("Best-checkpoint validation mismatch: " + "; ".join(mismatches))


def _resolve_end_epoch(
    *,
    start_epoch: int,
    configured_max_epochs: int,
    requested_end_epoch: int | None,
) -> int:
    end_epoch = configured_max_epochs if requested_end_epoch is None else min(
        int(requested_end_epoch), configured_max_epochs
    )
    if end_epoch < start_epoch:
        raise ValueError(
            f"Requested end epoch {end_epoch} is before start epoch {start_epoch}"
        )
    return end_epoch


def main() -> None:
    parser = argparse.ArgumentParser(description="Train frozen DINOv3 + Mask2Former baseline v1")
    parser.add_argument("--config", type=Path, required=True)
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument("--resume", type=Path)
    checkpoint_group.add_argument("--init-checkpoint", type=Path)
    checkpoint_group.add_argument("--finetune-from", type=Path)
    parser.add_argument("--end-epoch", type=int)
    args = parser.parse_args()
    from aic_dino_m2f.config import load_and_validate_config
    from aic_dino_m2f.runtime import build_training_runtime

    config = load_and_validate_config(args.config)
    project_root = Path(config["project_root"])
    output = prepare_run_directory(project_root, Path(config["training"]["output_dir"]), resume=args.resume is not None)
    runtime = build_training_runtime(config)
    model = runtime["model"]
    optimizer = runtime["optimizer"]
    scheduler = runtime["scheduler"]
    start_epoch = 1
    global_step = 0
    early = EarlyStoppingState(patience=int(config["training"]["patience"]))
    if args.resume is not None:
        payload = load_training_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            expected_metadata=runtime["checkpoint_metadata"],
            restore_rng=True,
        )
        start_epoch = int(payload["epoch"]) + 1
        global_step = int(payload["global_step"])
        early = EarlyStoppingState.from_state_dict(payload["early_stopping"])
    elif args.init_checkpoint is not None:
        payload = load_initialization_checkpoint(
            args.init_checkpoint,
            model=model,
            expected_metadata=runtime["checkpoint_metadata"],
        )
        start_epoch = int(payload["epoch"]) + 1
        global_step = int(payload["global_step"])
        early = EarlyStoppingState.from_state_dict(payload["early_stopping"])
        scheduler.set_step_count(global_step)
    elif args.finetune_from is not None:
        load_finetune_weights(
            args.finetune_from,
            model=model,
            expected_metadata=runtime["checkpoint_metadata"],
        )

    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    loss_seed = int(config["validation"]["loss_seed"])
    barren_min_miou = float(config.get("selection", {}).get("barren_min_miou", 0.792))
    barren_state = BarrenCheckpointState(min_miou=barren_min_miou)
    barren_metrics_path = output / "best_barren_metrics.json"
    if args.resume is not None and barren_metrics_path.is_file():
        recorded_barren = json.loads(barren_metrics_path.read_text(encoding="utf-8"))
        barren_state.best_iou = float(recorded_barren["per_class_iou_by_name"]["Barren"])
        barren_state.best_epoch = int(recorded_barren["selection_epoch"])
    if args.init_checkpoint is not None or args.finetune_from is not None:
        initial_metrics, initial_confusion = validate_fold0(
            model=model,
            criterion=runtime["criterion"],
            loader=runtime["val_loader"],
            device=runtime["device"],
            expected_samples=int(config["data"]["val_count"]),
            query_chunk=int(config["validation"]["query_chunk"]),
            description=f"initial epoch {start_epoch - 1} Fold0",
            loss_seed=loss_seed,
        )
        if args.init_checkpoint is not None and abs(float(initial_metrics["miou"]) - float(early.best_score)) > 1.0e-4:
            raise RuntimeError(
                "Initialization checkpoint mIoU mismatch: "
                f"checkpoint={early.best_score:.8f}, validation={float(initial_metrics['miou']):.8f}"
            )
        if args.finetune_from is not None:
            early.update(float(initial_metrics["miou"]), epoch=0)
        save_training_checkpoint(
            checkpoints / "best.pth",
            model=model,
            optimizer=None,
            scheduler=None,
            scaler=None,
            epoch=start_epoch - 1,
            global_step=global_step,
            early_stopping=early,
            metadata=runtime["checkpoint_metadata"],
            include_optimizer=False,
        )
        serialized_initial = json.dumps(_json_metrics(initial_metrics), indent=2)
        (output / "initial_metrics.json").write_text(serialized_initial, encoding="utf-8")
        (output / "best_metrics.json").write_text(serialized_initial, encoding="utf-8")
        np.savetxt(
            output / f"confusion_epoch_{start_epoch - 1:03d}.csv",
            initial_confusion,
            fmt="%d",
            delimiter=",",
        )
        if barren_state.update(initial_metrics, epoch=start_epoch - 1):
            save_training_checkpoint(
                checkpoints / "best_barren.pth",
                model=model,
                optimizer=None,
                scheduler=None,
                scaler=None,
                epoch=start_epoch - 1,
                global_step=global_step,
                early_stopping=early,
                metadata=runtime["checkpoint_metadata"],
                include_optimizer=False,
            )
            _write_barren_metrics(
                barren_metrics_path,
                initial_metrics,
                epoch=start_epoch - 1,
                min_miou=barren_min_miou,
            )

    run_end_epoch = _resolve_end_epoch(
        start_epoch=start_epoch,
        configured_max_epochs=int(config["training"]["max_epochs"]),
        requested_end_epoch=args.end_epoch,
    )
    for epoch in range(start_epoch, run_end_epoch + 1):
        train_metrics, global_step = train_one_epoch(
            model=model,
            criterion=runtime["criterion"],
            loader=runtime["train_loader"],
            optimizer=optimizer,
            scheduler=scheduler,
            device=runtime["device"],
            accumulation_steps=int(runtime["accumulation_steps"]),
            epoch=epoch,
            max_epochs=run_end_epoch,
            global_step=global_step,
            use_bf16=bool(config["training"]["bf16"]),
            gradient_clip_norm=float(config["training"]["gradient_clip_norm"]),
        )
        val_metrics, confusion = validate_fold0(
            model=model,
            criterion=runtime["criterion"],
            loader=runtime["val_loader"],
            device=runtime["device"],
            expected_samples=int(config["data"]["val_count"]),
            query_chunk=int(config["validation"]["query_chunk"]),
            loss_seed=loss_seed,
        )
        improved = early.update(float(val_metrics["miou"]), epoch=epoch)
        barren_improved = barren_state.update(val_metrics, epoch=epoch)
        per_class = {
            f"val_iou_{name}": value
            for name, value in val_metrics["per_class_iou_by_name"].items()
        }
        per_class_precision = {
            f"val_precision_{name}": value
            for name, value in val_metrics["per_class_precision_by_name"].items()
        }
        per_class_recall = {
            f"val_recall_{name}": value
            for name, value in val_metrics["per_class_recall_by_name"].items()
        }
        row = {"epoch": epoch, "global_step": global_step, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items() if not isinstance(v, (dict, np.ndarray))}, **per_class, **per_class_precision, **per_class_recall, "best_miou": early.best_score, "best_epoch": early.best_epoch, "best_barren_iou": barren_state.best_iou, "best_barren_epoch": barren_state.best_epoch, "bad_epochs": early.bad_epochs}
        _append_epoch_csv(output / "epoch_metrics.csv", row)
        np.savetxt(output / f"confusion_epoch_{epoch:03d}.csv", confusion, fmt="%d", delimiter=",")
        if improved:
            save_training_checkpoint(
                checkpoints / "best.pth", model=model, optimizer=None, scheduler=None, scaler=None,
                epoch=epoch, global_step=global_step, early_stopping=early,
                metadata=runtime["checkpoint_metadata"], include_optimizer=False,
            )
            (output / "best_metrics.json").write_text(json.dumps(_json_metrics(val_metrics), indent=2), encoding="utf-8")
        if barren_improved:
            save_training_checkpoint(
                checkpoints / "best_barren.pth", model=model, optimizer=None, scheduler=None, scaler=None,
                epoch=epoch, global_step=global_step, early_stopping=early,
                metadata=runtime["checkpoint_metadata"], include_optimizer=False,
            )
            _write_barren_metrics(
                barren_metrics_path,
                val_metrics,
                epoch=epoch,
                min_miou=barren_min_miou,
            )
        save_training_checkpoint(
            checkpoints / "last.pth", model=model, optimizer=optimizer, scheduler=scheduler, scaler=None,
            epoch=epoch, global_step=global_step, early_stopping=early,
            metadata=runtime["checkpoint_metadata"], include_optimizer=True,
        )
        print(json.dumps(row, default=str), flush=True)
        if early.should_stop:
            break

    recorded_best = json.loads((output / "best_metrics.json").read_text(encoding="utf-8"))
    load_training_checkpoint(
        checkpoints / "best.pth", model=model, optimizer=None, scheduler=None, scaler=None,
        expected_metadata=runtime["checkpoint_metadata"], restore_rng=False,
    )
    final_metrics, final_confusion = validate_fold0(
        model=model, criterion=runtime["criterion"], loader=runtime["val_loader"],
        device=runtime["device"], expected_samples=int(config["data"]["val_count"]),
        query_chunk=int(config["validation"]["query_chunk"]), description="final best Fold0",
        loss_seed=loss_seed,
    )
    _assert_metric_match(recorded_best, final_metrics)
    (output / "final_best_metrics.json").write_text(json.dumps(_json_metrics(final_metrics), indent=2), encoding="utf-8")
    np.savetxt(output / "final_best_confusion.csv", final_confusion, fmt="%d", delimiter=",")


if __name__ == "__main__":
    main()
