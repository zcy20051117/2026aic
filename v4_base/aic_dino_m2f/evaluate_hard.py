from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from aic_dino_m2f.data.dataset import AICSegmentationDataset
from aic_dino_m2f.data.hard_validation import degrade_fixed_v2_like
from aic_dino_m2f.engine.manifest import assert_project_path
from aic_dino_m2f.engine.train_epoch import collate_samples
from aic_dino_m2f.engine.validate import validate_fold0
from aic_dino_m2f.runtime import build_criterion, load_inference_runtime


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate best checkpoint on fixed degraded Fold0")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model, device, config = load_inference_runtime(args.config, args.checkpoint)
    data = config["data"]
    val_dataset = AICSegmentationDataset(
        Path(data["val_list"]), Path(data["images_dir"]), Path(data["masks_dir"]),
        expected_size=(int(data["image_size"]),) * 2,
        image_transform=degrade_fixed_v2_like,
    )
    loader = DataLoader(
        val_dataset, batch_size=int(config["validation"]["batch_size"]), shuffle=False,
        num_workers=int(config["training"]["num_workers"]), pin_memory=True,
        collate_fn=collate_samples,
    )
    metrics, confusion = validate_fold0(
        model=model, criterion=build_criterion(config).to(device), loader=loader,
        device=device, expected_samples=int(data["val_count"]),
        query_chunk=int(config["validation"]["query_chunk"]),
        description="fixed hard val", loss_seed=int(config["validation"]["loss_seed"]),
    )
    output = assert_project_path(args.output, Path(config["project_root"]))
    output.mkdir(parents=True, exist_ok=False)
    payload = {key: value.tolist() if isinstance(value, np.ndarray) else value
               for key, value in metrics.items()}
    payload["degradation"] = "deterministic v2-like PIL diagnostic; not exact trainadd/OpenCV reproduction"
    (output / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    np.savetxt(output / "confusion.csv", confusion, fmt="%d", delimiter=",")
    print(json.dumps({"miou": metrics["miou"], "valloss": metrics["loss_total"],
                      "output": str(output)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
