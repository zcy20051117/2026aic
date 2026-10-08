from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one checkpoint on frozen Fold0 only")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from aic_dino_m2f.runtime import load_evaluation_runtime
    from aic_dino_m2f.engine.manifest import assert_project_path
    from aic_dino_m2f.engine.safety import require_fold0_list
    from aic_dino_m2f.engine.validate import validate_fold0

    model, loader, device, config = load_evaluation_runtime(args.config, args.checkpoint)
    configured = Path(config["data"]["val_list"])
    require_fold0_list(args.split or configured, configured)
    metrics, confusion = validate_fold0(
        model=model,
        loader=loader,
        device=device,
        expected_samples=int(config["data"]["val_count"]),
        query_chunk=int(config["validation"]["query_chunk"]),
        description="evaluate Fold0",
    )
    output = assert_project_path(args.output, Path(config["project_root"]))
    output.mkdir(parents=True, exist_ok=False)
    serializable = {**metrics, "per_class_iou": metrics["per_class_iou"].tolist()}
    (output / "metrics.json").write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    np.savetxt(output / "confusion.csv", confusion, fmt="%d", delimiter=",")


if __name__ == "__main__":
    main()
