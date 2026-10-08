from __future__ import annotations

import argparse
import json
from pathlib import Path

from aic_dino_m2f.config import load_and_validate_config
from aic_dino_m2f.data.dataset import _read_names
from aic_dino_m2f.data.loveda_mix import discover_loveda_pairs, select_vehicle_top_names


def main() -> None:
    parser = argparse.ArgumentParser(description="Create reproducible AIC Vehicle + LoveDA Train manifests")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_and_validate_config(args.config)
    data = config["data"]
    paths = [Path(data[key]) for key in ("aic_selected_list", "loveda_rural_list", "loveda_urban_list")]
    summary_path = paths[0].parent / "selection_summary.json"
    if all(path.is_file() for path in paths) and summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if len(_read_names(paths[0])) != int(data["aic_selected_count"]):
            raise ValueError("Existing AIC selection has the wrong count")
        if sum(len(_read_names(path)) for path in paths[1:]) != int(data["loveda_count"]):
            raise ValueError("Existing LoveDA selections have the wrong count")
        print(json.dumps(summary, indent=2), flush=True)
        return
    if any(path.exists() for path in (*paths, summary_path)):
        raise FileExistsError("Partial manifest set exists; inspect it before regenerating")

    selected, stats = select_vehicle_top_names(
        Path(data["train_list"]), Path(data["val_list"]), Path(data["masks_dir"]),
        limit=int(data["aic_selected_count"]), seed=int(config["seed"]),
        expected_size=(int(data["image_size"]),) * 2,
    )
    loveda_root = Path(data["loveda_root"])
    regions = {}
    for region in ("Rural", "Urban"):
        base = loveda_root / "Train" / region
        regions[region] = discover_loveda_pairs(base / "images_png", base / "masks_png")
    if sum(map(len, regions.values())) != int(data["loveda_count"]):
        raise ValueError("LoveDA Train pair count differs from configuration")

    summary = {
        "selection": "highest AIC Vehicle pixel fraction; seeded hash breaks ties",
        "seed": int(config["seed"]),
        **stats,
        "loveda_rural": len(regions["Rural"]),
        "loveda_urban": len(regions["Urban"]),
        "total_training_samples": len(selected) + sum(map(len, regions.values())),
    }
    paths[0].parent.mkdir(parents=True, exist_ok=True)
    for path, names in zip(paths, (selected, regions["Rural"], regions["Urban"])):
        with path.open("x", encoding="utf-8") as handle:
            handle.write("\n".join(names) + "\n")
    with summary_path.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
