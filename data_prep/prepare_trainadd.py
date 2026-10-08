"""Package the 1027 processed images under their original training filenames."""

import csv
import hashlib
import json
import os
import tarfile
from pathlib import Path

ROOT = Path(os.environ.get("AIC_DATA_ROOT", Path(__file__).resolve().parents[1]))


PROCESSED_ROOT = ROOT / "dataset/scene_processed_all_v1"
SPLIT_PATH = ROOT / "splits/v1/trainadd.txt"
ARCHIVE_PATH = ROOT / "dataset/trainadd_1027.tar"


def build_entries(rows: list[dict], train_names: set[str], processed_root: Path):
    entries = []
    seen = set()
    for row in rows:
        name = row["source_name"]
        if name not in train_names:
            raise ValueError(f"image outside original training split: {name}")
        if name in seen or Path(name).name != name or Path(name).suffix.lower() != ".png":
            raise ValueError(f"duplicate or unsafe training filename: {name}")
        seen.add(name)
        processed = (processed_root / row["processed_file"]).resolve()
        if not processed.is_relative_to(processed_root.resolve()) or not processed.is_file():
            raise ValueError(f"missing or unsafe processed image: {processed}")
        entries.append((name, processed))
    return sorted(entries)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if SPLIT_PATH.exists() or ARCHIVE_PATH.exists():
        raise FileExistsError("trainadd split or archive already exists")
    train_names = set((ROOT / "splits/v1/train.txt").read_text(
        encoding="utf-8-sig").splitlines())
    val_names = set((ROOT / "splits/v1/val.txt").read_text(
        encoding="utf-8-sig").splitlines())
    with (PROCESSED_ROOT / "manifest.csv").open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    entries = build_entries(rows, train_names, PROCESSED_ROOT)
    if len(entries) != 1027 or any(name in val_names for name, _ in entries):
        raise ValueError("expected 1027 unique training images and no validation images")
    masks = ROOT / "dataset/train/train/masks"
    missing_masks = [name for name, _ in entries if not (masks / name).is_file()]
    if missing_masks:
        raise FileNotFoundError(f"missing original mask: {missing_masks[0]}")

    with tarfile.open(ARCHIVE_PATH, "w") as archive:
        for index, (name, processed) in enumerate(entries, 1):
            archive.add(processed, arcname=name, recursive=False)
            if index % 200 == 0:
                print(f"packed {index}/{len(entries)}", flush=True)
    SPLIT_PATH.write_text("".join(f"{name}\n" for name, _ in entries), encoding="utf-8")
    report = {"image_count": len(entries), "archive_bytes": ARCHIVE_PATH.stat().st_size,
              "archive_sha256": file_sha256(ARCHIVE_PATH),
              "split_file": str(SPLIT_PATH), "trainadd_image_dir": "trainadd",
              "mask_dir": "train/masks"}
    (ROOT / "dataset/trainadd_1027_bundle.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
