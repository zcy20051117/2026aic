from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from aic_dino_m2f.engine.manifest import assert_project_path, build_input_manifest


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_build_manifest_records_identity_hashes_counts_and_protected_roots(tmp_path: Path) -> None:
    train_list = tmp_path / "train.txt"
    val_list = tmp_path / "val.txt"
    fold_assignment = tmp_path / "fold_assignment.csv"
    backbone = tmp_path / "backbone.pth"
    train_list.write_text("a.png\nb.png\n", encoding="utf-8")
    val_list.write_text("c.png\n", encoding="utf-8")
    fold_assignment.write_text("filename,fold\na.png,1\nb.png,2\nc.png,0\n", encoding="utf-8")
    backbone.write_bytes(b"frozen-dino")

    manifest = build_input_manifest(
        model_identity="facebook/dinov3-vitl16-pretrain-lvd1689m",
        backbone_files=[backbone],
        train_list=train_list,
        val_list=val_list,
        fold_assignment=fold_assignment,
        protected_roots={
            "images": tmp_path / "images",
            "masks": tmp_path / "masks",
            "test1": tmp_path / "test_1",
        },
    )

    assert manifest["model_identity"] == "facebook/dinov3-vitl16-pretrain-lvd1689m"
    assert manifest["split"]["train_count"] == 2
    assert manifest["split"]["val_count"] == 1
    assert manifest["split"]["fold_assignment_sha256"] == _sha256(fold_assignment)
    assert manifest["backbone_files"][0]["sha256"] == _sha256(backbone)
    assert set(manifest["protected_roots"]) == {"images", "masks", "test1"}


def test_assert_project_path_rejects_outputs_outside_baseline(tmp_path: Path) -> None:
    project_root = tmp_path / "baseline_v1"
    project_root.mkdir()
    assert assert_project_path(project_root / "outputs" / "run", project_root) == (
        project_root / "outputs" / "run"
    ).resolve()

    with pytest.raises(ValueError, match="outside project root"):
        assert_project_path(tmp_path / "other_experiment", project_root)
