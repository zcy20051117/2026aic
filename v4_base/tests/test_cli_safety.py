from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from aic_dino_m2f.engine.safety import prepare_run_directory, require_fold0_list
from scripts.validate_prediction_format import validate_prediction_directory
from aic_dino_m2f.train import _assert_metric_match


def test_formal_run_refuses_escape_and_existing_output_without_resume(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    with pytest.raises(ValueError, match="outside"):
        prepare_run_directory(project, tmp_path / "elsewhere", resume=False)
    output = project / "outputs" / "run"
    output.mkdir(parents=True)
    (output / "sentinel.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="resume"):
        prepare_run_directory(project, output, resume=False)
    assert prepare_run_directory(project, output, resume=True) == output.resolve()
    assert (output / "sentinel.txt").read_text(encoding="utf-8") == "keep"


def test_evaluation_accepts_only_the_configured_fold0_list(tmp_path: Path) -> None:
    fold0 = tmp_path / "val.txt"
    other = tmp_path / "test.txt"
    fold0.write_text("a.png\n", encoding="utf-8")
    other.write_text("b.png\n", encoding="utf-8")
    assert require_fold0_list(fold0, fold0) == fold0.resolve()
    with pytest.raises(ValueError, match="Fold0"):
        require_fold0_list(other, fold0)


def test_prediction_format_checks_names_mode_size_and_official_ids(tmp_path: Path) -> None:
    expected = ["a.png", "b.png"]
    for name in expected:
        Image.fromarray(np.ones((4, 5), dtype=np.uint8)).save(tmp_path / name)
    report = validate_prediction_directory(tmp_path, expected, expected_size=(4, 5))
    assert report == {"files": 2, "width": 5, "height": 4}

    Image.fromarray(np.zeros((4, 5), dtype=np.uint8)).save(tmp_path / "a.png")
    with pytest.raises(ValueError, match="IDs"):
        validate_prediction_directory(tmp_path, expected, expected_size=(4, 5))


def test_final_best_metric_check_uses_overall_and_per_class_tolerance() -> None:
    recorded = {"miou": 0.5, "per_class_iou_by_name": {"Background": 0.4, "Road": 0.6}}
    _assert_metric_match(recorded, {"miou": 0.50001, "per_class_iou_by_name": {"Background": 0.40001, "Road": 0.59999}})
    with pytest.raises(RuntimeError, match="iou_Road"):
        _assert_metric_match(recorded, {"miou": 0.5, "per_class_iou_by_name": {"Background": 0.4, "Road": 0.61}})
