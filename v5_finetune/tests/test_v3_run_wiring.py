from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
def test_formal_run_starts_from_v4_best_weights_for_ten_epochs() -> None:
    launch = (ROOT / "scripts" / "launch_formal.sh").read_text(encoding="utf-8")
    assert "--finetune-from $checkpoint" in launch
    assert "--resume" not in launch
    assert "--end-epoch 10" in launch
    assert "scripts/prepare_loveda_mix.py" in launch
    assert "checkpoints/best.pth" in launch


def test_v5_paths_are_isolated_from_v4_outputs() -> None:
    config = (ROOT / "configs" / "baseline_v1.yaml").read_text(encoding="utf-8")
    assert "project_root: ${AIC_PROJECT_ROOT}/v5_finetune" in config
    assert (
        "output_dir: ${AIC_OUTPUT_ROOT}/"
        "outputs/dinov3_vitl16_m2f_loveda_vehicle_mix_v5"
    ) in config


def test_shell_scripts_are_exported_with_linux_line_endings() -> None:
    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert "*.sh text eol=lf" in attributes.splitlines()
