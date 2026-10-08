from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
def test_formal_and_smoke_start_from_dinov3_pretrained_backbone() -> None:
    launch = (ROOT / "scripts" / "launch_formal.sh").read_text(encoding="utf-8")
    smoke = (ROOT / "scripts" / "smoke_clean_hard.py").read_text(encoding="utf-8")

    assert "--init-checkpoint" not in launch
    assert "--resume" not in launch
    assert "--end-epoch 50" in launch
    assert "aic_dino_m2f.evaluate_hard" in launch
    assert "dinov3_vitl16_m2f_barren_clean_hard_v4_b4a2/checkpoints/best.pth" in launch
    assert "session=aic_dino_m2f_clean_hard_v4_b4a2" in launch
    assert "load_initialization_checkpoint" not in smoke
    assert "build_training_runtime(config, train_limit=8)" in smoke
    assert 'int(config["training"]["physical_batch_size"])' in smoke
    assert 'runtime["accumulation_steps"]' in smoke


def test_v4_paths_are_isolated_from_v3_outputs() -> None:
    config = (ROOT / "configs" / "baseline_v1.yaml").read_text(encoding="utf-8")
    assert "project_root: ${AIC_PROJECT_ROOT}/v4_base" in config
    assert (
        "output_dir: ${AIC_OUTPUT_ROOT}/"
        "outputs/dinov3_vitl16_m2f_barren_clean_hard_v4_b4a2"
    ) in config


def test_shell_scripts_are_exported_with_linux_line_endings() -> None:
    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert "*.sh text eol=lf" in attributes.splitlines()
