from pathlib import Path


def test_reproduction_references_existing_entry_points():
    text = Path("docs/reproduction.md").read_text(encoding="utf-8")
    for path in (
        "data_prep/prepare_trainadd.py",
        "v4_base/scripts/launch_formal.sh",
        "v5_finetune/scripts/launch_formal.sh",
        "v5_finetune/scripts/launch_resume15.sh",
        "v5_finetune/scripts/validate_prediction_format.py",
    ):
        assert path in text
        assert Path(path).is_file()
