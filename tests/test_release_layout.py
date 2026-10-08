from pathlib import Path


def test_ignore_rules_cover_release_artifacts():
    ignored = Path('.gitignore').read_text(encoding='utf-8')
    for item in ('*.pth', 'ssh*.txt', '.env', 'outputs/', 'logs/', '__pycache__/'):
        assert item in ignored


def test_release_contains_complete_best_checkpoint_chain():
    required = (
        "v4_base/aic_dino_m2f/train.py",
        "v4_base/scripts/launch_formal.sh",
        "v5_finetune/aic_dino_m2f/train.py",
        "v5_finetune/scripts/prepare_loveda_mix.py",
        "v5_finetune/scripts/prepare_resume15.py",
        "v5_finetune/scripts/launch_resume15.sh",
        "data_prep/prepare_trainadd.py",
        "data_prep/trainadd_data.py",
        "data_prep/verify_trainadd.py",
    )
    for name in required:
        assert Path(name).is_file(), name
