from pathlib import Path


def test_configs_contain_no_server_or_personal_root():
    for config in Path('.').glob('v*_*/configs/*.yaml'):
        text = config.read_text(encoding='utf-8')
        assert "/".join(("", "root", "autodl-tmp", "")) not in text
        assert 'D:\\aic\\' not in text
