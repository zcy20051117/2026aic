from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml


def expand(value: object) -> object:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {key: expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand(item) for item in value]
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description='Check that a release config has all required local inputs.')
    parser.add_argument('config', type=Path)
    args = parser.parse_args()
    config = expand(yaml.safe_load(args.config.read_text(encoding='utf-8')))
    required = [
        config['data']['images_dir'], config['data']['masks_dir'],
        config['data']['train_list'], config['data']['val_list'],
        config['model']['backbone_weights'],
    ]
    missing = [str(path) for path in required if not Path(path).exists()]
    if missing:
        print('Missing required inputs:')
        print('\n'.join(missing))
        return 1
    print('Environment check passed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

