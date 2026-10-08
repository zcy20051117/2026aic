from __future__ import annotations

import argparse
from pathlib import Path

from aic_dino_m2f.config import load_and_validate_config
from aic_dino_m2f.engine.continuation import (
    prepare_continuation_output,
    validate_continuation_configs,
)
from aic_dino_m2f.engine.manifest import assert_project_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare an isolated, exact-state V5 continuation")
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--target-config", type=Path, required=True)
    parser.add_argument("--expected-last-epoch", type=int, required=True)
    args = parser.parse_args()
    from aic_dino_m2f.runtime import checkpoint_metadata

    source = load_and_validate_config(args.source_config)
    target = load_and_validate_config(args.target_config)
    validate_continuation_configs(source, target)
    project_root = Path(source["project_root"])
    source_output = assert_project_path(Path(source["training"]["output_dir"]), project_root)
    target_output = assert_project_path(Path(target["training"]["output_dir"]), project_root)
    prepare_continuation_output(
        source_output, target_output,
        checkpoint_metadata(source), checkpoint_metadata(target),
        args.expected_last_epoch, int(target["training"]["patience"]),
    )
    print(f"Prepared continuation: {target_output}", flush=True)


if __name__ == "__main__":
    main()
