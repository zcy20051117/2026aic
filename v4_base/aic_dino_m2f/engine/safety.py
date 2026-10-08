from __future__ import annotations

from pathlib import Path

from .manifest import assert_project_path


def prepare_run_directory(project_root: Path, output_dir: Path, *, resume: bool) -> Path:
    output = assert_project_path(output_dir, project_root)
    if resume:
        if not output.is_dir():
            raise FileNotFoundError(f"Resume output directory does not exist: {output}")
        return output
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Run output already exists; use explicit --resume: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def require_fold0_list(requested: Path, configured_fold0: Path) -> Path:
    requested_resolved = Path(requested).resolve(strict=True)
    fold0_resolved = Path(configured_fold0).resolve(strict=True)
    if requested_resolved != fold0_resolved:
        raise ValueError(f"Evaluation is restricted to the configured Fold0 list: {fold0_resolved}")
    return requested_resolved
