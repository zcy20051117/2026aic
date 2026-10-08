from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping, Sequence


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest for a regular file without loading it at once."""
    resolved = Path(path).resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"Expected a regular file: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_nonempty_lines(path: Path) -> int:
    resolved = Path(path).resolve(strict=True)
    with resolved.open("r", encoding="utf-8-sig") as handle:
        return sum(1 for line in handle if line.strip())


def assert_project_path(path: Path, project_root: Path) -> Path:
    """Resolve a prospective output and reject escape from the project root."""
    root = Path(project_root).resolve(strict=True)
    candidate = Path(path).resolve(strict=False)
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"Path is outside project root: {candidate}")
    return candidate


def build_input_manifest(
    *,
    model_identity: str,
    backbone_files: Sequence[Path],
    train_list: Path,
    val_list: Path,
    fold_assignment: Path,
    protected_roots: Mapping[str, Path],
) -> dict[str, object]:
    """Build immutable input metadata used by preflight and resume checks."""
    if not model_identity.strip():
        raise ValueError("model_identity must not be empty")
    if not backbone_files:
        raise ValueError("At least one backbone file is required")

    backbone_entries = []
    for source in backbone_files:
        resolved = Path(source).resolve(strict=True)
        backbone_entries.append(
            {
                "path": str(resolved),
                "size_bytes": resolved.stat().st_size,
                "sha256": sha256_file(resolved),
            }
        )

    protected = {
        name: str(Path(root).resolve(strict=False))
        for name, root in sorted(protected_roots.items())
    }
    return {
        "schema_version": 1,
        "model_identity": model_identity,
        "backbone_files": backbone_entries,
        "split": {
            "train_list": str(Path(train_list).resolve(strict=True)),
            "train_count": count_nonempty_lines(train_list),
            "train_sha256": sha256_file(train_list),
            "val_list": str(Path(val_list).resolve(strict=True)),
            "val_count": count_nonempty_lines(val_list),
            "val_sha256": sha256_file(val_list),
            "fold_assignment": str(Path(fold_assignment).resolve(strict=True)),
            "fold_assignment_sha256": sha256_file(fold_assignment),
        },
        "protected_roots": protected,
    }
