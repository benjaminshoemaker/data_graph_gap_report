from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Mapping

from data_needs_reporter.utils.io import write_json_atomic


def compute_file_hash(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def write_hash_manifest(
    base_path: Path,
    files: Mapping[str, Path],
    seeds: Mapping[str, object],
    *,
    algorithm: str = "sha256",
    manifest_name: str = "hashes.json",
) -> None:
    normalized: Dict[str, str] = {}
    for name, file_path in files.items():
        normalized[name] = compute_file_hash(file_path)
    manifest = {
        "hash_algorithm": algorithm,
        "files": normalized,
        "seeds": dict(seeds),
    }
    write_json_atomic(base_path / manifest_name, manifest)


__all__ = ["compute_file_hash", "write_hash_manifest"]
