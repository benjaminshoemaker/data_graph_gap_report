#!/usr/bin/env python3
"""Utility for synchronising project version metadata."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT_PATH = PROJECT_ROOT / "pyproject.toml"
PACKAGE_INIT_PATH = PROJECT_ROOT / "src" / "data_needs_reporter" / "__init__.py"

VERSION_PATTERN = re.compile(
    r"^\d+\.\d+\.\d+(?:-[0-9A-Za-z][0-9A-Za-z\.-]*)?(?:\+[0-9A-Za-z][0-9A-Za-z\.-]*)?$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bump project version.")
    parser.add_argument("version", help="Target semantic version (e.g. 0.2.0).")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned changes without modifying files.",
    )
    return parser.parse_args()


def validate_version(version: str) -> None:
    if not VERSION_PATTERN.fullmatch(version):
        raise ValueError(
            f"Invalid version '{version}'. Expected semantic version like 1.2.3."
        )


def extract_current_version() -> str:
    content = PYPROJECT_PATH.read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', content, re.MULTILINE)
    if not match:
        raise RuntimeError("Unable to find version in pyproject.toml.")
    return match.group(1)


def update_pyproject(version: str, dry_run: bool) -> None:
    content = PYPROJECT_PATH.read_text(encoding="utf-8")
    new_content, count = re.subn(
        r'^(version\s*=\s*)"[^"]+"',
        rf'\1"{version}"',
        content,
        count=1,
        flags=re.MULTILINE,
    )
    if count != 1:
        raise RuntimeError("Failed to update version in pyproject.toml.")
    if dry_run:
        return
    PYPROJECT_PATH.write_text(new_content, encoding="utf-8")


def update_package_init(version: str, dry_run: bool) -> None:
    content = PACKAGE_INIT_PATH.read_text(encoding="utf-8")
    new_content, count = re.subn(
        r'^(__version__\s*=\s*)"[^"]+"',
        rf'\1"{version}"',
        content,
        count=1,
        flags=re.MULTILINE,
    )
    if count != 1:
        raise RuntimeError("Failed to update __version__ in package __init__.py.")
    if dry_run:
        return
    PACKAGE_INIT_PATH.write_text(new_content, encoding="utf-8")


def main() -> int:
    args = parse_args()
    target_version = args.version.strip()
    validate_version(target_version)

    current_version = extract_current_version()
    if current_version == target_version:
        print(f"Version already at {target_version}; nothing to do.")
        return 0

    print(f"Bumping version: {current_version} -> {target_version}")
    if args.dry_run:
        print("Dry run enabled; files will not be modified.")

    update_pyproject(target_version, args.dry_run)
    update_package_init(target_version, args.dry_run)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # pragma: no cover - CLI guard
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
