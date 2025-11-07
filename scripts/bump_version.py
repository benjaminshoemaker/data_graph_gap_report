#!/usr/bin/env python3
"""Utility to bump the package version across tracked files."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INIT_FILE = ROOT / "src" / "data_needs_reporter" / "__init__.py"
PYPROJECT_FILE = ROOT / "pyproject.toml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bump the project version")
    parser.add_argument(
        "--level",
        choices=("patch", "minor", "major"),
        default="patch",
        help="Version component to bump (default: patch)",
    )
    return parser.parse_args()


def extract_version() -> str:
    text = INIT_FILE.read_text(encoding="utf-8")
    match = re.search(r'__version__\s*=\s*"([0-9]+\.[0-9]+\.[0-9]+)"', text)
    if not match:
        raise RuntimeError(f"Unable to find __version__ in {INIT_FILE}")
    return match.group(1)


def bump_value(version: str, level: str) -> str:
    parts = version.split(".")
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        raise ValueError(f"Unsupported version format: {version}")
    major, minor, patch = map(int, parts)
    if level == "major":
        major += 1
        minor = 0
        patch = 0
    elif level == "minor":
        minor += 1
        patch = 0
    else:
        patch += 1
    return f"{major}.{minor}.{patch}"


def update_init(new_version: str) -> None:
    text = INIT_FILE.read_text(encoding="utf-8")
    pattern = r'(__version__\s*=\s*")([0-9]+\.[0-9]+\.[0-9]+)(")'
    new_text = re.sub(
        pattern,
        lambda match: f"{match.group(1)}{new_version}{match.group(3)}",
        text,
        count=1,
    )
    if text == new_text:
        raise RuntimeError("Failed to update __init__ version")
    INIT_FILE.write_text(new_text, encoding="utf-8")


def _replace_section_version(content: str, section: str, new_version: str) -> str:
    lines = content.splitlines()
    inside = False
    target_header = f"[{section}]"
    replaced = False
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            inside = stripped == target_header
        elif inside and stripped.startswith("version"):
            pattern = r'(version\s*=\s*")([0-9]+\.[0-9]+\.[0-9]+)(")'
            lines[idx] = re.sub(
                pattern,
                lambda match: f"{match.group(1)}{new_version}{match.group(3)}",
                line,
                count=1,
            )
            replaced = True
            break
    if not replaced:
        raise RuntimeError(f"Unable to locate version field in [{section}] section")
    return "\n".join(lines)


def update_pyproject(new_version: str) -> None:
    content = PYPROJECT_FILE.read_text(encoding="utf-8")
    content = _replace_section_version(content, "project", new_version)
    content = _replace_section_version(content, "tool.poetry", new_version)
    PYPROJECT_FILE.write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    current = extract_version()
    new_version = bump_value(current, args.level)
    update_init(new_version)
    update_pyproject(new_version)
    print(new_version)


if __name__ == "__main__":
    main()
