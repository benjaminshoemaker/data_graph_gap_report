from __future__ import annotations

import itertools
import subprocess
import sys
from typing import Iterable, List

BATCH_SIZE = 50


def _batched(items: Iterable[str], size: int = BATCH_SIZE) -> Iterable[List[str]]:
    iterator = iter(items)
    while True:
        batch = list(itertools.islice(iterator, size))
        if not batch:
            return
        yield batch


def _split_command(argv: list[str]) -> tuple[list[str], list[str]]:
    if "--" in argv:
        idx = argv.index("--")
        return argv[:idx], argv[idx + 1 :]
    return argv, []


def _collect_stdin_paths() -> list[str]:
    if sys.stdin is None or sys.stdin.isatty():
        return []
    return [line.strip() for line in sys.stdin if line.strip()]


def main() -> int:
    cmd, initial_files = _split_command(sys.argv[1:])
    if not cmd:
        print(
            "run_hook_paths.py requires at least one command argument.", file=sys.stderr
        )
        return 1

    stdin_paths = _collect_stdin_paths()
    files = stdin_paths or initial_files

    if not files:
        proc = subprocess.run(cmd, check=False)
        return proc.returncode

    exit_code = 0
    for batch in _batched(files):
        proc = subprocess.run(cmd + batch, check=False)
        if proc.returncode != 0:
            exit_code = proc.returncode
            break
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
