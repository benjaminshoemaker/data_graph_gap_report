#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

try:  # pragma: no cover - unavailable on Windows
    import resource
except ImportError:  # pragma: no cover
    resource = None  # type: ignore[assignment]


DEFAULT_CONFIG = Path("configs/default.yaml")
DEFAULT_WAREHOUSE_SEED = 11
DEFAULT_COMMS_SEED = 17
COMMAND_TARGETS: dict[str, float] = {
    "gen-warehouse": 3.0,
    "gen-comms": 3.5,
    "run-report": 2.5,
    "validate": 2.0,
}


@dataclass(frozen=True)
class BenchCommand:
    name: str
    args: List[str]
    target_s: float


@dataclass
class BenchResult:
    name: str
    command: List[str]
    elapsed_s: float
    peak_rss_mb: float | None
    target_s: float
    exit_code: int

    @property
    def within_budget(self) -> bool:
        return self.exit_code == 0 and self.elapsed_s <= self.target_s


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run dnr performance benchmarks and capture timing + RSS."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Base config to clone for the benchmark run.",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        help="Directory to store generated artifacts (defaults to a temp dir).",
    )
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="Keep temporary artifacts even when no workspace is provided.",
    )
    parser.add_argument(
        "--warehouse-seed",
        type=int,
        default=DEFAULT_WAREHOUSE_SEED,
        help="Seed override for warehouse generation.",
    )
    parser.add_argument(
        "--comms-seed",
        type=int,
        default=DEFAULT_COMMS_SEED,
        help="Seed override for communications generation.",
    )
    parser.add_argument(
        "--report-format",
        choices=("table", "json"),
        default="table",
        help="Output format for the benchmark summary.",
    )
    parser.add_argument(
        "--report-file",
        type=Path,
        help="Optional path to write the benchmark summary when using JSON output.",
    )
    parser.add_argument(
        "--fail-on-budget",
        action="store_true",
        help="Exit with status 1 when any command exceeds its target budget.",
    )
    parser.add_argument(
        "--skip-validate",
        action="store_true",
        help="Skip the validate command (primarily for debugging).",
    )
    return parser.parse_args(argv)


def _load_config(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Config at {path} must be valid JSON; unable to parse."
        ) from exc


def _prepare_config(
    base_config: Path,
    dest_config: Path,
    *,
    warehouse_seed: int,
    comms_seed: int,
) -> Path:
    payload = _load_config(base_config)
    warehouse = payload.setdefault("warehouse", {})
    warehouse["scale"] = "quickstart"
    warehouse["months"] = min(int(warehouse.get("months", 18)), 6)
    warehouse["seed"] = int(warehouse_seed)

    comms = payload.setdefault("comms", {})
    comms["seed"] = int(comms_seed)
    comms["slack_threads"] = min(int(comms.get("slack_threads", 3000)), 400)
    comms["email_threads"] = min(int(comms.get("email_threads", 800)), 120)
    comms["nlq"] = min(int(comms.get("nlq", 1000)), 200)

    dest_config.parent.mkdir(parents=True, exist_ok=True)
    dest_config.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return dest_config


def _rss_to_mb(raw: int | None) -> float | None:
    if raw is None or raw <= 0:
        return None
    if sys.platform == "darwin":
        bytes_used = raw
    else:
        bytes_used = raw * 1024
    return bytes_used / (1024 * 1024)


def _run_with_metrics(
    command: Sequence[str], cwd: Path
) -> tuple[int, float, float | None]:
    env = os.environ.copy()
    start = time.perf_counter()
    peak_rss: float | None = None

    if resource is not None and hasattr(os, "fork"):
        exit_code, rss_raw = _fork_and_exec(command, cwd, env)
        peak_rss = _rss_to_mb(rss_raw)
    else:  # pragma: no cover - Windows fallback
        completed = subprocess.run(command, cwd=cwd, env=env, check=False)
        exit_code = completed.returncode

    elapsed = time.perf_counter() - start
    return exit_code, elapsed, peak_rss


def _fork_and_exec(
    command: Sequence[str],
    cwd: Path,
    env: dict[str, str],
) -> tuple[int, int | None]:
    pid = os.fork()
    if pid == 0:  # pragma: no cover - child path
        try:
            os.chdir(cwd)
            os.execvpe(command[0], list(command), env)
        except Exception as exc:  # pragma: no cover - logged in parent
            print(f"[bench] failed to exec {command}: {exc}", file=sys.stderr)
        os._exit(1)

    while True:
        try:
            _, status, usage = os.wait4(pid, 0)
            exit_code = os.waitstatus_to_exitcode(status)
            return exit_code, usage.ru_maxrss if usage else None
        except InterruptedError:
            continue


def _build_commands(
    base_flags: Sequence[str],
    *,
    warehouse_dir: Path,
    comms_dir: Path,
    reports_dir: Path,
    skip_validate: bool,
) -> List[BenchCommand]:
    commands: List[BenchCommand] = [
        BenchCommand(
            "gen-warehouse",
            [
                *base_flags,
                "gen-warehouse",
                "--archetype",
                "neobank",
                "--out",
                str(warehouse_dir),
            ],
            COMMAND_TARGETS["gen-warehouse"],
        ),
        BenchCommand(
            "gen-comms",
            [
                *base_flags,
                "gen-comms",
                "--archetype",
                "neobank",
                "--out",
                str(comms_dir),
            ],
            COMMAND_TARGETS["gen-comms"],
        ),
        BenchCommand(
            "run-report",
            [
                *base_flags,
                "run-report",
                "--warehouse",
                str(warehouse_dir),
                "--comms",
                str(comms_dir),
                "--out",
                str(reports_dir),
            ],
            COMMAND_TARGETS["run-report"],
        ),
    ]
    if not skip_validate:
        qc_dir = reports_dir / "qc"
        commands.append(
            BenchCommand(
                "validate",
                [
                    *base_flags,
                    "validate",
                    "--warehouse",
                    str(warehouse_dir),
                    "--comms",
                    str(comms_dir),
                    "--out",
                    str(qc_dir),
                    "--strict",
                ],
                COMMAND_TARGETS["validate"],
            )
        )
    return commands


def _print_table(results: Iterable[BenchResult]) -> None:
    header = f"{'Command':<18}{'Seconds':>10}{'Peak RSS (MB)':>16}{'Target (s)':>12}{'Status':>10}"
    print(header)
    print("-" * len(header))
    for result in results:
        rss = f"{result.peak_rss_mb:.1f}" if result.peak_rss_mb is not None else "—"
        status = (
            "ok"
            if result.exit_code == 0 and result.within_budget
            else ("slow" if result.exit_code == 0 else "fail")
        )
        print(
            f"{result.name:<18}"
            f"{result.elapsed_s:>10.2f}"
            f"{rss:>16}"
            f"{result.target_s:>12.2f}"
            f"{status:>10}"
        )


def _summarize_results(results: List[BenchResult]) -> Dict[str, Any]:
    return {
        "commands": [
            {
                "name": item.name,
                "elapsed_s": round(item.elapsed_s, 4),
                "peak_rss_mb": (
                    round(item.peak_rss_mb, 2) if item.peak_rss_mb is not None else None
                ),
                "target_s": item.target_s,
                "exit_code": item.exit_code,
                "within_budget": item.within_budget,
            }
            for item in results
        ],
        "all_within_budget": all(item.within_budget for item in results),
        "completed_commands": len(results),
        "max_peak_rss_mb": (
            max((item.peak_rss_mb or 0.0) for item in results) if results else 0.0
        ),
        "total_elapsed_s": round(sum(item.elapsed_s for item in results), 4),
        "platform": platform.platform(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    workspace_ctx = None
    workspace: Path

    if args.workspace:
        workspace = args.workspace
        workspace.mkdir(parents=True, exist_ok=True)
    elif args.keep_artifacts:
        workspace = Path(tempfile.mkdtemp(prefix="dnr-bench-"))
    else:
        workspace_ctx = tempfile.TemporaryDirectory(prefix="dnr-bench-")
        workspace = Path(workspace_ctx.name)

    config_path = _prepare_config(
        args.config,
        workspace / "bench_config.json",
        warehouse_seed=args.warehouse_seed,
        comms_seed=args.comms_seed,
    )

    base_flags = ["dnr", "--config", str(config_path), "--no-llm"]
    warehouse_dir = workspace / "warehouse"
    comms_dir = workspace / "comms"
    reports_dir = workspace / "reports"
    commands = _build_commands(
        base_flags,
        warehouse_dir=warehouse_dir,
        comms_dir=comms_dir,
        reports_dir=reports_dir,
        skip_validate=args.skip_validate,
    )

    results: List[BenchResult] = []
    exit_code = 0
    for command in commands:
        rc, elapsed, rss_mb = _run_with_metrics(command.args, workspace)
        result = BenchResult(
            name=command.name,
            command=list(command.args),
            elapsed_s=elapsed,
            peak_rss_mb=rss_mb,
            target_s=command.target_s,
            exit_code=rc,
        )
        results.append(result)
        if rc != 0:
            exit_code = rc
            break

    summary = _summarize_results(results)
    if args.report_format == "json":
        summary_text = json.dumps(summary, indent=2)
        if args.report_file:
            args.report_file.parent.mkdir(parents=True, exist_ok=True)
            args.report_file.write_text(summary_text + "\n", encoding="utf-8")
        print(summary_text)
    else:
        _print_table(results)

    if exit_code == 0 and args.fail_on_budget and not summary["all_within_budget"]:
        exit_code = 1

    if workspace_ctx is not None:
        workspace_ctx.cleanup()

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
