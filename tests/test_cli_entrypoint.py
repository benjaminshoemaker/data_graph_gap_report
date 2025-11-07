from __future__ import annotations

import os
import subprocess
import sys
import venv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _scripts_dir(venv_dir: Path) -> Path:
    return venv_dir / ("Scripts" if os.name == "nt" else "bin")


def _host_site_packages() -> list[str]:
    paths: list[str] = []
    for entry in sys.path:
        if "site-packages" in entry and Path(entry).exists():
            if entry not in paths:
                paths.append(entry)
    return paths


def _install_editable(python_bin: Path, env: dict[str, str]) -> None:
    env = dict(env)
    env.setdefault("PIP_NO_BUILD_ISOLATION", "1")
    subprocess.run(
        [
            str(python_bin),
            "-m",
            "pip",
            "install",
            "--no-build-isolation",
            "--no-deps",
            "-e",
            str(PROJECT_ROOT),
        ],
        check=True,
        env=env,
    )


def test_cli_entrypoint(tmp_path: Path) -> None:
    venv_dir = tmp_path / "pkg_env"
    venv.EnvBuilder(with_pip=True, clear=True).create(venv_dir)
    scripts_dir = _scripts_dir(venv_dir)
    python_bin = scripts_dir / ("python.exe" if os.name == "nt" else "python")
    dnr_bin = scripts_dir / ("dnr.exe" if os.name == "nt" else "dnr")

    env = os.environ.copy()
    host_paths = _host_site_packages()
    if host_paths:
        existing = env.get("PYTHONPATH")
        combined = host_paths + ([existing] if existing else [])
        env["PYTHONPATH"] = os.pathsep.join([p for p in combined if p])

    _install_editable(python_bin, env)

    proc = subprocess.run(
        [str(dnr_bin), "--help"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr
    help_text = proc.stdout
    for command in ("init", "gen-warehouse", "eval-labels", "run-report"):
        assert command in help_text
