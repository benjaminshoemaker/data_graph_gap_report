from __future__ import annotations

import json
from pathlib import Path

import polars as pl
from typer.testing import CliRunner

from data_needs_reporter.cli import app

runner = CliRunner()


def _load(path: Path) -> pl.DataFrame:
    assert path.exists(), f"Missing labels file: {path}"
    df = pl.read_parquet(path)
    assert df.height > 0, f"No rows in {path}"
    return df


def test_slack_labels_schema() -> None:
    df = _load(Path("meta") / "labels_slack.parquet")
    expected = {"thread_id", "theme", "relevance"}
    assert expected.issubset(df.columns)
    assert df["thread_id"].dtype == pl.Int64
    assert df["theme"].dtype == pl.Utf8
    assert df["relevance"].dtype in {pl.Float32, pl.Float64}
    assert df.height >= 24


def test_email_labels_schema() -> None:
    df = _load(Path("meta") / "labels_email.parquet")
    expected = {"thread_id", "theme", "relevance"}
    assert expected.issubset(df.columns)
    assert df["thread_id"].dtype == pl.Int64
    assert df["theme"].dtype == pl.Utf8
    assert df["relevance"].dtype in {pl.Float32, pl.Float64}
    assert df.height >= 24


def test_nlq_labels_schema() -> None:
    df = _load(Path("meta") / "labels_nlq.parquet")
    expected = {"query_id", "theme"}
    assert expected.issubset(df.columns)
    assert df["query_id"].dtype == pl.Int64
    assert df["theme"].dtype == pl.Utf8
    assert df.height >= 24


def _cycle_theme(theme: str) -> str:
    order = ["data_quality", "pipeline_health", "governance"]
    if theme not in order:
        return theme
    idx = order.index(theme)
    return order[(idx + 1) % len(order)]


def _apply_noise(frame: pl.DataFrame, flip_fraction: float) -> pl.DataFrame:
    if flip_fraction <= 0:
        return frame
    total = frame.height
    flip_count = int(total * flip_fraction)
    if flip_count <= 0:
        return frame
    return frame.with_columns(
        pl.when(pl.arange(0, total) < flip_count)
        .then(pl.col("theme").map_elements(_cycle_theme, return_dtype=pl.Utf8))
        .otherwise(pl.col("theme"))
        .alias("theme")
    )


def _limit_rows(frame: pl.DataFrame, drop_fraction: float) -> pl.DataFrame:
    if drop_fraction <= 0:
        return frame
    total = frame.height
    drop_count = int(total * drop_fraction)
    keep = max(1, total - drop_count)
    return frame.head(keep)


def _build_predictions(
    base_dir: Path,
    *,
    drop_fraction: float = 0.0,
    flip_fraction: float = 0.0,
) -> Path:
    preds_dir = base_dir / "preds"
    preds_dir.mkdir(parents=True, exist_ok=True)

    slack = pl.read_parquet(Path("meta") / "labels_slack.parquet")
    email = pl.read_parquet(Path("meta") / "labels_email.parquet")
    nlq = pl.read_parquet(Path("meta") / "labels_nlq.parquet")

    slack_base = slack.select(
        pl.lit("slack").alias("source"),
        pl.col("thread_id").cast(pl.Int64).alias("thread_id"),
        pl.lit(-1).cast(pl.Int64).alias("query_id"),
        pl.col("theme"),
        pl.col("relevance"),
        pl.col("relevance").clip(0.0, 1.0).alias("confidence"),
    )
    email_base = email.select(
        pl.lit("email").alias("source"),
        pl.col("thread_id").cast(pl.Int64).alias("thread_id"),
        pl.lit(-1).cast(pl.Int64).alias("query_id"),
        pl.col("theme"),
        pl.col("relevance"),
        pl.col("relevance").clip(0.0, 1.0).alias("confidence"),
    )
    nlq_base = nlq.select(
        pl.lit("nlq").alias("source"),
        pl.lit(-1).cast(pl.Int64).alias("thread_id"),
        pl.col("query_id").cast(pl.Int64).alias("query_id"),
        pl.col("theme"),
        pl.lit(None, dtype=pl.Float64).alias("relevance"),
        pl.lit(0.85).alias("confidence"),
    )

    slack_pred = _limit_rows(_apply_noise(slack_base, flip_fraction), drop_fraction)
    email_pred = _limit_rows(_apply_noise(email_base, flip_fraction), drop_fraction)
    nlq_pred = _limit_rows(_apply_noise(nlq_base, flip_fraction), drop_fraction)

    predictions = pl.concat([slack_pred, email_pred, nlq_pred], how="vertical")
    predictions.write_parquet(preds_dir / "predictions.parquet")
    return preds_dir


def test_eval_labels_cli_pass(tmp_path: Path) -> None:
    preds_dir = _build_predictions(tmp_path)
    out_base = tmp_path / "reports" / "eval"
    result = runner.invoke(
        app,
        [
            "eval-labels",
            "--pred",
            str(preds_dir),
            "--labels",
            str(Path("meta")),
            "--out",
            str(out_base),
        ],
    )
    assert result.exit_code == 0, result.stdout
    run_dirs = list(out_base.iterdir())
    assert len(run_dirs) == 1
    out_dir = run_dirs[0]
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["gates_pass"] is True
    assert (out_dir / "per_class.csv").exists()
    assert (out_dir / "confusion_slack_theme.json").exists()


def test_eval_labels_cli_gate_fail(tmp_path: Path) -> None:
    preds_dir = _build_predictions(tmp_path, drop_fraction=0.2, flip_fraction=0.2)
    out_base = tmp_path / "reports" / "eval_fail"
    result = runner.invoke(
        app,
        [
            "eval-labels",
            "--pred",
            str(preds_dir),
            "--labels",
            str(Path("meta")),
            "--out",
            str(out_base),
        ],
    )
    assert result.exit_code == 1
    run_dirs = list(out_base.iterdir())
    assert len(run_dirs) == 1
    out_dir = run_dirs[0]
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["gates_pass"] is False
