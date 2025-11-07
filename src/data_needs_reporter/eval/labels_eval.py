from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Callable, Dict, Mapping, Sequence

import polars as pl

THEME_SOURCES: Mapping[str, Dict[str, str]] = {
    "slack": {"id": "thread_id", "labels": "labels_slack.parquet"},
    "email": {"id": "thread_id", "labels": "labels_email.parquet"},
    "nlq": {"id": "query_id", "labels": "labels_nlq.parquet"},
}

DEFAULT_THEME_GATE = 0.72
RELEVANCE_GATE = 0.75
COVERAGE_GATE = 0.95
BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SEED = 42
ECE_BINS = 10


__all__ = [
    "DEFAULT_THEME_GATE",
    "RELEVANCE_GATE",
    "COVERAGE_GATE",
    "THEME_SOURCES",
    "load_predictions",
    "collect_eval_rows",
    "compute_confusion",
    "per_class_metrics",
    "macro_f1",
    "compute_accuracy",
    "expected_calibration_error",
    "bootstrap_ci",
    "summarize_pairs",
    "evaluate_labels",
]


def _load_parquet(path: Path) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing parquet file: {path}")
    df = pl.scan_parquet(str(path)).collect(streaming=True)
    if df.height == 0:
        raise ValueError(f"Parquet file {path} is empty")
    return df


def load_predictions(pred_dir: Path) -> pl.DataFrame:
    """Load prediction parquet files and normalize the source column."""

    pred_dir = Path(pred_dir)
    lazy_frames: list[pl.LazyFrame] = []
    if pred_dir.is_file():
        lazy_frames.append(pl.scan_parquet(str(pred_dir)))
    else:
        direct = pred_dir / "predictions.parquet"
        if direct.exists():
            lazy_frames.append(pl.scan_parquet(str(direct)))
        else:
            parquet_files = sorted(pred_dir.glob("*.parquet"))
            if not parquet_files:
                raise FileNotFoundError(
                    f"No prediction parquet files found under {pred_dir}"
                )
            for file_path in parquet_files:
                frame = pl.scan_parquet(str(file_path))
                if "source" not in frame.columns:
                    inferred = _infer_source_from_name(file_path.name)
                    if inferred is None:
                        raise ValueError(
                            f"Unable to infer source for {file_path}; add a 'source' column."
                        )
                    frame = frame.with_columns(pl.lit(inferred).alias("source"))
                lazy_frames.append(frame)

    if not lazy_frames:
        raise ValueError(f"No prediction frames read from {pred_dir}")

    predictions = pl.concat(lazy_frames, how="vertical_relaxed").collect(
        streaming=True
    )
    if "source" not in predictions.columns:
        raise ValueError("Predictions must include a 'source' column")
    return predictions.with_columns(pl.col("source").str.to_lowercase())


def _infer_source_from_name(file_name: str) -> str | None:
    lower_name = file_name.lower()
    for source_name in THEME_SOURCES:
        if source_name in lower_name:
            return source_name
    return None


def _prepare_joined(
    predictions: pl.DataFrame, labels: pl.DataFrame, id_col: str
) -> pl.DataFrame:
    if id_col not in predictions.columns:
        raise ValueError(f"Predictions missing id column '{id_col}'")
    if id_col not in labels.columns:
        raise ValueError(f"Labels missing id column '{id_col}'")

    pred_cols = [id_col]
    rename_map: dict[str, str] = {}
    for column in ("theme", "relevance", "confidence"):
        if column in predictions.columns:
            pred_cols.append(column)
            rename_map[column] = f"{column}_pred"
    pred_subset = predictions.select([pl.col(col) for col in pred_cols]).rename(
        rename_map
    )

    label_cols = [id_col]
    label_rename: dict[str, str] = {}
    for column in ("theme", "relevance"):
        if column in labels.columns:
            label_cols.append(column)
            label_rename[column] = f"{column}_true"
    label_subset = labels.select(label_cols).rename(label_rename)

    return pred_subset.join(label_subset, on=id_col, how="inner")


def collect_eval_rows(
    pred_df: pl.DataFrame,
    labels_dir: Path,
    *,
    source: str | None = None,
    task: str = "theme",
    threshold: float = 0.5,
) -> tuple[list[str], list[str], list[float], float, int]:
    """Join predictions with labels and return aligned pairs for a task."""

    labels_dir = Path(labels_dir)
    if source is None:
        if pred_df.height == 0:
            raise ValueError("Source must be provided when prediction frame is empty")
        if "source" not in pred_df.columns:
            raise ValueError("Predictions missing 'source' column")
        unique_sources = pred_df.get_column("source").unique().to_list()
        if len(unique_sources) != 1:
            raise ValueError("pred_df must contain exactly one source when source is omitted")
        source = str(unique_sources[0])
    source = source.lower()

    if source not in THEME_SOURCES:
        raise ValueError(f"Unsupported source '{source}'")

    info = THEME_SOURCES[source]
    labels = _load_parquet(labels_dir / info["labels"])

    if task == "theme" and "theme" not in pred_df.columns:
        raise ValueError("Predictions missing 'theme' column for theme evaluation")
    if task == "relevance" and "relevance" not in pred_df.columns:
        raise ValueError("Predictions missing 'relevance' column for relevance evaluation")

    joined = _prepare_joined(pred_df, labels, info["id"])
    coverage = joined.height / labels.height if labels.height else 0.0

    if task == "theme":
        true_labels = (
            joined["theme_true"].to_list() if "theme_true" in joined.columns else []
        )
        pred_labels = (
            joined["theme_pred"].to_list() if "theme_pred" in joined.columns else []
        )
        confidences = (
            [
                max(0.0, min(1.0, float(val)))
                for val in joined["confidence_pred"].to_list()
            ]
            if "confidence_pred" in joined.columns
            else []
        )
    elif task == "relevance":
        if "relevance_true" not in joined.columns or "relevance_pred" not in joined.columns:
            raise ValueError(f"Relevance columns missing for source '{source}'")
        true_labels = [
            "positive" if float(val) >= threshold else "negative"
            for val in joined["relevance_true"].to_list()
        ]
        pred_scores = [
            max(0.0, min(1.0, float(val) if val is not None else 0.0))
            for val in joined["relevance_pred"].to_list()
        ]
        pred_labels = ["positive" if score >= threshold else "negative" for score in pred_scores]
        confidences = pred_scores
    else:
        raise ValueError(f"Unsupported task '{task}'")

    return true_labels, pred_labels, confidences, coverage, labels.height


def compute_confusion(
    predictions: Sequence[str],
    truths: Sequence[str],
    classes: Sequence[str] | None = None,
) -> dict[str, dict[str, int]]:
    label_set = list(classes) if classes is not None else sorted(
        {str(val) for val in truths} | {str(val) for val in predictions}
    )
    confusion: dict[str, dict[str, int]] = {
        cls: {other: 0 for other in label_set} for cls in label_set
    }
    for truth, pred in zip(truths, predictions):
        if truth not in confusion:
            confusion[truth] = {other: 0 for other in label_set}
        if pred not in confusion[truth]:
            for row in confusion.values():
                if pred not in row:
                    row[pred] = 0
        confusion[truth][pred] += 1
    return confusion


def per_class_metrics(
    confusion: Mapping[str, Mapping[str, int]],
    classes: Sequence[str],
) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for cls in classes:
        tp = confusion.get(cls, {}).get(cls, 0)
        fp = sum(
            confusion.get(other, {}).get(cls, 0) for other in classes if other != cls
        )
        fn = sum(
            confusion.get(cls, {}).get(other, 0) for other in classes if other != cls
        )
        support = tp + fn
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / support if support else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        metrics[cls] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    return metrics


def macro_f1(per_class: Mapping[str, Mapping[str, float]]) -> float:
    scores = [vals["f1"] for vals in per_class.values() if vals.get("support", 0) > 0]
    return sum(scores) / len(scores) if scores else 0.0


def compute_accuracy(confusion: Mapping[str, Mapping[str, int]]) -> float:
    total = 0
    correct = 0
    for cls, preds in confusion.items():
        for pred, count in preds.items():
            total += count
            if cls == pred:
                correct += count
    return correct / total if total else 0.0


def expected_calibration_error(
    confidences: Sequence[float],
    correct: Sequence[bool],
    bins: int = ECE_BINS,
) -> float:
    pairs = list(zip(confidences, correct))
    if not pairs or bins <= 0:
        return 0.0
    totals = [0] * bins
    sum_correct = [0] * bins
    sum_conf = [0.0] * bins
    for conf, is_correct in pairs:
        if not isinstance(conf, (int, float)) or math.isnan(conf):
            conf = 1.0
        conf = max(0.0, min(1.0, float(conf)))
        idx = min(bins - 1, int(conf * bins))
        totals[idx] += 1
        sum_correct[idx] += 1 if is_correct else 0
        sum_conf[idx] += conf
    total_count = len(pairs)
    ece = 0.0
    for total_bin, correct_bin, conf_sum in zip(totals, sum_correct, sum_conf):
        if total_bin == 0:
            continue
        acc = correct_bin / total_bin
        avg_conf = conf_sum / total_bin
        ece += abs(acc - avg_conf) * (total_bin / total_count)
    return ece


def _resample_metric_factory(
    true_labels: Sequence[str],
    pred_labels: Sequence[str],
    classes: Sequence[str],
    kind: str,
) -> Callable[[random.Random], float]:
    n = len(true_labels)
    if n == 0:
        return lambda *_: 0.0

    def _metric(rng: random.Random) -> float:
        indices = [rng.randrange(n) for _ in range(n)]
        sample_true = [true_labels[i] for i in indices]
        sample_pred = [pred_labels[i] for i in indices]
        confusion = compute_confusion(sample_pred, sample_true, classes)
        if kind == "macro_f1":
            return macro_f1(per_class_metrics(confusion, classes))
        if kind == "accuracy":
            return compute_accuracy(confusion)
        raise ValueError(f"Unsupported metric kind '{kind}'")

    return _metric


def bootstrap_ci(
    metric_fn: Callable[[random.Random], float],
    n: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    if n <= 0:
        raise ValueError("n must be positive for bootstrap_ci")
    rng = random.Random(seed)
    values = [metric_fn(rng) for _ in range(n)]
    values.sort()
    lower_idx = max(0, int(0.025 * n))
    upper_idx = min(n - 1, int(0.975 * n))
    return values[lower_idx], values[upper_idx]


def summarize_pairs(
    true_labels: Sequence[str],
    pred_labels: Sequence[str],
    confidences: Sequence[float] | None = None,
) -> dict[str, object]:
    confidences = list(confidences or [])
    classes = sorted({str(val) for val in true_labels} | {str(val) for val in pred_labels})
    if not true_labels:
        confusion = {cls: {other: 0 for other in classes} for cls in classes}
        per_class = {
            cls: {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}
            for cls in classes
        }
        ci = {"macro_f1": [0.0, 0.0], "accuracy": [0.0, 0.0]}
        return {
            "overall": {"accuracy": 0.0, "macro_f1": 0.0, "ece": 0.0},
            "per_class": per_class,
            "confusion": confusion,
            "ci": ci,
        }

    confusion = compute_confusion(pred_labels, true_labels, classes)
    per_class = per_class_metrics(confusion, classes)
    accuracy = compute_accuracy(confusion)
    macro = macro_f1(per_class)
    correct_flags = [truth == pred for truth, pred in zip(true_labels, pred_labels)]
    ece = expected_calibration_error(confidences, correct_flags) if confidences else 0.0

    macro_ci = bootstrap_ci(
        _resample_metric_factory(true_labels, pred_labels, classes, "macro_f1")
    )
    accuracy_ci = bootstrap_ci(
        _resample_metric_factory(true_labels, pred_labels, classes, "accuracy")
    )

    return {
        "overall": {"accuracy": accuracy, "macro_f1": macro, "ece": ece},
        "per_class": per_class,
        "confusion": confusion,
        "ci": {"macro_f1": list(macro_ci), "accuracy": list(accuracy_ci)},
    }


def evaluate_labels(
    pred_df: pl.DataFrame,
    labels_dir: Path,
    *,
    task: str = "theme",
    threshold: float = 0.5,
) -> dict[str, object]:
    """Evaluate predictions for a single source and task against oracle labels."""

    true_labels, pred_labels, confidences, coverage, _ = collect_eval_rows(
        pred_df, labels_dir, task=task, threshold=threshold
    )
    metrics = summarize_pairs(true_labels, pred_labels, confidences)
    metrics.setdefault("overall", {})["coverage"] = coverage
    return metrics
