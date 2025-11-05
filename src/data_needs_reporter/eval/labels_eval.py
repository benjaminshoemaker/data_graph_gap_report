from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import polars as pl

THEME_SOURCES: Mapping[str, Dict[str, str]] = {
    "slack": {"id": "thread_id", "labels": "labels_slack.parquet"},
    "email": {"id": "thread_id", "labels": "labels_email.parquet"},
    "nlq": {"id": "query_id", "labels": "labels_nlq.parquet"},
}

DEFAULT_THEME_GATE = 0.72
RELEVANCE_GATE = 0.75
COVERAGE_GATE = 0.95
BOOTSTRAP_SAMPLES = 200
BOOTSTRAP_SEED = 42
ECE_BINS = 10


def _load_parquet(path: Path) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing parquet file: {path}")
    df = pl.read_parquet(path)
    if df.height == 0:
        raise ValueError(f"Parquet file {path} is empty")
    return df


def _load_predictions(preds_dir: Path) -> pl.DataFrame:
    preds_dir = Path(preds_dir)
    direct = preds_dir / "predictions.parquet"
    frames: List[pl.DataFrame] = []
    if direct.exists():
        frames.append(pl.read_parquet(direct))
    else:
        parquet_files = sorted(preds_dir.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No prediction parquet files under {preds_dir}")
        for file_path in parquet_files:
            df = pl.read_parquet(file_path)
            if "source" not in df.columns:
                inferred = None
                lower = file_path.name.lower()
                for source in THEME_SOURCES:
                    if source in lower:
                        inferred = source
                        break
                if inferred is None:
                    raise ValueError(
                        f"Prediction file {file_path} missing 'source' column and unable to infer source."
                    )
                df = df.with_columns(pl.lit(inferred).alias("source"))
            frames.append(df)
    predictions = pl.concat(frames, how="vertical_relaxed")
    if "source" not in predictions.columns:
        raise ValueError("Predictions must include a 'source' column.")
    predictions = predictions.with_columns(pl.col("source").str.to_lowercase())
    return predictions


def _load_labels(labels_dir: Path) -> Dict[str, pl.DataFrame]:
    labels_dir = Path(labels_dir)
    label_frames: Dict[str, pl.DataFrame] = {}
    for source, info in THEME_SOURCES.items():
        path = labels_dir / info["labels"]
        df = _load_parquet(path)
        label_frames[source] = df
    return label_frames


def _unique_classes(
    true_labels: Sequence[str], pred_labels: Sequence[str]
) -> List[str]:
    classes: List[str] = []
    for value in list(true_labels) + list(pred_labels):
        if value not in classes:
            classes.append(value)
    return classes


def _confusion_matrix(
    true_labels: Sequence[str], pred_labels: Sequence[str], classes: Sequence[str]
) -> Dict[str, Dict[str, int]]:
    confusion: Dict[str, Dict[str, int]] = {
        cls: {other: 0 for other in classes} for cls in classes
    }
    for true, pred in zip(true_labels, pred_labels):
        if true not in confusion:
            confusion[true] = {other: 0 for other in classes}
        if pred not in confusion[true]:
            for cls in confusion:
                if pred not in confusion[cls]:
                    confusion[cls][pred] = 0
            confusion[true][pred] = 0
        confusion[true][pred] += 1
    return confusion


def _per_class_metrics(
    confusion: Dict[str, Dict[str, int]], classes: Sequence[str]
) -> Dict[str, Dict[str, float]]:
    per_class: Dict[str, Dict[str, float]] = {}
    for cls in classes:
        tp = confusion.get(cls, {}).get(cls, 0)
        fp = sum(
            confusion.get(other, {}).get(cls, 0) for other in classes if other != cls
        )
        fn = sum(
            confusion.get(cls, {}).get(other, 0) for other in classes if other != cls
        )
        support = tp + fn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / support if support > 0 else 0.0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        per_class[cls] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    return per_class


def _accuracy(confusion: Dict[str, Dict[str, int]]) -> float:
    total = 0
    correct = 0
    for true, row in confusion.items():
        for pred, value in row.items():
            total += value
            if true == pred:
                correct += value
    return correct / total if total else 0.0


def _macro_f1(per_class: Mapping[str, Mapping[str, float]]) -> float:
    scores = [metrics["f1"] for metrics in per_class.values() if metrics["support"] > 0]
    return sum(scores) / len(scores) if scores else 0.0


def _expected_calibration_error(
    confidences: Sequence[float], correct: Sequence[bool], bins: int = ECE_BINS
) -> float:
    if not confidences:
        return 0.0
    total = len(confidences)
    bin_totals = [0] * bins
    bin_correct = [0] * bins
    bin_conf_sum = [0.0] * bins
    for conf, is_correct in zip(confidences, correct):
        if not isinstance(conf, (int, float)) or math.isnan(conf):
            conf = 1.0
        conf = max(0.0, min(1.0, float(conf)))
        idx = min(bins - 1, int(conf * bins))
        bin_totals[idx] += 1
        bin_correct[idx] += 1 if is_correct else 0
        bin_conf_sum[idx] += conf
    ece = 0.0
    for total_bin, correct_bin, conf_sum in zip(bin_totals, bin_correct, bin_conf_sum):
        if total_bin == 0:
            continue
        acc = correct_bin / total_bin
        avg_conf = conf_sum / total_bin
        ece += abs(acc - avg_conf) * (total_bin / total)
    return ece


def _bootstrap_macro_f1(
    true_labels: Sequence[str],
    pred_labels: Sequence[str],
    classes: Sequence[str],
    samples: int,
    seed: int,
) -> Dict[str, float]:
    n = len(true_labels)
    if n == 0:
        return {"lower": 0.0, "upper": 0.0}
    rng = random.Random(seed)
    scores: List[float] = []
    for _ in range(samples):
        indices = [rng.randrange(n) for _ in range(n)]
        sample_true = [true_labels[i] for i in indices]
        sample_pred = [pred_labels[i] for i in indices]
        confusion = _confusion_matrix(sample_true, sample_pred, classes)
        per_class = _per_class_metrics(confusion, classes)
        scores.append(_macro_f1(per_class))
    scores.sort()
    lower_idx = max(0, int(0.025 * len(scores)) - 1)
    upper_idx = min(len(scores) - 1, int(0.975 * len(scores)))
    return {"lower": scores[lower_idx], "upper": scores[upper_idx]}


def _classification_metrics(
    true_labels: Sequence[str],
    pred_labels: Sequence[str],
    confidences: Sequence[float],
    classes: Sequence[str],
    coverage: float,
) -> Dict[str, object]:
    confusion = _confusion_matrix(true_labels, pred_labels, classes)
    per_class = _per_class_metrics(confusion, classes)
    accuracy = _accuracy(confusion)
    macro_f1 = _macro_f1(per_class)
    correct_flags = [t == p for t, p in zip(true_labels, pred_labels)]
    ece = _expected_calibration_error(confidences, correct_flags)
    ci = _bootstrap_macro_f1(
        true_labels, pred_labels, classes, BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED
    )
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion": confusion,
        "coverage": coverage,
        "ece": ece,
        "macro_f1_ci": ci,
    }


def _prepare_joined(
    predictions: pl.DataFrame, labels: pl.DataFrame, id_col: str
) -> pl.DataFrame:
    if id_col not in predictions.columns:
        raise ValueError(f"Predictions missing required id column '{id_col}'")
    if id_col not in labels.columns:
        raise ValueError(f"Labels missing required id column '{id_col}'")

    pred_cols = [id_col, "theme"]
    rename_map = {"theme": "theme_pred"}
    if "relevance" in predictions.columns:
        pred_cols.append("relevance")
        rename_map["relevance"] = "relevance_pred"
    if "confidence" in predictions.columns:
        pred_cols.append("confidence")
        rename_map["confidence"] = "confidence_pred"

    pred_subset = predictions.select([pl.col(col) for col in pred_cols]).rename(
        rename_map
    )

    label_cols = [id_col, "theme"]
    label_rename = {"theme": "theme_true"}
    if "relevance" in labels.columns:
        label_cols.append("relevance")
        label_rename["relevance"] = "relevance_true"
    label_subset = labels.select(label_cols).rename(label_rename)

    joined = pred_subset.join(label_subset, on=id_col, how="inner")
    return joined


def evaluate_labels(
    preds_dir: Path,
    labels_dir: Path,
    out_dir: Path,
    gate_f1: float = DEFAULT_THEME_GATE,
) -> Dict[str, object]:
    preds_dir = Path(preds_dir)
    labels_dir = Path(labels_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    predictions = _load_predictions(preds_dir)
    label_frames = _load_labels(labels_dir)

    summary_sources: Dict[str, Dict[str, object]] = {}
    per_class_rows: List[Dict[str, object]] = []
    confusion_outputs: Dict[str, Dict[str, Dict[str, int]]] = {}
    overall_true: List[str] = []
    overall_pred: List[str] = []
    overall_conf: List[float] = []
    gates: List[Dict[str, object]] = []

    for source, info in THEME_SOURCES.items():
        id_col = info["id"]
        source_preds = predictions.filter(pl.col("source") == source)
        source_labels = label_frames[source]
        label_count = source_labels.height
        if label_count == 0:
            raise ValueError(f"Label file for {source} is empty")

        if source_preds.height == 0:
            summary_sources[source] = {
                "theme": {
                    "accuracy": 0.0,
                    "macro_f1": 0.0,
                    "macro_f1_ci": {"lower": 0.0, "upper": 0.0},
                    "ece": 0.0,
                    "coverage": 0.0,
                }
            }
            gates.append(
                {
                    "name": f"{source}_theme_macro_f1",
                    "value": 0.0,
                    "threshold": gate_f1,
                    "passed": False,
                }
            )
            gates.append(
                {
                    "name": f"{source}_coverage",
                    "value": 0.0,
                    "threshold": COVERAGE_GATE,
                    "passed": False,
                }
            )
            continue

        joined = _prepare_joined(source_preds, source_labels, id_col)
        coverage = joined.height / label_count if label_count else 0.0
        classes = _unique_classes(
            joined["theme_true"].to_list(), joined["theme_pred"].to_list()
        )
        theme_confidences = (
            joined["confidence_pred"].to_list()
            if "confidence_pred" in joined.columns
            else (
                joined["relevance_pred"].to_list()
                if "relevance_pred" in joined.columns
                else [1.0] * joined.height
            )
        )
        metrics = _classification_metrics(
            joined["theme_true"].to_list(),
            joined["theme_pred"].to_list(),
            theme_confidences,
            classes,
            coverage,
        )
        summary_sources[source] = {
            "theme": {
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "macro_f1_ci": metrics["macro_f1_ci"],
                "ece": metrics["ece"],
                "coverage": metrics["coverage"],
            }
        }
        confusion_outputs[f"confusion_{source}_theme.json"] = metrics["confusion"]  # type: ignore[assignment]
        gates.append(
            {
                "name": f"{source}_theme_macro_f1",
                "value": metrics["macro_f1"],
                "threshold": gate_f1,
                "passed": metrics["macro_f1"] >= gate_f1,
            }
        )
        gates.append(
            {
                "name": f"{source}_coverage",
                "value": coverage,
                "threshold": COVERAGE_GATE,
                "passed": coverage >= COVERAGE_GATE,
            }
        )

        for cls, cls_metrics in metrics["per_class"].items():
            per_class_rows.append(
                {
                    "source": source,
                    "task": "theme",
                    "class": cls,
                    **cls_metrics,
                }
            )

        overall_true.extend(joined["theme_true"].to_list())
        overall_pred.extend(joined["theme_pred"].to_list())
        overall_conf.extend(theme_confidences)

        if (
            source in {"slack", "email"}
            and "relevance_true" in joined.columns
            and "relevance_pred" in joined.columns
        ):
            rel_true = [
                "positive" if float(val) >= 0.5 else "negative"
                for val in joined["relevance_true"].to_list()
            ]
            rel_pred_scores = [
                max(0.0, min(1.0, float(val)))
                for val in joined["relevance_pred"].to_list()
            ]
            rel_pred = [
                "positive" if score >= 0.5 else "negative" for score in rel_pred_scores
            ]
            rel_classes = _unique_classes(rel_true, rel_pred)
            rel_metrics = _classification_metrics(
                rel_true, rel_pred, rel_pred_scores, rel_classes, coverage
            )
            summary_sources[source]["relevance"] = {
                "accuracy": rel_metrics["accuracy"],
                "macro_f1": rel_metrics["macro_f1"],
                "macro_f1_ci": rel_metrics["macro_f1_ci"],
                "ece": rel_metrics["ece"],
            }
            confusion_outputs[f"confusion_{source}_relevance.json"] = rel_metrics["confusion"]  # type: ignore[assignment]
            gates.append(
                {
                    "name": f"{source}_relevance_macro_f1",
                    "value": rel_metrics["macro_f1"],
                    "threshold": max(gate_f1, RELEVANCE_GATE),
                    "passed": rel_metrics["macro_f1"] >= max(gate_f1, RELEVANCE_GATE),
                }
            )
            for cls, cls_metrics in rel_metrics["per_class"].items():
                per_class_rows.append(
                    {
                        "source": source,
                        "task": "relevance",
                        "class": cls,
                        **cls_metrics,
                    }
                )

    overall_classes = _unique_classes(overall_true, overall_pred)
    overall_metrics = _classification_metrics(
        overall_true, overall_pred, overall_conf, overall_classes, 1.0
    )

    summary = {
        "overall": {
            "accuracy": overall_metrics["accuracy"],
            "macro_f1": overall_metrics["macro_f1"],
            "macro_f1_ci": overall_metrics["macro_f1_ci"],
            "ece": overall_metrics["ece"],
        },
        "sources": summary_sources,
    }

    gates_pass = all(gate["passed"] for gate in gates)
    summary["gates"] = {gate["name"]: gate for gate in gates}
    summary["gates_pass"] = gates_pass

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    if per_class_rows:
        pl.DataFrame(per_class_rows).write_csv(out_dir / "per_class.csv")

    for filename, matrix in confusion_outputs.items():
        (out_dir / filename).write_text(json.dumps(matrix, indent=2), encoding="utf-8")

    return summary
