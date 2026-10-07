"""Leakage-safe helpers for the CSV-backed autoencoder experiment."""

from __future__ import annotations

import hashlib
from itertools import combinations
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from app.ml.android_feature_schema import ANDROID_FEATURE_NAMES
from app.novelty.preprocessing import FeaturePreprocessor

SEED_DEFAULT = 2026
CLASS_ID_TO_NAME = {
    1: "Adware",
    2: "Banking",
    3: "SMS",
    4: "Riskware",
    5: "Benign",
}
KNOWN_CLASS_NAMES = ("Adware", "Banking", "Benign")
NOVEL_CLASS_NAMES = ("SMS", "Riskware")


def validate_class_mapping(frame: pd.DataFrame) -> None:
    """Ensure the CSV's numeric and textual labels match the verified mapping."""
    if "Class" not in frame or "Class_Name" not in frame:
        raise ValueError("Dataset must contain both 'Class' and 'Class_Name' labels")
    observed = frame[["Class", "Class_Name"]].drop_duplicates()
    for row in observed.itertuples(index=False):
        class_id, class_name = int(row.Class), str(row.Class_Name)
        if CLASS_ID_TO_NAME.get(class_id) != class_name:
            raise ValueError(
                f"Unexpected dataset label mapping: Class={class_id}, Class_Name={class_name!r}"
            )
    if set(frame["Class_Name"].dropna().astype(str)) - set(CLASS_ID_TO_NAME.values()):
        raise ValueError("Dataset contains a class name outside the verified label mapping")


def split_training_known(
    known_rows: pd.DataFrame,
    *,
    seed: int = SEED_DEFAULT,
    monitor_fraction: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split known train.csv rows into model-fit and train-only monitor subsets."""
    if not 0 < monitor_fraction < 1:
        raise ValueError("monitor_fraction must be between 0 and 1")
    fit_indices, monitor_indices = train_test_split(
        known_rows.index.to_numpy(),
        test_size=monitor_fraction,
        random_state=seed,
        stratify=known_rows["Class_Name"],
    )
    return known_rows.loc[fit_indices].copy(), known_rows.loc[monitor_indices].copy()


def fit_training_preprocessor(
    train_fit_features: pd.DataFrame,
) -> tuple[FeaturePreprocessor, np.ndarray]:
    """Fit scaling only on the model-fitting subset and preserve its column order."""
    if tuple(train_fit_features.columns) != ANDROID_FEATURE_NAMES:
        raise ValueError("Training-fit features do not match the canonical 470-column schema")
    values = train_fit_features.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Training-fit features contain non-finite values")
    preprocessor = FeaturePreprocessor()
    preprocessor.feature_columns = list(ANDROID_FEATURE_NAMES)
    return preprocessor, preprocessor.fit_transform(train_fit_features)


def select_validation_threshold(
    validation_errors: Sequence[float],
    validation_is_novel: Sequence[bool],
    *,
    method: str = "max_f1",
    target_fpr: float = 0.05,
) -> tuple[float, dict[str, Any]]:
    """Choose a reconstruction-error threshold using validation data only."""
    errors = np.asarray(validation_errors, dtype=np.float64)
    labels = np.asarray(validation_is_novel, dtype=bool)
    if errors.ndim != 1 or labels.ndim != 1 or len(errors) != len(labels) or not len(errors):
        raise ValueError("Validation errors and labels must be non-empty, matching 1D arrays")
    if not np.isfinite(errors).all() or len(np.unique(labels)) != 2:
        raise ValueError("Validation calibration needs finite errors and both known/novel labels")
    if method not in {"max_f1", "target_fpr"}:
        raise ValueError("threshold method must be 'max_f1' or 'target_fpr'")
    if not 0 <= target_fpr < 1:
        raise ValueError("target_fpr must be between 0 (inclusive) and 1 (exclusive)")

    candidates = np.unique(np.concatenate((
        errors,
        [np.nextafter(errors.min(), -np.inf), np.nextafter(errors.max(), np.inf)],
    )))
    rows: list[tuple[float, float, float, float]] = []
    known = ~labels
    for threshold in candidates:
        predicted = errors > threshold
        tp = int(np.sum(predicted & labels))
        fp = int(np.sum(predicted & known))
        fn = int(np.sum(~predicted & labels))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        fpr = fp / int(known.sum())
        rows.append((float(threshold), f1, precision, fpr))

    if method == "max_f1":
        # Stable tie-break: precision, lower FPR, then the more conservative threshold.
        threshold, f1, precision, fpr = max(rows, key=lambda row: (row[1], row[2], -row[3], row[0]))
        criterion = {"objective": "maximum validation F1"}
    else:
        feasible = [row for row in rows if row[3] <= target_fpr]
        threshold, f1, precision, fpr = max(
            feasible, key=lambda row: (row[1], row[2], -row[3], row[0])
        )
        criterion = {"objective": "maximum validation F1 subject to target FPR",
                     "target_fpr": target_fpr}
    return threshold, {
        "method": method,
        "threshold": threshold,
        "validation_f1": f1,
        "validation_precision": precision,
        "validation_false_positive_rate": fpr,
        "candidate_threshold_count": len(rows),
        "criterion": criterion,
        "tie_break": "higher precision, lower false-positive rate, then higher threshold",
    }


def evaluate_errors(
    errors: Sequence[float],
    class_names: Sequence[str],
    threshold: float,
) -> dict[str, Any]:
    """Evaluate known-versus-novel detection and summarize errors by class."""
    values = np.asarray(errors, dtype=np.float64)
    classes = np.asarray(class_names, dtype=str)
    if values.ndim != 1 or classes.ndim != 1 or len(values) != len(classes):
        raise ValueError("Errors and class labels must be matching 1D arrays")
    if not np.isfinite(values).all():
        raise ValueError("Reconstruction errors contain non-finite values")
    is_novel = np.isin(classes, NOVEL_CLASS_NAMES)
    predicted_novel = values > threshold
    tp = int(np.sum(predicted_novel & is_novel))
    fp = int(np.sum(predicted_novel & ~is_novel))
    fn = int(np.sum(~predicted_novel & is_novel))
    tn = int(np.sum(~predicted_novel & ~is_novel))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    result: dict[str, Any] = {
        "threshold": float(threshold),
        "sample_count": len(values),
        "known_sample_count": int((~is_novel).sum()),
        "novel_sample_count": int(is_novel.sum()),
        "confusion_matrix": {"labels": ["known", "novel"], "matrix": [[tn, fp], [fn, tp]]},
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fp / (tn + fp) if tn + fp else 0.0,
        "false_negative_rate": fn / (tp + fn) if tp + fn else 0.0,
    }
    if len(np.unique(is_novel)) == 2:
        result["roc_auc"] = float(roc_auc_score(is_novel, values))
        result["pr_auc"] = float(average_precision_score(is_novel, values))
    else:
        result["roc_auc"] = None
        result["pr_auc"] = None

    def stats(group_errors: np.ndarray) -> dict[str, float | int | None]:
        if not len(group_errors):
            return {"count": 0, "mean": None, "median": None, "std": None,
                    "min": None, "max": None, "p95": None}
        return {"count": int(len(group_errors)), "mean": float(np.mean(group_errors)),
                "median": float(np.median(group_errors)), "std": float(np.std(group_errors)),
                "min": float(np.min(group_errors)), "max": float(np.max(group_errors)),
                "p95": float(np.percentile(group_errors, 95))}

    result["reconstruction_error_statistics"] = {
        "known": stats(values[~is_novel]),
        "novel": stats(values[is_novel]),
        "by_class": {
            name: {
                **stats(values[classes == name]),
                "novelty_detection_rate": float(np.mean(predicted_novel[classes == name]))
                if np.any(classes == name) else None,
                "actual_group": "novel" if name in NOVEL_CLASS_NAMES else "known",
            }
            for name in CLASS_ID_TO_NAME.values()
        },
    }
    return result


def feature_overlap_report(
    datasets: dict[str, pd.DataFrame],
    feature_columns: Sequence[str] = ANDROID_FEATURE_NAMES,
) -> dict[str, Any]:
    """Count exact feature-vector overlaps without asserting APK identity."""
    all_keys: dict[str, set[bytes]] = {}
    known_keys: dict[str, set[bytes]] = {}
    for split, frame in datasets.items():
        matrix = frame[list(feature_columns)].to_numpy(dtype=np.float64)
        if not np.isfinite(matrix).all():
            raise ValueError(f"{split} dataset contains non-finite feature values")
        all_keys[split] = {row.tobytes() for row in matrix}
        known = frame[frame["Class_Name"].isin(KNOWN_CLASS_NAMES)]
        known_matrix = known[list(feature_columns)].to_numpy(dtype=np.float64)
        known_keys[split] = {row.tobytes() for row in known_matrix}

    report = {}
    for left, right in combinations(datasets, 2):
        report[f"{left}_{right}"] = {
            "all_label_unique_vector_overlap_count": len(all_keys[left] & all_keys[right]),
            "known_class_unique_vector_overlap_count": len(known_keys[left] & known_keys[right]),
            "identity_claimed": False,
        }
    return report


def class_distribution(frame: pd.DataFrame) -> dict[str, int]:
    return {name: int((frame["Class_Name"] == name).sum()) for name in CLASS_ID_TO_NAME.values()}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_experiment_metadata(metadata: dict[str, Any]) -> None:
    """Guard the persisted experiment record against losing reproducibility fields."""
    required = {
        "version", "seed", "class_id_to_name", "training_known_classes",
        "validation_novel_classes", "feature_schema", "scaler_fit_rows",
        "model_fit_rows", "threshold_selection", "dataset_sha256",
        "possible_feature_level_duplicates", "test_metrics",
        "runtime_feature_compatibility",
    }
    missing = sorted(required - metadata.keys())
    if missing:
        raise ValueError(f"Experiment metadata is missing required fields: {', '.join(missing)}")
    schema = metadata["feature_schema"]
    if schema.get("feature_count") != len(ANDROID_FEATURE_NAMES):
        raise ValueError("Persisted experiment metadata must describe the canonical 470-feature schema")
    if tuple(schema.get("ordered_names", ())) != ANDROID_FEATURE_NAMES:
        raise ValueError("Persisted experiment feature order does not match the canonical schema")
