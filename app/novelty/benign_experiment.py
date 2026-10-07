"""Controlled benign-only autoencoder novelty experiment (experimental; unregistered)."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from app.config import get_settings
from app.ml.android_feature_schema import ANDROID_FEATURE_NAMES
from app.novelty.autoencoder import build_autoencoder
from app.novelty.dataset_experiment import (
    CLASS_ID_TO_NAME,
    class_distribution,
    select_validation_threshold,
    validate_class_mapping,
)

SEED_DEFAULT = 2026
VERSION = "0.3.0"
EXPERIMENT_NAME = "android-novelty-autoencoder"
HUBER_DELTA = 1.0
ROBUST_CLIP = 5.0
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent.parent / "data"
TRAIN_DATA = DATA_DIR / "train.csv"
VALIDATION_DATA = DATA_DIR / "validation.csv"
TEST_DATA = DATA_DIR / "test.csv"
logger = logging.getLogger(__name__)


class BenignRobustPreprocessor:
    """Fit-only benign transform: preserve learned binary columns, robustly scale counts."""

    def __init__(self, columns: Sequence[str] = ANDROID_FEATURE_NAMES, clip: float = ROBUST_CLIP):
        self.columns = tuple(columns)
        self.clip = float(clip)
        self.binary_mask: np.ndarray | None = None
        self.medians: np.ndarray | None = None
        self.iqrs: np.ndarray | None = None

    def fit(self, benign_train_fit: pd.DataFrame) -> "BenignRobustPreprocessor":
        if tuple(benign_train_fit.columns) != self.columns:
            raise ValueError("Benign fit data must match the exact ordered 470-feature schema")
        values = benign_train_fit.to_numpy(dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.columns) or not np.isfinite(values).all():
            raise ValueError("Benign fit data must be a finite two-dimensional feature matrix")
        if np.any(values < 0):
            raise ValueError("log1p count transformation requires nonnegative feature values")
        # Binary type is inferred from the fit population alone; constant columns
        # are treated as counts so held-out nonzero observations remain transformable.
        self.binary_mask = np.array([
            np.array_equal(np.unique(values[:, index]), np.array([0.0, 1.0]))
            for index in range(values.shape[1])
        ], dtype=bool)
        # Fit count-branch parameters for every feature. This also handles a
        # column that is only 0/1 among Benign rows but has count values later.
        logged = np.log1p(values)
        q25, self.medians, q75 = np.percentile(logged, [25, 50, 75], axis=0)
        self.iqrs = q75 - q25
        self.iqrs[self.iqrs <= np.finfo(np.float64).eps] = 1.0
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.binary_mask is None or self.medians is None or self.iqrs is None:
            raise RuntimeError("Preprocessor must be fitted on Benign training data first")
        if tuple(frame.columns) != self.columns:
            raise ValueError("Input features do not match the exact ordered 470-feature schema")
        values = frame.to_numpy(dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.columns) or not np.isfinite(values).all():
            raise ValueError("Input features must be finite and have exactly 470 columns")
        if np.any(values < 0):
            raise ValueError("log1p count transformation requires nonnegative feature values")
        # Exact 0/1 values in fit-inferred binary columns remain unchanged.
        # If that column carries a count >1 at inference, use the already
        # Benign-fitted robust count branch instead of misclassifying it.
        preserve_binary = np.broadcast_to(self.binary_mask, values.shape) & np.isin(values, [0.0, 1.0])
        result = np.clip(
            (np.log1p(values) - self.medians) / self.iqrs,
            -self.clip,
            self.clip,
        )
        result[preserve_binary] = values[preserve_binary]
        if not np.isfinite(result).all():
            raise ValueError("Preprocessing produced non-finite transformed features")
        return result.astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        if self.binary_mask is None or self.medians is None or self.iqrs is None:
            raise RuntimeError("Cannot serialize an unfitted preprocessor")
        return {
            "feature_count": len(self.columns),
            "ordered_feature_names": list(self.columns),
            "binary_feature_names": [name for name, binary in zip(self.columns, self.binary_mask) if binary],
            "count_feature_names": [name for name, binary in zip(self.columns, self.binary_mask) if not binary],
            "binary_rule": "Columns whose Benign-fit support is exactly {0, 1}: observed 0/1 values pass through unchanged",
            "count_transform": "All other values use log1p(x), subtract Benign-fit median, divide by Benign-fit IQR, clip to [-5, 5]",
            "binary_column_count_fallback": "For a fit-inferred binary column, an observed value outside {0, 1} uses that column's Benign-fit robust count transform",
            "zero_iqr_rule": "IQR values at or below machine epsilon are replaced with 1",
            "clip": self.clip,
            "fit_population": "All Benign rows from train.csv only; no validation or test rows",
            "medians": self.medians.tolist(),
            "iqrs": self.iqrs.tolist(),
        }


def split_benign_train_rows(train: pd.DataFrame, seed: int = SEED_DEFAULT
                            ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select only verified Benign train rows, then make a deterministic train-only split."""
    validate_class_mapping(train)
    benign = train[train["Class"].eq(5) & train["Class_Name"].eq("Benign")]
    if len(benign) < 20:
        raise ValueError("Benign-only training population has fewer than 20 rows")
    fit_idx, monitor_idx = train_test_split(
        benign.index.to_numpy(), test_size=0.20, random_state=seed, shuffle=True,
    )
    return train.loc[fit_idx].copy(), train.loc[monitor_idx].copy()


def huber_reconstruction_errors(original: np.ndarray, reconstructed: np.ndarray,
                                delta: float = HUBER_DELTA) -> np.ndarray:
    """Return per-sample mean Huber loss over features, matching Keras training loss."""
    original = np.asarray(original, dtype=np.float64)
    reconstructed = np.asarray(reconstructed, dtype=np.float64)
    if original.shape != reconstructed.shape or original.ndim != 2:
        raise ValueError("Original and reconstructed matrices must have the same 2D shape")
    if not np.isfinite(original).all() or not np.isfinite(reconstructed).all() or delta <= 0:
        raise ValueError("Huber scoring requires finite matrices and a positive delta")
    residual = np.abs(original - reconstructed)
    element_loss = np.where(residual <= delta, 0.5 * residual**2,
                            delta * (residual - 0.5 * delta))
    return element_loss.mean(axis=1)


def _read_dataset(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    validate_class_mapping(frame)
    missing = [name for name in ANDROID_FEATURE_NAMES if name not in frame.columns]
    extras = [name for name in frame.columns if name not in ANDROID_FEATURE_NAMES
              and name not in {"Class", "Class_Name"}]
    if missing or extras:
        raise ValueError(f"{path.name} schema mismatch: missing={missing[:5]}, unexpected={extras[:5]}")
    frame = frame.loc[:, [*ANDROID_FEATURE_NAMES, "Class", "Class_Name"]]
    frame.loc[:, ANDROID_FEATURE_NAMES] = frame.loc[:, ANDROID_FEATURE_NAMES].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(frame.loc[:, ANDROID_FEATURE_NAMES].to_numpy(dtype=np.float64)).all():
        raise ValueError(f"{path.name} contains non-finite values")
    return frame


def _errors(model: tf.keras.Model, transformed: np.ndarray) -> np.ndarray:
    predictions = model(transformed.astype(np.float32, copy=False), training=False).numpy()
    return huber_reconstruction_errors(transformed, predictions)


def _metrics(errors: np.ndarray, class_names: Sequence[str], threshold: float) -> dict[str, Any]:
    values = np.asarray(errors, dtype=np.float64)
    classes = np.asarray(class_names, dtype=str)
    if values.ndim != 1 or len(values) != len(classes) or not np.isfinite(values).all():
        raise ValueError("Evaluation requires matching finite reconstruction errors and labels")
    is_novel = classes != "Benign"
    predicted = values > threshold
    tp = int(np.sum(predicted & is_novel))
    fp = int(np.sum(predicted & ~is_novel))
    fn = int(np.sum(~predicted & is_novel))
    tn = int(np.sum(~predicted & ~is_novel))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    result: dict[str, Any] = {
        "threshold": float(threshold), "sample_count": len(values),
        "known_sample_count": int((~is_novel).sum()), "novel_sample_count": int(is_novel.sum()),
        "confusion_matrix": {"labels": ["known", "novel"], "matrix": [[tn, fp], [fn, tp]]},
        "true_positive": tp, "false_positive": fp, "true_negative": tn, "false_negative": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "false_positive_rate": fp / (tn + fp) if tn + fp else 0.0,
        "false_negative_rate": fn / (tp + fn) if tp + fn else 0.0,
        "roc_auc": float(roc_auc_score(is_novel, values)),
        "pr_auc": float(average_precision_score(is_novel, values)),
    }
    class_statistics = {
        name: {**_distribution(values[classes == name]),
               "novelty_detection_rate": float(np.mean(predicted[classes == name]))
               if np.any(classes == name) else None}
        for name in ("Benign", "Adware", "Banking", "SMS", "Riskware")
    }
    result["reconstruction_error_statistics"] = {
        "known": _distribution(values[~is_novel]),
        "novel": _distribution(values[is_novel]),
        "by_class": class_statistics,
    }
    result["novel_class_counts"] = {
        name: int(np.sum(classes == name)) for name in ("Adware", "Banking", "SMS", "Riskware")
    }
    result["known_class_counts"] = {"Benign": int(np.sum(classes == "Benign"))}
    known_stats = result["reconstruction_error_statistics"]["known"]
    novel_stats = result["reconstruction_error_statistics"]["novel"]
    result["median_novel_minus_known"] = novel_stats["median"] - known_stats["median"]
    result["median_novel_to_known_ratio"] = (
        novel_stats["median"] / known_stats["median"] if known_stats["median"] else None
    )
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def _save_weights(model: tf.keras.Model, path: Path) -> None:
    np.savez_compressed(path, **{
        f"weight_{index:03d}": values for index, values in enumerate(model.get_weights())
    })


def load_experiment_model(artifact_dir: str | Path) -> tf.keras.Model:
    artifact_dir = Path(artifact_dir)
    config = json.loads((artifact_dir / "model_config.json").read_text(encoding="utf-8"))
    model = build_autoencoder(int(config["input_dim"]), int(config["latent_dim"]))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss=tf.keras.losses.Huber(delta=float(config["huber_delta"])))
    with np.load(artifact_dir / "autoencoder_weights.npz", allow_pickle=False) as archive:
        model.set_weights([archive[key] for key in sorted(archive.files)])
    return model


def run_experiment(seed: int = SEED_DEFAULT) -> dict[str, Any]:
    """Train Benign-only, calibrate on validation, then evaluate test exactly once."""
    artifact_root = Path(get_settings().model_registry_dir).resolve() / EXPERIMENT_NAME
    artifact_dir = artifact_root / VERSION
    if artifact_dir.exists():
        raise FileExistsError(f"Experimental version is immutable and already exists: {artifact_dir}")

    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        logger.warning("TensorFlow deterministic operations unavailable")

    train = _read_dataset(TRAIN_DATA)
    benign_train = train[train["Class"].eq(5) & train["Class_Name"].eq("Benign")]
    fit_rows, monitor_rows = split_benign_train_rows(train, seed)
    fit_idx, monitor_idx = fit_rows.index.to_numpy(), monitor_rows.index.to_numpy()
    columns = list(ANDROID_FEATURE_NAMES)
    preprocessor = BenignRobustPreprocessor().fit(benign_train.loc[:, columns])
    fit_values = preprocessor.transform(fit_rows.loc[:, columns])
    monitor_values = preprocessor.transform(monitor_rows.loc[:, columns])

    model = build_autoencoder(input_dim=len(columns), latent_dim=32)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.Huber(delta=HUBER_DELTA),
    )
    history = model.fit(
        fit_values, fit_values,
        validation_data=(monitor_values, monitor_values),
        epochs=50, batch_size=64, shuffle=True,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10,
                                             restore_best_weights=True, verbose=1),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                                 patience=5, verbose=1),
        ],
        verbose=1,
    )

    # Validation is first opened after preprocessing and model fitting are fixed.
    validation = _read_dataset(VALIDATION_DATA)
    val_values = preprocessor.transform(validation.loc[:, columns])
    val_errors = _errors(model, val_values)
    val_is_novel = validation["Class"].isin([1, 2, 3, 4]).to_numpy()
    threshold, calibration = select_validation_threshold(
        val_errors, val_is_novel, method="max_f1"
    )
    validation_metrics = _metrics(val_errors, validation["Class_Name"], threshold)

    # Final test is loaded and evaluated once, after threshold selection.
    test = _read_dataset(TEST_DATA)
    test_values = preprocessor.transform(test.loc[:, columns])
    test_errors = _errors(model, test_values)
    test_metrics = _metrics(test_errors, test["Class_Name"], threshold)

    # The existing 0.2.0 experiment is read only for a metric comparison.
    prior_path = artifact_root / "0.2.0" / "metadata.json"
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    prior_metrics = prior["test_metrics"]
    comparison = {
        "0.2.0_known_population": prior["training_known_classes"],
        "0.3.0_known_population": ["Benign"],
        "test_metrics": {
            "roc_auc": {"0.2.0": prior_metrics["roc_auc"], "0.3.0": test_metrics["roc_auc"]},
            "pr_auc": {"0.2.0": prior_metrics["pr_auc"], "0.3.0": test_metrics["pr_auc"]},
            "f1": {"0.2.0": prior_metrics["f1"], "0.3.0": test_metrics["f1"]},
            "false_positive_rate": {
                "0.2.0": prior_metrics["false_positive_rate"],
                "0.3.0": test_metrics["false_positive_rate"],
            },
        },
        "distribution_separation": {
            "median_novel_minus_known": {
                "0.2.0": prior_metrics["reconstruction_error_statistics"]["novel"]["median"]
                         - prior_metrics["reconstruction_error_statistics"]["known"]["median"],
                "0.3.0": test_metrics["median_novel_minus_known"],
            },
            "median_novel_to_known_ratio": {
                "0.2.0": prior_metrics["reconstruction_error_statistics"]["novel"]["median"]
                         / prior_metrics["reconstruction_error_statistics"]["known"]["median"],
                "0.3.0": test_metrics["median_novel_to_known_ratio"],
            },
            "test_roc_auc_is_rank_separation": True,
        },
        "test_comparison_used_for_tuning": False,
    }

    fit_predictions = model(fit_values, training=False).numpy()
    monitor_predictions = model(monitor_values, training=False).numpy()
    training_diagnostics = {
        "epochs_completed": len(history.history["loss"]),
        "best_monitor_loss": float(min(history.history["val_loss"])),
        "training_loss_by_epoch": [float(value) for value in history.history["loss"]],
        "monitor_loss_by_epoch": [float(value) for value in history.history["val_loss"]],
        "train_fit_reconstruction": _distribution(huber_reconstruction_errors(fit_values, fit_predictions)),
        "train_monitor_reconstruction": _distribution(huber_reconstruction_errors(monitor_values, monitor_predictions)),
    }
    preprocessor_metadata = preprocessor.to_dict()
    schema_hash = hashlib.sha256("\n".join(columns).encode()).hexdigest()
    distributions = {
        "train": class_distribution(train),
        "validation": class_distribution(validation),
        "test": class_distribution(test),
    }
    metadata = {
        "name": EXPERIMENT_NAME,
        "version": VERSION,
        "status": "experimental_unregistered",
        "seed": seed,
        "class_id_to_name": {str(key): value for key, value in CLASS_ID_TO_NAME.items()},
        "known_training_classes": ["Benign"],
        "novel_classes": ["Adware", "Banking", "SMS", "Riskware"],
        "class_distributions": distributions,
        "feature_schema": {"count": len(columns), "ordered_names": columns, "sha256": schema_hash},
        "preprocessing": preprocessor_metadata,
        "objective": {"training_loss": "per-feature mean Huber loss", "delta": HUBER_DELTA,
                      "scoring": "mean per-feature Huber loss; higher is more novel"},
        "threshold_selection": calibration,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "training_diagnostics": training_diagnostics,
        "dataset_sha256": {"train": _sha256(TRAIN_DATA), "validation": _sha256(VALIDATION_DATA),
                           "test": _sha256(TEST_DATA)},
        "runtime_compatibility": {
            "fully_validated": False,
            "deterministic_extraction_coverage": "435/470",
            "unsupported_features": 35,
        },
        "registered": False,
        "comparison": comparison,
    }
    split_metadata = {
        "seed": seed,
        "method": "80/20 shuffled split of Class=5, Class_Name=Benign rows from train.csv; fixed random_state",
        "fit_row_count": len(fit_idx), "fit_source_row_indexes": [int(i) for i in fit_idx],
        "monitor_row_count": len(monitor_idx), "monitor_source_row_indexes": [int(i) for i in monitor_idx],
        "validation_role": "threshold calibration and independent validation metrics",
        "test_role": "one final evaluation after all fit and threshold decisions",
        "class_distributions": distributions,
        "test_source_row_indexes": [int(i) for i in test.index],
    }

    artifact_root.mkdir(parents=True, exist_ok=True)
    if artifact_dir.exists():
        raise FileExistsError(f"Experimental version is immutable and already exists: {artifact_dir}")
    with tempfile.TemporaryDirectory(prefix=f".{VERSION}-", dir=artifact_root) as temporary:
        staging = Path(temporary)
        _save_weights(model, staging / "autoencoder_weights.npz")
        _write_json(staging / "model_config.json", {
            "builder": "app.novelty.autoencoder.build_autoencoder", "input_dim": len(columns),
            "latent_dim": 32, "loss": "Huber", "huber_delta": HUBER_DELTA,
            "architecture_changed": False,
        })
        joblib.dump(preprocessor, staging / "preprocessor.pkl")
        _write_json(staging / "preprocessing.json", preprocessor_metadata)
        pd.Series(columns).to_json(staging / "feature_columns.json")
        _write_json(staging / "threshold.json", {"threshold": float(threshold), **calibration})
        _write_json(staging / "evaluation.json", {
            "validation": validation_metrics, "test": test_metrics, "comparison": comparison,
        })
        _write_json(staging / "metadata.json", metadata)
        _write_json(staging / "split_metadata.json", split_metadata)
        (staging / "README.md").write_text(_experiment_readme(metadata), encoding="utf-8")
        os.replace(staging, artifact_dir)
    return {"artifact_dir": str(artifact_dir), "metadata": metadata}


def _distribution(errors: np.ndarray) -> dict[str, float]:
    values = np.asarray(errors, dtype=np.float64)
    return {
        "count": int(len(values)), "mean": float(np.mean(values)),
        "median": float(np.median(values)), "std": float(np.std(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)), "p99": float(np.percentile(values, 99)),
        "min": float(np.min(values)), "max": float(np.max(values)),
    }


def _experiment_readme(metadata: dict[str, Any]) -> str:
    return """# Android Novelty Autoencoder 0.3.0 (experimental, unregistered)

This is a controlled Benign-only dataset experiment. It is not registered, not
connected to the runtime six-layer novelty detector, and not approved for use.

## Protocol

- Known/training class: Class 5 (Benign) only.
- Held-out novel classes: Adware, Banking, SMS, and Riskware.
- Feature typing and robust count parameters are fitted on all Benign rows in
  `train.csv` only. The autoencoder weights are fitted on a deterministic
  Benign-only training subset; its separate Benign monitor subset is used only
  for early stopping.
- Binary candidates inferred from Benign training support pass 0/1 values
  through unchanged. Remaining values use log1p, Benign-fit median/IQR scaling,
  and a [-5, 5] cap. A candidate binary column that carries a value above 1 is
  treated through its already-fitted count transform. The feature order remains
  the canonical 470-column schema.
- The unchanged dense autoencoder architecture is trained with Huber loss
  (delta 1); scoring uses the same per-feature Huber aggregation.
- Threshold is selected on validation data by maximum F1. Test data is only
  opened after fitting and threshold selection and is used once for final metrics.

## Runtime limitation

The APK extractor has deterministic mappings for 435/470 features. Thirty-five
features remain unavailable, so this experiment does not validate runtime APK
compatibility. See `metadata.json`, `split_metadata.json`, `preprocessing.json`,
`threshold.json`, and `evaluation.json` for recorded results.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    args = parser.parse_args()
    print(json.dumps(run_experiment(args.seed), indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
