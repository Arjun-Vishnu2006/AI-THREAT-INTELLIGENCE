"""Train and evaluate an isolated, leakage-safe CSV autoencoder experiment."""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from app.config import get_settings
from app.ml.android_feature_schema import ANDROID_FEATURE_NAMES
from app.novelty.autoencoder import build_autoencoder
from app.novelty.dataset_experiment import (
    CLASS_ID_TO_NAME,
    KNOWN_CLASS_NAMES,
    NOVEL_CLASS_NAMES,
    SEED_DEFAULT,
    class_distribution,
    evaluate_errors,
    feature_overlap_report,
    fit_training_preprocessor,
    select_validation_threshold,
    sha256_file,
    split_training_known,
    validate_class_mapping,
    validate_experiment_metadata,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent.parent / "data"
TRAIN_DATA = DATA_DIR / "train.csv"
VALIDATION_DATA = DATA_DIR / "validation.csv"
TEST_DATA = DATA_DIR / "test.csv"
EXPERIMENT_NAME = "android-novelty-autoencoder"
EXPERIMENT_VERSION = "0.2.0"
EPOCHS = 50
BATCH_SIZE = 64

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _read_dataset(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    validate_class_mapping(frame)
    missing = [name for name in ANDROID_FEATURE_NAMES if name not in frame.columns]
    extras = [name for name in frame.columns if name not in ANDROID_FEATURE_NAMES and name not in {"Class", "Class_Name"}]
    if missing or extras:
        raise ValueError(f"{path.name} schema mismatch: missing={missing[:5]}, unexpected={extras[:5]}")
    frame = frame.loc[:, [*ANDROID_FEATURE_NAMES, "Class", "Class_Name"]]
    features = frame.loc[:, ANDROID_FEATURE_NAMES].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(features.to_numpy(dtype=np.float64)).all():
        raise ValueError(f"{path.name} contains non-finite feature values")
    frame.loc[:, ANDROID_FEATURE_NAMES] = features
    return frame


def _reconstruction_errors(model: tf.keras.Model, values: np.ndarray, batch_size: int = 256) -> np.ndarray:
    errors: list[np.ndarray] = []
    for start in range(0, len(values), batch_size):
        batch = values[start:start + batch_size].astype(np.float32, copy=False)
        reconstructed = model(batch, training=False).numpy()
        errors.append(np.mean(np.square(batch - reconstructed), axis=1))
    return np.concatenate(errors) if errors else np.empty(0, dtype=np.float64)


def _json_write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def _persist_experiment(
    *, version: str, seed: int, model: tf.keras.Model, preprocessor: Any,
    threshold: float, threshold_metadata: dict[str, Any], metadata: dict[str, Any],
    evaluation: dict[str, Any], split_metadata: dict[str, Any],
) -> Path:
    root = Path(get_settings().model_registry_dir).resolve() / EXPERIMENT_NAME
    root.mkdir(parents=True, exist_ok=True)
    final_dir = root / version
    if final_dir.exists():
        raise FileExistsError(f"Experiment version is immutable and already exists: {final_dir}")
    with tempfile.TemporaryDirectory(prefix=f".{version}-", dir=root) as temporary:
        staging = Path(temporary)
        logger.info("Saving experiment artifacts to staging directory %s", staging)
        # Keras .save() hangs while creating its archive on this Windows runtime.
        # Store NumPy weights and the unchanged builder configuration instead;
        # load_experiment_model() reconstructs the same architecture and restores them.
        np.savez_compressed(
            staging / "autoencoder_weights.npz",
            **{f"weight_{index:03d}": value for index, value in enumerate(model.get_weights())},
        )
        _json_write(staging / "model_config.json", {
            "builder": "app.novelty.autoencoder.build_autoencoder",
            "input_dim": len(ANDROID_FEATURE_NAMES),
            "latent_dim": 32,
            "architecture_changed": False,
        })
        logger.info("Saved autoencoder weights and unchanged model configuration")
        joblib.dump(preprocessor.scaler, staging / "scaler.pkl")
        logger.info("Saved fitted scaler")
        pd.Series(list(ANDROID_FEATURE_NAMES)).to_json(staging / "feature_columns.json")
        logger.info("Saved canonical feature schema")
        _json_write(staging / "threshold.json", {
            "threshold": float(threshold),
            "percentile": None,
            **threshold_metadata,
        })
        _json_write(staging / "metadata.json", metadata)
        _json_write(staging / "evaluation.json", evaluation)
        _json_write(staging / "split_metadata.json", split_metadata)
        logger.info("Saved experiment metadata and evaluation")
        os.replace(staging, final_dir)
    return final_dir


def load_experiment_model(artifact_dir: str | Path) -> tf.keras.Model:
    """Recreate the unchanged architecture and load a versioned weight archive."""
    artifact_dir = Path(artifact_dir)
    config = json.loads((artifact_dir / "model_config.json").read_text(encoding="utf-8"))
    if config.get("builder") != "app.novelty.autoencoder.build_autoencoder":
        raise ValueError("Unsupported experiment model builder")
    model = build_autoencoder(input_dim=int(config["input_dim"]), latent_dim=int(config["latent_dim"]))
    with np.load(artifact_dir / "autoencoder_weights.npz", allow_pickle=False) as archive:
        model.set_weights([archive[key] for key in sorted(archive.files)])
    return model


def run_experiment(
    version: str = EXPERIMENT_VERSION,
    seed: int = SEED_DEFAULT,
    threshold_method: str = "max_f1",
    target_fpr: float = 0.05,
) -> dict[str, Any]:
    """Fit on train.csv, calibrate on validation.csv, then evaluate test.csv."""
    if version != EXPERIMENT_VERSION:
        raise ValueError(f"This isolated experiment is versioned as {EXPERIMENT_VERSION}")

    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        logger.warning("TensorFlow op determinism is unavailable on this runtime")

    # Read only train.csv before the model and scaler have been fitted.
    train = _read_dataset(TRAIN_DATA)
    train_known = train[train["Class_Name"].isin(KNOWN_CLASS_NAMES)]
    train_fit, train_monitor = split_training_known(train_known, seed=seed)
    feature_columns = list(ANDROID_FEATURE_NAMES)
    preprocessor, fit_scaled = fit_training_preprocessor(train_fit[feature_columns])
    monitor_scaled = preprocessor.transform(train_monitor[feature_columns]).astype(np.float32)

    model = build_autoencoder(input_dim=len(feature_columns), latent_dim=32)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10,
                                         restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                             patience=5, verbose=1),
    ]
    history = model.fit(
        fit_scaled.astype(np.float32), fit_scaled.astype(np.float32),
        validation_data=(monitor_scaled, monitor_scaled),
        epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True,
        callbacks=callbacks, verbose=1,
    )

    # Validation is read only after the scaler and all model weights are fixed.
    validation = _read_dataset(VALIDATION_DATA)
    validation_scaled = preprocessor.transform(validation[feature_columns]).astype(np.float32)
    validation_errors = _reconstruction_errors(model, validation_scaled)
    validation_is_novel = validation["Class_Name"].isin(NOVEL_CLASS_NAMES).to_numpy()
    threshold, calibration = select_validation_threshold(
        validation_errors, validation_is_novel,
        method=threshold_method, target_fpr=target_fpr,
    )
    validation_metrics = evaluate_errors(validation_errors, validation["Class_Name"], threshold)

    # Test data is opened only after training, early stopping, and threshold selection.
    test = _read_dataset(TEST_DATA)
    test_scaled = preprocessor.transform(test[feature_columns]).astype(np.float32)
    test_errors = _reconstruction_errors(model, test_scaled)
    test_metrics = evaluate_errors(test_errors, test["Class_Name"], threshold)

    # Exact vectors can overlap across files; no sample identity is inferred.
    overlaps = feature_overlap_report({"train": train, "validation": validation, "test": test})
    distributions = {
        "train": class_distribution(train),
        "validation": class_distribution(validation),
        "test": class_distribution(test),
    }
    fit_index = [int(index) for index in train_fit.index]
    monitor_index = [int(index) for index in train_monitor.index]
    split_metadata = {
        "seed": seed,
        "method": "stratified 80/20 split of known-class rows from train.csv; original CSV row indexes persisted",
        "train_fit_row_count": len(fit_index),
        "train_monitor_row_count": len(monitor_index),
        "train_fit_source_indexes": fit_index,
        "train_monitor_source_indexes": monitor_index,
        "validation_role": "independent threshold calibration and error analysis",
        "test_role": "final evaluation only after calibration",
        "class_distributions": distributions,
        "possible_feature_level_duplicates": overlaps,
    }
    metadata = {
        "name": EXPERIMENT_NAME,
        "version": version,
        "task": "csv_autoencoder_novelty_experiment",
        "status": "completed",
        "seed": seed,
        "randomness": "TensorFlow seed and deterministic ops requested; stratified split uses random_state=seed",
        "class_id_to_name": {str(key): value for key, value in CLASS_ID_TO_NAME.items()},
        "training_known_classes": list(KNOWN_CLASS_NAMES),
        "validation_known_classes": list(KNOWN_CLASS_NAMES),
        "validation_novel_classes": list(NOVEL_CLASS_NAMES),
        "test_known_classes": list(KNOWN_CLASS_NAMES),
        "test_novel_classes": list(NOVEL_CLASS_NAMES),
        "label_semantics": "known means represented in training, not benign/normal; Adware and Banking are malware classes",
        "dataset_paths": {"train": str(TRAIN_DATA), "validation": str(VALIDATION_DATA), "test": str(TEST_DATA)},
        "dataset_sha256": {"train": sha256_file(TRAIN_DATA),
                           "validation": sha256_file(VALIDATION_DATA),
                           "test": sha256_file(TEST_DATA)},
        "feature_schema": {
            "feature_count": len(feature_columns),
            "ordered_names": feature_columns,
            "sha256": __import__("hashlib").sha256("\n".join(feature_columns).encode()).hexdigest(),
        },
        "scaler_fit_rows": len(train_fit),
        "model_fit_rows": len(train_fit),
        "training_monitor_rows": len(train_monitor),
        "epochs_completed": len(history.history["loss"]),
        "best_monitor_loss": float(min(history.history["val_loss"])),
        "threshold_selection": calibration,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "class_distributions": distributions,
        "possible_feature_level_duplicates": overlaps,
        "runtime_feature_compatibility": {
            "fully_validated": False,
            "unsupported_training_schema_features": 35,
            "note": "Dataset experiment only; runtime APK extraction does not yet reproduce all dataset feature semantics.",
        },
        "architecture": "Existing build_autoencoder(input_dim=470, latent_dim=32); unchanged",
        "model_serialization": "NumPy weight arrays plus builder configuration; load with load_experiment_model",
    }
    validate_experiment_metadata(metadata)
    artifact_dir = _persist_experiment(
        version=version, seed=seed, model=model, preprocessor=preprocessor,
        threshold=threshold, threshold_metadata=calibration,
        metadata=metadata, evaluation={"validation": validation_metrics, "test": test_metrics},
        split_metadata=split_metadata,
    )
    logger.info("Experiment artifacts finalized at %s", artifact_dir)
    return {"artifact_dir": str(artifact_dir), "metadata": metadata}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default=EXPERIMENT_VERSION)
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    parser.add_argument("--threshold-method", choices=("max_f1", "target_fpr"), default="max_f1")
    parser.add_argument("--target-fpr", type=float, default=0.05)
    args = parser.parse_args()
    print(json.dumps(run_experiment(args.version, args.seed, args.threshold_method, args.target_fpr), indent=2))


if __name__ == "__main__":
    main()
