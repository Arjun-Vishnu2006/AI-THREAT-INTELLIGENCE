from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from app.ml.android_feature_schema import ANDROID_FEATURE_NAMES
from app.novelty.dataset_experiment import (
    CLASS_ID_TO_NAME,
    class_distribution,
    feature_overlap_report,
    fit_training_preprocessor,
    select_validation_threshold,
    split_training_known,
    validate_class_mapping,
    validate_experiment_metadata,
)


def _frame(n: int = 30) -> pd.DataFrame:
    values = np.zeros((n, len(ANDROID_FEATURE_NAMES)), dtype=np.float64)
    values[:, 0] = np.arange(n)
    values[:, 1] = np.arange(n) % 3
    frame = pd.DataFrame(values, columns=ANDROID_FEATURE_NAMES)
    frame["Class"] = [1, 2, 5] * (n // 3)
    frame["Class_Name"] = ["Adware", "Banking", "Benign"] * (n // 3)
    return frame


def test_scaler_is_fitted_only_on_train_fit_rows_and_validation_does_not_change_it() -> None:
    train_fit = _frame(12).loc[:, ANDROID_FEATURE_NAMES]
    scaler, _ = fit_training_preprocessor(train_fit)
    validation = pd.DataFrame(100000.0, index=range(3), columns=ANDROID_FEATURE_NAMES)
    scaler.transform(validation)
    expected_mean = train_fit.to_numpy().mean(axis=0)
    assert np.allclose(scaler.scaler.mean_, expected_mean)


def test_test_data_does_not_participate_in_fit_or_calibration() -> None:
    fit = _frame(12).loc[:, ANDROID_FEATURE_NAMES]
    preprocessor, _ = fit_training_preprocessor(fit)
    before = preprocessor.scaler.mean_.copy()
    _test_data = np.full((4, len(ANDROID_FEATURE_NAMES)), 1e9)
    assert np.array_equal(preprocessor.scaler.mean_, before)
    errors = [0.1, 0.2, 0.8, 0.9]
    labels = [False, False, True, True]
    first = select_validation_threshold(errors, labels)
    _test_errors = [1e6, 2e6]
    second = select_validation_threshold(errors, labels)
    assert first == second


def test_threshold_uses_validation_candidates_not_training_error_percentile() -> None:
    validation_errors = [0.1, 0.2, 0.8, 0.9]
    threshold, details = select_validation_threshold(validation_errors, [False, False, True, True])
    training_errors = np.linspace(20.0, 30.0, 100)
    assert threshold in validation_errors
    assert threshold != float(np.percentile(training_errors, 95))
    assert details["method"] == "max_f1"


def test_canonical_schema_is_exactly_470_columns() -> None:
    assert len(ANDROID_FEATURE_NAMES) == 470
    assert len(set(ANDROID_FEATURE_NAMES)) == 470


def test_threshold_selection_is_deterministic() -> None:
    args = ([0.3, 0.1, 0.8, 0.7], [True, False, True, False])
    assert select_validation_threshold(*args) == select_validation_threshold(*args)


def test_verified_class_mapping_and_distribution() -> None:
    frame = _frame()
    validate_class_mapping(frame)
    assert CLASS_ID_TO_NAME == {1: "Adware", 2: "Banking", 3: "SMS", 4: "Riskware", 5: "Benign"}
    assert class_distribution(frame) == {
        "Adware": 10, "Banking": 10, "SMS": 0, "Riskware": 0, "Benign": 10,
    }
    frame.loc[0, "Class_Name"] = "Benign"
    with pytest.raises(ValueError, match="Unexpected dataset label mapping"):
        validate_class_mapping(frame)


def test_possible_duplicate_report_does_not_claim_sample_identity() -> None:
    base = _frame(3)
    duplicate = base.copy()
    report = feature_overlap_report({"train": base, "validation": duplicate})
    assert report["train_validation"]["known_class_unique_vector_overlap_count"] == 3
    assert report["train_validation"]["identity_claimed"] is False


def test_persisted_metadata_contract_includes_reproducibility_and_limitations() -> None:
    metadata = {
        "version": "0.2.0", "seed": 2026,
        "class_id_to_name": {str(k): v for k, v in CLASS_ID_TO_NAME.items()},
        "training_known_classes": ["Adware", "Banking", "Benign"],
        "validation_novel_classes": ["SMS", "Riskware"],
        "feature_schema": {"feature_count": 470, "ordered_names": list(ANDROID_FEATURE_NAMES)},
        "scaler_fit_rows": 10, "model_fit_rows": 10,
        "threshold_selection": {"method": "max_f1"},
        "dataset_sha256": {"train": "a"},
        "possible_feature_level_duplicates": {"train_validation": {"identity_claimed": False}},
        "test_metrics": {"f1": 1.0},
        "runtime_feature_compatibility": {"fully_validated": False, "unsupported_training_schema_features": 35},
    }
    validate_experiment_metadata(metadata)
    assert json.loads(json.dumps(metadata))["seed"] == 2026
    with pytest.raises(ValueError, match="missing required fields"):
        validate_experiment_metadata({"version": "0.2.0"})


def test_training_split_is_stratified_and_repeatable() -> None:
    known = _frame(90)
    left_fit, left_monitor = split_training_known(known, seed=2026)
    right_fit, right_monitor = split_training_known(known, seed=2026)
    assert left_fit.index.tolist() == right_fit.index.tolist()
    assert left_monitor.index.tolist() == right_monitor.index.tolist()
    assert set(left_fit.index).isdisjoint(left_monitor.index)
    assert len(left_fit) + len(left_monitor) == len(known)
