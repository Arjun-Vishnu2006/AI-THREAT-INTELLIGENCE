from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from app.ml.android_feature_schema import ANDROID_FEATURE_NAMES
from app.novelty.benign_experiment import (
    BenignRobustPreprocessor,
    huber_reconstruction_errors,
    split_benign_train_rows,
)
from app.novelty.dataset_experiment import select_validation_threshold


def _benign_train_fit() -> pd.DataFrame:
    train = pd.read_csv("data/train.csv")
    fit, _ = split_benign_train_rows(train, seed=2026)
    return fit.loc[:, list(ANDROID_FEATURE_NAMES)]


def test_fitting_population_is_benign_only_and_split_is_deterministic() -> None:
    train = pd.read_csv("data/train.csv")
    fit_a, monitor_a = split_benign_train_rows(train, seed=2026)
    fit_b, monitor_b = split_benign_train_rows(train, seed=2026)
    assert set(fit_a["Class_Name"]) == {"Benign"}
    assert set(fit_a["Class"]) == {5}
    assert set(monitor_a["Class_Name"]) == {"Benign"}
    assert fit_a.index.tolist() == fit_b.index.tolist()
    assert monitor_a.index.tolist() == monitor_b.index.tolist()
    assert set(fit_a.index).isdisjoint(monitor_a.index)


def test_preprocessing_parameters_come_only_from_benign_train_fit_rows() -> None:
    fit = _benign_train_fit()
    preprocessor = BenignRobustPreprocessor().fit(fit)
    values = fit.to_numpy(dtype=np.float64)
    binary = preprocessor.binary_mask
    expected = np.percentile(np.log1p(values[:, ~binary]), [25, 50, 75], axis=0)
    observed_median = preprocessor.medians[~binary]
    observed_iqr = preprocessor.iqrs[~binary]
    expected_iqr = expected[2] - expected[0]
    expected_iqr[expected_iqr <= np.finfo(float).eps] = 1.0
    assert np.allclose(observed_median, expected[1])
    assert np.allclose(observed_iqr, expected_iqr)
    assert preprocessor.to_dict()["fit_population"] == "All Benign rows from train.csv only; no validation or test rows"


def test_transformation_is_deterministic_and_preserves_fit_inferred_binary_values() -> None:
    fit = _benign_train_fit()
    first = BenignRobustPreprocessor().fit(fit)
    second = BenignRobustPreprocessor().fit(fit.copy())
    transformed = first.transform(fit)
    assert np.array_equal(transformed, second.transform(fit))
    binary_indexes = np.flatnonzero(first.binary_mask)
    assert len(binary_indexes) > 0
    assert np.array_equal(transformed[:, binary_indexes], fit.to_numpy()[:, binary_indexes])


def test_count_transform_clips_extreme_values_without_changing_fitted_parameters() -> None:
    fit = _benign_train_fit()
    preprocessor = BenignRobustPreprocessor().fit(fit)
    count_index = int(np.flatnonzero(~preprocessor.binary_mask)[0])
    before_medians = preprocessor.medians.copy()
    before_iqrs = preprocessor.iqrs.copy()
    row = fit.iloc[[0]].copy()
    row.iloc[0, count_index] = 1e12
    transformed = preprocessor.transform(row)
    assert transformed[0, count_index] == 5.0
    assert np.array_equal(preprocessor.medians, before_medians)
    assert np.array_equal(preprocessor.iqrs, before_iqrs)


def test_threshold_helper_has_validation_only_inputs_and_uses_validation_candidates() -> None:
    signature = inspect.signature(select_validation_threshold)
    assert set(signature.parameters) >= {"validation_errors", "validation_is_novel"}
    threshold, details = select_validation_threshold(
        [0.1, 0.2, 0.8, 0.9], [False, False, True, True], method="max_f1"
    )
    assert threshold in {0.1, 0.2, 0.8, 0.9}
    assert details["method"] == "max_f1"


def test_exact_470_column_schema_is_required() -> None:
    fit = _benign_train_fit()
    assert tuple(fit.columns) == ANDROID_FEATURE_NAMES
    assert len(ANDROID_FEATURE_NAMES) == 470
    preprocessor = BenignRobustPreprocessor().fit(fit)
    with pytest.raises(ValueError, match="exact ordered 470-feature schema"):
        preprocessor.transform(fit.iloc[:, :-1])


def test_huber_reconstruction_score_limits_single_extreme_feature() -> None:
    actual = np.zeros((1, 470), dtype=np.float64)
    prediction = np.zeros_like(actual)
    prediction[0, 0] = 1000.0
    huber = huber_reconstruction_errors(actual, prediction, delta=1.0)[0]
    mse = np.mean((actual - prediction) ** 2)
    assert huber == pytest.approx(999.5 / 470)
    assert huber < mse / 1000
