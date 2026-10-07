from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.ml.android_feature_schema import ANDROID_FEATURE_NAMES, FeatureSchemaError
from app.novelty.ae_layers.autoencoder_novelty_layer import AutoencoderNoveltyLayer
from app.novelty.engine import NoveltyEngine
from app.novelty.types import NoveltySnapshot


class _Tensor:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values

    def numpy(self) -> np.ndarray:
        return self.values


class _IdentityScaler:
    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        return frame.to_numpy(dtype=np.float32)


class _ZeroModel:
    def __call__(self, values: np.ndarray, training: bool = False) -> _Tensor:
        assert training is False
        return _Tensor(np.zeros_like(values))


def _snapshot(score: float | None) -> NoveltySnapshot:
    vector = None
    if score is not None:
        vector = dict.fromkeys(ANDROID_FEATURE_NAMES, 0.0)
        vector[ANDROID_FEATURE_NAMES[0]] = float(np.sqrt(score * len(ANDROID_FEATURE_NAMES)))
    return NoveltySnapshot((), (), "", {}, (), (), android_feature_vector=vector)


def _engine() -> NoveltyEngine:
    # Bypass filesystem model loading; inference uses deterministic test doubles.
    layer = object.__new__(AutoencoderNoveltyLayer)
    layer.model = _ZeroModel()
    layer.scaler = _IdentityScaler()
    layer.feature_names = list(ANDROID_FEATURE_NAMES)
    layer.threshold = 0.5
    layer.baseline_scaled = np.empty((0, len(ANDROID_FEATURE_NAMES)), dtype=np.float32)
    return NoveltyEngine(layers={"autoencoder": layer})


def test_engine_calibrates_autoencoder_from_actual_matching_schema_scores() -> None:
    engine = _engine()
    calibration = [_snapshot(float(score)) for score in range(20)]
    engine.fit([_snapshot(None) for _ in range(10)], calibration)
    assert engine.calibration_scores["autoencoder"] == pytest.approx(
        sorted(float(score) for score in range(20)), abs=1e-5
    )
    assert engine.calibration_scores["autoencoder"] != [0.5] * 20


def test_engine_fails_clearly_without_matching_schema_calibration_samples() -> None:
    engine = _engine()
    calibration = [_snapshot(None) for _ in range(25)]
    with pytest.raises(ValueError, match="fewer than 20 matching-schema calibration observations"):
        engine.fit([_snapshot(None) for _ in range(10)], calibration)


def test_engine_requires_twenty_scored_android_calibration_samples() -> None:
    engine = _engine()
    calibration = [_snapshot(float(index)) for index in range(19)] + [_snapshot(None)]
    with pytest.raises(ValueError, match="got 19"):
        engine.fit([_snapshot(None) for _ in range(10)], calibration)


def test_engine_rejects_malformed_android_calibration_schema() -> None:
    engine = _engine()
    malformed = dict.fromkeys(ANDROID_FEATURE_NAMES, 0.0)
    malformed.pop(ANDROID_FEATURE_NAMES[0])
    invalid = NoveltySnapshot((), (), "", {}, (), (), android_feature_vector=malformed)
    calibration = [_snapshot(float(index)) for index in range(19)] + [invalid]
    with pytest.raises(FeatureSchemaError, match="Missing 1 required Android features"):
        engine.fit([_snapshot(None) for _ in range(10)], calibration)
