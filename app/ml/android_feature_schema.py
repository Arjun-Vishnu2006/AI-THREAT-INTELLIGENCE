"""Canonical schema and validation for the trained Android autoencoder."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

FEATURE_COUNT = 470
ANDROID_FEATURE_KEY = "android.permission_api_features"
_SCHEMA_PATH = Path(__file__).resolve().parents[1] / "novelty" / "models" / "feature_columns.json"


class FeatureSchemaError(ValueError):
    """Raised when Android features do not match the persisted model contract."""


def load_android_feature_schema() -> tuple[str, ...]:
    """Load the exact ordered feature names used by the saved scaler and model."""
    with _SCHEMA_PATH.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    if isinstance(data, list):
        names = tuple(str(item) for item in data)
    elif isinstance(data, dict):
        try:
            names = tuple(str(data[str(index)]) for index in range(len(data)))
        except KeyError as exc:
            raise FeatureSchemaError("Persisted Android feature schema has missing indexes") from exc
    else:
        raise FeatureSchemaError("Persisted Android feature schema must be a list or indexed object")
    if len(names) != FEATURE_COUNT or len(set(names)) != FEATURE_COUNT:
        raise FeatureSchemaError(
            f"Persisted Android feature schema must contain {FEATURE_COUNT} unique names; got {len(names)}"
        )
    return names


ANDROID_FEATURE_NAMES = load_android_feature_schema()


def validate_android_features(features: Mapping[str, Any]) -> np.ndarray:
    """Validate a named feature row and return float32 values in model order."""
    if not isinstance(features, Mapping):
        shape = getattr(features, "shape", None)
        if shape is not None and len(shape) == 1 and shape[0] != FEATURE_COUNT:
            raise FeatureSchemaError(
                f"Android feature dimension must be {FEATURE_COUNT}; got {shape[0]}"
            )
        raise FeatureSchemaError("Android autoencoder input must be a named feature mapping")
    names = set(features)
    required = set(ANDROID_FEATURE_NAMES)
    missing = [name for name in ANDROID_FEATURE_NAMES if name not in names]
    extra = sorted(str(name) for name in names - required)
    if missing:
        preview = ", ".join(missing[:5])
        raise FeatureSchemaError(f"Missing {len(missing)} required Android features: {preview}")
    if extra:
        raise FeatureSchemaError(f"Unexpected Android features ({len(extra)}): {', '.join(extra[:5])}")
    if len(features) != FEATURE_COUNT:
        raise FeatureSchemaError(
            f"Android feature dimension must be {FEATURE_COUNT}; got {len(features)}"
        )
    values: list[float] = []
    for name in ANDROID_FEATURE_NAMES:
        value = features[name]
        if not isinstance(value, (int, float, np.number)):
            raise FeatureSchemaError(f"Android feature {name!r} must be numeric")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise FeatureSchemaError(f"Android feature {name!r} must be finite")
        values.append(numeric)
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (FEATURE_COUNT,) or not np.isfinite(result).all():
        raise FeatureSchemaError("Android feature row cannot be represented as a finite 470-value vector")
    return result


def extract_android_feature_row(features: Mapping[str, Any]) -> dict[str, float]:
    """Return a validated, deterministically ordered row for a NoveltySnapshot."""
    values = validate_android_features(features)
    return dict(zip(ANDROID_FEATURE_NAMES, map(float, values)))
