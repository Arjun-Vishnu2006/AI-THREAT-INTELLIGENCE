"""Compatibility exports for the shared Android autoencoder feature schema."""

from app.ml.android_feature_schema import (
    ANDROID_FEATURE_KEY,
    ANDROID_FEATURE_NAMES,
    FEATURE_COUNT,
    FeatureSchemaError,
    extract_android_feature_row,
    load_android_feature_schema,
    validate_android_features,
)

__all__ = [
    "ANDROID_FEATURE_KEY",
    "ANDROID_FEATURE_NAMES",
    "FEATURE_COUNT",
    "FeatureSchemaError",
    "extract_android_feature_row",
    "load_android_feature_schema",
    "validate_android_features",
]
