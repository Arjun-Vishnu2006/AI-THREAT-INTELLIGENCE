from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path

import numpy as np
import pytest

from app.novelty.feature_schema import (
    ANDROID_FEATURE_KEY,
    ANDROID_FEATURE_NAMES,
    FEATURE_COUNT,
    FeatureSchemaError,
    extract_android_feature_row,
    validate_android_features,
)
from app.novelty.types import snapshot_from_analysis
from app.services.analyzers.file import FileAnalyzer
from app.services.correlation import CorrelationResult
from app.services.feature_types import FeatureBundle
from app.services.threat_graph import ThreatGraph


def _row() -> dict[str, float]:
    return {name: float(index % 2) for index, name in enumerate(ANDROID_FEATURE_NAMES)}


def test_canonical_schema_matches_training_csv_and_valid_row() -> None:
    with Path("data/train.csv").open(encoding="utf-8-sig", newline="") as fp:
        columns = next(csv.reader(fp))
    assert tuple(columns[:-2]) == ANDROID_FEATURE_NAMES
    assert len(ANDROID_FEATURE_NAMES) == FEATURE_COUNT == 470
    vector = validate_android_features(_row())
    assert vector.shape == (FEATURE_COUNT,)
    assert np.isfinite(vector).all()


def test_reordered_columns_are_returned_in_canonical_order() -> None:
    row = _row()
    reordered = dict(reversed(list(row.items())))
    assert np.array_equal(validate_android_features(reordered), validate_android_features(row))
    assert tuple(extract_android_feature_row(reordered)) == ANDROID_FEATURE_NAMES


def test_missing_columns_are_rejected() -> None:
    row = _row()
    row.pop(ANDROID_FEATURE_NAMES[0])
    with pytest.raises(FeatureSchemaError, match="Missing 1 required"):
        validate_android_features(row)


def test_extra_columns_are_rejected_explicitly() -> None:
    row = _row()
    row["unexpected.permission"] = 1.0
    with pytest.raises(FeatureSchemaError, match="Unexpected Android features"):
        validate_android_features(row)


def test_wrong_feature_dimension_is_rejected() -> None:
    with pytest.raises(FeatureSchemaError, match="dimension must be 470; got 469"):
        validate_android_features(np.zeros(FEATURE_COUNT - 1, dtype=np.float32))


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_values_are_rejected(bad_value: float) -> None:
    row = _row()
    row[ANDROID_FEATURE_NAMES[0]] = bad_value
    with pytest.raises(FeatureSchemaError, match="must be finite"):
        validate_android_features(row)


def test_snapshot_conversion_preserves_a_named_android_vector() -> None:
    bundle = FeatureBundle(
        analyzer="file",
        artifact_kind="apk",
        features={ANDROID_FEATURE_KEY: _row()},
    )
    correlation = CorrelationResult(entities=(), relationships=())
    snapshot = snapshot_from_analysis(
        [bundle], correlation, ThreatGraph.from_correlation(correlation)
    )
    assert snapshot.android_feature_vector is not None
    assert tuple(snapshot.android_feature_vector) == ANDROID_FEATURE_NAMES
    assert snapshot.evidence_sources["autoencoder_feature_status"] == "available"


def test_runtime_file_analyzer_does_not_fabricate_android_features() -> None:
    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, "w") as archive:
        archive.writestr("AndroidManifest.xml", "placeholder")
    bundle = FileAnalyzer().analyze({"name": "sample.apk", "content": archive_buffer.getvalue()})
    assert ANDROID_FEATURE_KEY not in bundle.features
    correlation = CorrelationResult(entities=(), relationships=())
    snapshot = snapshot_from_analysis(
        [bundle], correlation, ThreatGraph.from_correlation(correlation)
    )
    assert snapshot.android_feature_vector is None
    assert snapshot.evidence_sources["autoencoder_feature_status"] == "unavailable_apk_analysis"
    assert "could not parse" in bundle.features["android.apk_analysis"]["error"]
