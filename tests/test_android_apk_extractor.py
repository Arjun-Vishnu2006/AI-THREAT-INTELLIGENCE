from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from app.ml.android_feature_schema import ANDROID_FEATURE_NAMES
from app.novelty.types import snapshot_from_analysis
from app.services.analyzers.android_apk import (
    APKFeatureExtractionError,
    AndroidAPKFeatureExtractor,
)
from app.services.analyzers.file import FileAnalyzer
from app.services.correlation import CorrelationResult
from app.services.threat_graph import ThreatGraph


class _Method:
    def __init__(self, name: str) -> None:
        self.name = name

    def get_name(self) -> str:
        return self.name


class _MethodAnalysis:
    def __init__(self, name: str) -> None:
        self.method = _Method(name)

    def get_method(self) -> _Method:
        return self.method


class _APKFixture:
    def is_valid_APK(self) -> bool:
        return True

    def get_permissions(self) -> list[str]:
        return ["android.permission.INTERNET", "android.permission.CAMERA"]


class _AnalysisFixture:
    def get_external_methods(self) -> list[_MethodAnalysis]:
        return [_MethodAnalysis("write"), _MethodAnalysis("accept"), _MethodAnalysis("write")]


def _parser(source: object) -> tuple[_APKFixture, list[object], _AnalysisFixture]:
    del source
    return _APKFixture(), [object()], _AnalysisFixture()


def _apk_bytes() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("AndroidManifest.xml", b"fixture manifest")
        archive.writestr("classes.dex", b"fixture dex")
    return buffer.getvalue()


def test_extractor_maps_only_observed_canonical_dex_method_names() -> None:
    result = AndroidAPKFeatureExtractor(parser=_parser).extract(_apk_bytes())
    assert result.status == "partial"
    assert result.permissions == (
        "android.permission.CAMERA",
        "android.permission.INTERNET",
    )
    assert result.api_method_names == ("accept", "write")
    assert result.feature_values["accept"] == 1
    assert result.feature_status["accept"] == "present"
    assert result.feature_values["write"] == 1
    assert result.feature_values["getLastLocation"] == 0
    assert result.feature_status["getLastLocation"] == "absent"
    assert all(result.feature_values[name] is None for name in ANDROID_FEATURE_NAMES[:35])
    assert all(result.feature_status[name] == "unavailable" for name in ANDROID_FEATURE_NAMES[:35])
    assert not result.autoencoder_vector_available
    assert len(result.feature_values) == len(result.feature_status) == 470


def test_no_dex_marks_api_features_unavailable_not_absent() -> None:
    def parser(source: object) -> tuple[_APKFixture, list[object], _AnalysisFixture]:
        del source
        return _APKFixture(), [], _AnalysisFixture()

    result = AndroidAPKFeatureExtractor(parser=parser).extract(_apk_bytes())
    assert result.permissions
    assert all(result.feature_values[name] is None for name in ANDROID_FEATURE_NAMES[35:])
    assert all(result.feature_status[name] == "unavailable" for name in ANDROID_FEATURE_NAMES[35:])
    assert any("No DEX files" in limitation for limitation in result.limitations)


def test_parser_failures_are_reported_without_inventing_feature_values() -> None:
    def parser(source: object) -> tuple[object, object, object]:
        del source
        raise ValueError("invalid APK fixture")

    with pytest.raises(APKFeatureExtractionError, match="invalid APK fixture"):
        AndroidAPKFeatureExtractor(parser=parser).extract(_apk_bytes())


def test_file_analyzer_integrates_apk_results_into_snapshot() -> None:
    bundle = FileAnalyzer(
        android_feature_extractor=AndroidAPKFeatureExtractor(parser=_parser)
    ).analyze({"name": "fixture.apk", "content": _apk_bytes()})
    extraction = bundle.features["android.apk_analysis"]
    assert extraction["feature_status_counts"]["unavailable"] == 35
    assert extraction["feature_status_counts"]["present"] == 2
    assert extraction["feature_status_counts"]["absent"] == 433

    correlation = CorrelationResult(entities=(), relationships=())
    snapshot = snapshot_from_analysis(
        [bundle], correlation, ThreatGraph.from_correlation(correlation)
    )
    assert snapshot.android_feature_vector is None
    assert snapshot.android_feature_extraction == extraction
    assert snapshot.evidence_sources["autoencoder_feature_status"] == (
        "unavailable_incomplete_feature_schema"
    )


def test_file_analyzer_reports_missing_androguard_clearly() -> None:
    extractor = AndroidAPKFeatureExtractor(parser=lambda source: (_ for _ in ()).throw(
        APKFeatureExtractionError("Androguard parser is unavailable")
    ))
    bundle = FileAnalyzer(android_feature_extractor=extractor).analyze(
        {"name": "fixture.apk", "content": _apk_bytes()}
    )
    result = bundle.features["android.apk_analysis"]
    assert result["status"] == "unavailable"
    assert "Androguard parser is unavailable" in result["error"]
    assert set(result["feature_status"].values()) == {"unavailable"}


def test_extractor_accepts_file_paths() -> None:
    apk_path = Path("sample.apk")
    result = AndroidAPKFeatureExtractor(parser=_parser).extract(apk_path)
    assert result.status == "partial"
    assert len(result.feature_values) == 470
