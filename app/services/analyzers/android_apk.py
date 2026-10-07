"""Static Android APK permission and DEX method-reference extraction."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.ml.android_feature_schema import ANDROID_FEATURE_NAMES

_BEHAVIOR_FEATURE_COUNT = 35
_API_FEATURE_NAMES = ANDROID_FEATURE_NAMES[_BEHAVIOR_FEATURE_COUNT:]


class APKFeatureExtractionError(RuntimeError):
    """Raised when an APK cannot be parsed for manifest and DEX features."""


@dataclass(frozen=True)
class APKFeatureExtraction:
    status: str
    permissions: tuple[str, ...]
    api_method_names: tuple[str, ...]
    feature_values: dict[str, int | None]
    feature_status: dict[str, str]
    limitations: tuple[str, ...]
    error: str | None = None

    @classmethod
    def unavailable(cls, reason: str) -> "APKFeatureExtraction":
        return cls(
            status="unavailable",
            permissions=(),
            api_method_names=(),
            feature_values={name: None for name in ANDROID_FEATURE_NAMES},
            feature_status={name: "unavailable" for name in ANDROID_FEATURE_NAMES},
            limitations=("APK manifest and DEX features were unavailable; no values were inferred.",),
            error=reason,
        )

    @property
    def autoencoder_vector_available(self) -> bool:
        return all(value is not None for value in self.feature_values.values())

    def to_dict(self) -> dict[str, Any]:
        counts = {
            state: sum(value == state for value in self.feature_status.values())
            for state in ("present", "absent", "unavailable")
        }
        return {
            "status": self.status,
            "permissions": list(self.permissions),
            "api_method_names": list(self.api_method_names),
            "feature_values": self.feature_values,
            "feature_status": self.feature_status,
            "feature_status_counts": counts,
            "autoencoder_vector_available": self.autoencoder_vector_available,
            "limitations": list(self.limitations),
            "error": self.error,
        }


def _androguard_parser(source: bytes | bytearray | str | Path) -> tuple[Any, Any, Any]:
    try:
        from androguard.misc import AnalyzeAPK
    except ImportError as exc:
        raise APKFeatureExtractionError(
            "APK analysis requires the 'androguard' dependency; install project requirements"
        ) from exc
    if isinstance(source, (bytes, bytearray)):
        return AnalyzeAPK(bytes(source), raw=True)
    return AnalyzeAPK(str(source))


class AndroidAPKFeatureExtractor:
    """Extract declared permissions and exact-name external DEX method references.

    The schema's first 35 behavior-category features have no documented static
    derivation in this repository and remain unavailable. The remaining names
    are checked against external DEX method references found in the APK.
    """

    def __init__(self, parser: Callable[[bytes | bytearray | str | Path], tuple[Any, Any, Any]] | None = None):
        self.parser = parser or _androguard_parser

    def extract(self, source: bytes | bytearray | str | Path) -> APKFeatureExtraction:
        try:
            apk, dex_files, analysis = self.parser(source)
            is_valid_apk = getattr(apk, "is_valid_APK", None)
            if callable(is_valid_apk) and not is_valid_apk():
                raise APKFeatureExtractionError(
                    "Androguard could not parse the APK AndroidManifest.xml"
                )
            permissions = tuple(sorted(set(str(item) for item in apk.get_permissions())))
            api_available = bool(dex_files)
            method_names: set[str] = set()
            if api_available:
                for method_analysis in analysis.get_external_methods():
                    method = method_analysis.get_method()
                    method_names.add(str(method.get_name()))
        except APKFeatureExtractionError:
            raise
        except Exception as exc:
            raise APKFeatureExtractionError(
                f"APK manifest/DEX parsing failed: {type(exc).__name__}: {exc}"
            ) from exc

        feature_values: dict[str, int | None] = {}
        feature_status: dict[str, str] = {}
        for name in ANDROID_FEATURE_NAMES:
            if name in _API_FEATURE_NAMES and api_available:
                present = name in method_names
                feature_values[name] = int(present)
                feature_status[name] = "present" if present else "absent"
            else:
                feature_values[name] = None
                feature_status[name] = "unavailable"

        limitations = [
            "Presence means a matching external DEX method reference occurs in parsed bytecode; it does not prove runtime execution.",
            "Schema features are matched by method name because the trained schema contains no declaring-class or method-signature data; same-name references can collide across classes.",
            "The first 35 behavior-category features have no supported APK-to-feature mapping in this repository.",
        ]
        if not api_available:
            limitations.append("No DEX files were available; API-call feature states are unavailable, not absent.")
        return APKFeatureExtraction(
            status="partial",
            permissions=permissions,
            api_method_names=tuple(sorted(method_names & set(_API_FEATURE_NAMES))),
            feature_values=feature_values,
            feature_status=feature_status,
            limitations=tuple(limitations),
        )
