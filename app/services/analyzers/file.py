from __future__ import annotations

import re
import zipfile
from io import BytesIO
from typing import Any

from app.services.analyzers.common import (
    EMAIL_RE, URL_RE, artifact_id, evidence, indicator, normalize_url, safe_bytes, sha256,
)
from app.services.file_malware_ml import predict_pe_malware
from app.services.contracts import AnalysisContext
from app.services.feature_types import FeatureBundle
from app.services.analyzers.android_apk import (
    APKFeatureExtraction,
    APKFeatureExtractionError,
    AndroidAPKFeatureExtractor,
)

_TEXT_SUFFIXES = {".txt", ".html", ".htm", ".xml", ".json", ".js", ".vbs", ".ps1", ".bat", ".cmd", ".py", ".hta", ".svg", ".url", ".eml", ".csv"}
_SCRIPT_SUFFIXES = {".js", ".vbs", ".ps1", ".bat", ".cmd", ".py", ".hta", ".svg"}
_URL_RE = URL_RE
_MAX_ZIP_ENTRIES = 2000
_MAX_ZIP_TOTAL_UNCOMPRESSED = 100 * 1024 * 1024
_MAX_ZIP_ENTRY_UNCOMPRESSED = 20 * 1024 * 1024


def _suffix(name: str) -> str:
    return "." + name.rsplit(".", 1)[-1].casefold() if "." in name.rsplit("/", 1)[-1] else ""


def _magic(data: bytes) -> tuple[str, str]:
    if data.startswith(b"MZ"):
        return "application/vnd.microsoft.portable-executable", "pe"
    if data.startswith(b"\x7fELF"):
        return "application/x-elf", "elf"
    if data.startswith(b"%PDF-"):
        return "application/pdf", "pdf"
    if data.startswith(b"PK\x03\x04"):
        return "application/zip", "zip"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png", "png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg", "jpeg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif", "gif"
    if data.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"):
        return "application/x-ole-storage", "ole"
    return "application/octet-stream", "unknown"


class FileAnalyzer:
    name = "file"
    supported_kinds = frozenset({"file"})

    def __init__(self, android_feature_extractor: AndroidAPKFeatureExtractor | None = None) -> None:
        self.android_feature_extractor = android_feature_extractor or AndroidAPKFeatureExtractor()

    def analyze(self, artifact: dict[str, Any], context: AnalysisContext | None = None) -> FeatureBundle:
        raw = next((artifact[key] for key in ("content", "file_bytes") if key in artifact and artifact[key] is not None), None)
        if raw is None:
            raise ValueError("File artifact requires content bytes")
        data = safe_bytes(raw)
        filename = str(artifact.get("name") or artifact.get("filename") or "")
        suffix = _suffix(filename)
        mime, detected_type = _magic(data)
        digest = sha256(data)
        matches = list(dict.fromkeys(m.group(0).rstrip(".,);]>") for m in _URL_RE.finditer(data.decode("latin-1", errors="ignore"))))
        emails = list(dict.fromkeys(EMAIL_RE.findall(data.decode("latin-1", errors="ignore"))))
        features: dict[str, Any] = {
            "file.name": filename or None,
            "file.suffix": suffix,
            "file.size_bytes": len(data),
            "file.sha256": digest,
            "file.detected_mime_type": mime,
            "file.detected_type": detected_type,
            "file.declared_media_type": artifact.get("media_type"),
            "file.declared_mime_mismatch": bool(artifact.get("media_type") and artifact["media_type"] != mime),
            "file.is_empty": len(data) == 0,
            "file.contains_urls": matches,
            "file.contains_email_addresses": emails,
            "file.content_executed": False,
        }
        observations = []
        is_apk = suffix == ".apk"

        # EMBER ML assessment: Windows PE files only; never execute the file.
        if detected_type == "pe":
            ml_result = predict_pe_malware(data)
            features["file.ml_status"] = ml_result.get("status")

            if ml_result.get("status") == "available":
                features["file.ml_prediction"] = ml_result["prediction"]
                features["file.ml_malware_probability"] = ml_result[
                    "malware_probability"
                ]
                observations.append(evidence(
                    "file_ml_prediction",
                    "EMBER 2018 static PE classifier produced a malware assessment.",
                    attributes={
                        "prediction": ml_result["prediction"],
                        "malware_probability": ml_result["malware_probability"],
                        "feature_dimensions": ml_result["feature_dimensions"],
                        "model": ml_result["model"],
                    },
                ))
            else:
                features["file.ml_reason"] = ml_result.get("reason")
                if ml_result.get("error"):
                    features["file.ml_error"] = ml_result["error"]
                observations.append(evidence(
                    "file_ml_unavailable",
                    "File-malware ML assessment could not be completed; existing static checks remain available.",
                    attributes={
                        "reason": ml_result.get("reason"),
                        "error": ml_result.get("error"),
                    },
                ))
        if suffix in _SCRIPT_SUFFIXES or detected_type in {"pe", "elf", "ole"}:
            observations.append(evidence("file_type", "File is a script, executable, or OLE container by extension or signature.", attributes={"extension": suffix, "detected_type": detected_type}))
        if features["file.declared_mime_mismatch"]:
            observations.append(evidence("file_type", "Declared media type does not match the detected file signature.", attributes={"declared": artifact.get("media_type"), "detected": mime}))
        if matches:
            observations.append(evidence("file_content", "Static byte-string scan found URL-like strings.", attributes={"count": len(matches)}))

        if detected_type == "zip":
            try:
                with zipfile.ZipFile(BytesIO(data)) as archive:
                    infos = archive.infolist()
                    total = sum(item.file_size for item in infos)
                    features["file.archive_entry_count"] = len(infos)
                    features["file.archive_uncompressed_bytes"] = total
                    if len(infos) > _MAX_ZIP_ENTRIES or total > _MAX_ZIP_TOTAL_UNCOMPRESSED or any(i.file_size > _MAX_ZIP_ENTRY_UNCOMPRESSED for i in infos):
                        features["file.archive_inspection_limited"] = True
                        observations.append(evidence("file_archive", "Archive listing exceeded static inspection limits; entries were not extracted."))
                    else:
                        names = [i.filename for i in infos]
                        is_apk = is_apk or "AndroidManifest.xml" in names
                        features["file.archive_entries"] = names[:_MAX_ZIP_ENTRIES]
                        features["file.archive_contains_vba_project"] = any(n.casefold().endswith("vbaproject.bin") for n in names)
                        features["file.archive_contains_macro_enabled_type"] = any(n.casefold().endswith((".xlsm", ".docm", ".pptm")) for n in names)
                        features["file.archive_path_traversal_names"] = [n for n in names if n.startswith(("/", "\\")) or "../" in n.replace("\\", "/")]
                        if features["file.archive_contains_vba_project"] or features["file.archive_contains_macro_enabled_type"]:
                            observations.append(evidence("file_archive", "Office archive contains macro-related names; macro code was not executed or extracted."))
            except (zipfile.BadZipFile, OSError, RuntimeError):
                features["file.archive_parse_error"] = True
                observations.append(evidence("file_archive", "ZIP signature was present but archive metadata could not be read."))
        elif suffix in _TEXT_SUFFIXES and len(data) <= 2 * 1024 * 1024:
            text = data.decode("utf-8", errors="replace")
            features["file.text_character_count"] = len(text)
            features["file.text_line_count"] = len(text.splitlines())
            features["file.script_like_extension"] = suffix in _SCRIPT_SUFFIXES
        elif detected_type == "pdf":
            pdf_text = data.decode("latin-1", errors="ignore")
            features["file.pdf_page_marker_count"] = len(re.findall(r"/Type\s*/Page\b", pdf_text))
            features["file.pdf_javascript_marker"] = bool(re.search(r"/JavaScript\b|/JS\b", pdf_text))
            features["file.pdf_embedded_file_marker"] = bool(re.search(r"/EmbeddedFile\b|/Filespec\b", pdf_text))
            if features["file.pdf_javascript_marker"] or features["file.pdf_embedded_file_marker"]:
                observations.append(evidence("file_pdf", "PDF contains JavaScript or embedded-file structural markers."))

        limitations = [
            "Static metadata and bounded signature/string inspection only; file was never executed or extracted.",
            "PDF analysis uses structural markers and does not render or fully parse the document.",
        ]
        if is_apk:
            try:
                apk_features = self.android_feature_extractor.extract(data)
            except APKFeatureExtractionError as exc:
                apk_features = APKFeatureExtraction.unavailable(str(exc))
                observations.append(evidence(
                    "android_apk_analysis",
                    "APK feature extraction was unavailable.",
                    attributes={"reason": str(exc)},
                ))
            else:
                observations.append(evidence(
                    "android_apk_analysis",
                    "APK manifest permissions and statically referenced DEX methods were extracted.",
                    attributes={
                        "permission_count": len(apk_features.permissions),
                        "api_method_count": len(apk_features.api_method_names),
                        "feature_status_counts": apk_features.to_dict()["feature_status_counts"],
                    },
                ))
            features["android.apk_analysis"] = apk_features.to_dict()
            limitations.extend(apk_features.limitations)

        indicators = [indicator("file_hash_sha256", digest, normalized=digest)]
        indicators.extend(indicator("url", value, normalized=normalize_url(value)) for value in matches)
        indicators.extend(indicator("email", value, normalized=value.casefold()) for value in emails)
        return FeatureBundle(self.name, "file", artifact_id(artifact), features, tuple(indicators), evidence=tuple(observations), limitations=tuple(limitations))
