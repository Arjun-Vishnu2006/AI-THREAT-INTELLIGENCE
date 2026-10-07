"""Provider-neutral exact-match IOC lookup; no network queries or inferred verdicts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from app.services.contracts import IndicatorResult
from app.services.entities import normalize_indicator


@dataclass(frozen=True)
class IOCRecord:
    id: str
    type: str
    value: str
    source: str
    classification: str
    tags: tuple[str, ...]
    normalized_value: str


@dataclass(frozen=True)
class IOCMatch:
    record_id: str
    indicator_type: str
    matched_value: str
    source: str
    classification: str
    tags: tuple[str, ...]


class ThreatIntelProvider(Protocol):
    name: str

    def lookup(self, indicator: IndicatorResult) -> IOCMatch | None: ...


class LocalJSONThreatIntelProvider:
    name = "local_json_ioc"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.dataset_id: str | None = None
        self.records: dict[tuple[str, str], IOCRecord] = {}
        self.reload()

    def reload(self) -> None:
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1 or not isinstance(payload.get("records"), list):
            raise ValueError(f"Unsupported IOC dataset schema: {self.path}")
        self.dataset_id = str(payload.get("dataset_id", self.path.stem))
        loaded: dict[tuple[str, str], IOCRecord] = {}
        for raw in payload["records"]:
            if not isinstance(raw, dict):
                raise ValueError("IOC records must be JSON objects")
            required = ("id", "type", "value", "source", "classification")
            if any(not isinstance(raw.get(field), str) or not raw[field].strip() for field in required):
                raise ValueError("IOC record is missing a required non-empty text field")
            normalized = normalize_indicator(IndicatorResult(raw["type"], raw["value"], raw["value"]))
            key = (normalized.type, normalized.normalized_value)
            if key in loaded:
                raise ValueError(f"Duplicate normalized IOC in dataset: {raw['id']}")
            loaded[key] = IOCRecord(
                id=raw["id"], type=normalized.type, value=raw["value"], source=raw["source"],
                classification=raw["classification"], tags=tuple(raw.get("tags", [])),
                normalized_value=normalized.normalized_value,
            )
        self.records = loaded

    def lookup(self, indicator: IndicatorResult) -> IOCMatch | None:
        normalized = normalize_indicator(indicator)
        record = self.records.get((normalized.type, normalized.normalized_value))
        if record is None:
            return None
        return IOCMatch(record.id, record.type, record.value, record.source,
                        record.classification, record.tags)

    def info(self) -> dict[str, Any]:
        return {"provider": self.name, "dataset_id": self.dataset_id,
                "record_count": len(self.records), "path": str(self.path)}


@dataclass(frozen=True)
class IndicatorAssessment:
    indicator_type: str
    value: str
    knowledge_status: str  # KNOWN or UNKNOWN
    behavior_status: str  # NOT_ASSESSED, ANOMALOUS, or NOVEL; independent axis
    match: IOCMatch | None = None


class KnownThreatBaseline:
    def __init__(self, provider: ThreatIntelProvider) -> None:
        self.provider = provider

    def assess(self, indicator: IndicatorResult) -> IndicatorAssessment:
        match = self.provider.lookup(indicator)
        return IndicatorAssessment(
            indicator_type=indicator.type,
            value=indicator.value,
            knowledge_status="KNOWN" if match else "UNKNOWN",
            behavior_status="NOT_ASSESSED",
            match=match,
        )
