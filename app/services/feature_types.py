"""Shared, JSON-friendly feature and evidence structures emitted by analyzers."""

from dataclasses import dataclass, field
from typing import Any

from app.services.contracts import Evidence, IndicatorResult, RelationshipResult


@dataclass(frozen=True)
class FeatureBundle:
    """Uniform analyzer output; feature names are analyzer-prefixed and values are serializable."""

    analyzer: str
    artifact_kind: str
    artifact_id: str | None = None
    features: dict[str, Any] = field(default_factory=dict)
    indicators: tuple[IndicatorResult, ...] = ()
    relationships: tuple[RelationshipResult, ...] = ()
    evidence: tuple[Evidence, ...] = ()
    limitations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "analyzer": self.analyzer,
            "artifact_kind": self.artifact_kind,
            "artifact_id": self.artifact_id,
            "features": self.features,
            "indicators": [
                {"type": i.type, "value": i.value, "normalized_value": i.normalized_value,
                 "attributes": i.attributes}
                for i in self.indicators
            ],
            "relationships": [
                {"source": r.source.normalized_value, "target": r.target.normalized_value,
                 "kind": r.kind, "attributes": r.attributes}
                for r in self.relationships
            ],
            "evidence": [
                {"kind": e.kind, "summary": e.summary, "references": list(e.references),
                 "attributes": e.attributes}
                for e in self.evidence
            ],
            "limitations": list(self.limitations),
        }
