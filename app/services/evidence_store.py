"""Persistence for structured evidence without retaining submitted content or file bytes."""

from datetime import datetime, timezone
from typing import Sequence

from sqlalchemy.orm import Session

from app.models import AnalysisSession, SecurityEvidence
from app.services.correlation import CorrelationResult
from app.services.feature_types import FeatureBundle


def persist_security_evidence(db: Session, session_id: str,
                              bundles: Sequence[FeatureBundle],
                              correlation: CorrelationResult | None = None,
                              *, include_evidence_ids: bool = False) -> int | dict[str, object]:
    """Store evidence summaries, normalized indicators and edges; never store raw artifact bodies."""
    if db.get(AnalysisSession, session_id) is None:
        raise LookupError("Analysis session not found")
    count = 0
    created_records: list[SecurityEvidence] = []
    observed_at = datetime.now(timezone.utc)
    for bundle in bundles:
        artifact_id = bundle.artifact_id
        for item in bundle.evidence:
            record = SecurityEvidence(
                session_id=session_id, artifact_id=artifact_id, kind=item.kind,
                summary=item.summary, producer=bundle.analyzer, observed_at=observed_at,
                payload={"references": list(item.references), "attributes": item.attributes},
            )
            db.add(record)
            created_records.append(record)
            count += 1
        for item in bundle.indicators:
            record = SecurityEvidence(
                session_id=session_id, artifact_id=artifact_id, kind="indicator_observed",
                summary=f"Observed {item.type} indicator", producer=bundle.analyzer,
                observed_at=observed_at,
                payload={"type": item.type, "value": item.value,
                         "normalized_value": item.normalized_value, "attributes": item.attributes},
            )
            db.add(record)
            created_records.append(record)
            count += 1
        for edge in bundle.relationships:
            record = SecurityEvidence(
                session_id=session_id, artifact_id=artifact_id, kind="relationship_observed",
                summary=f"Observed analyzer relationship: {edge.kind}", producer=bundle.analyzer,
                observed_at=observed_at,
                payload={"source_type": edge.source.type, "source": edge.source.normalized_value,
                         "target_type": edge.target.type, "target": edge.target.normalized_value,
                         "kind": edge.kind, "attributes": edge.attributes},
            )
            db.add(record)
            created_records.append(record)
            count += 1
    if correlation is not None:
        entities = {entity.id: entity for entity in correlation.entities}
        for edge in correlation.relationships:
            source, target = entities[edge.source_id], entities[edge.target_id]
            record = SecurityEvidence(
                session_id=session_id, kind="correlated_relationship",
                summary=f"Correlated {source.type} {edge.kind} {target.type}",
                producer="cross_input_correlator", observed_at=observed_at,
                payload={"source": source.to_dict(), "target": target.to_dict(),
                         "kind": edge.kind, "attributes": edge.attributes},
            )
            db.add(record)
            created_records.append(record)
            count += 1
    db.commit()
    if include_evidence_ids:
        return {"record_count": count, "evidence_ids": [record.id for record in created_records]}
    return count
