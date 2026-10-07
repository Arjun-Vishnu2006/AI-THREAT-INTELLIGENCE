"""Build per-session temporal references from stored structured evidence."""

from collections import Counter

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import SecurityEvidence
from app.novelty.types import TemporalHistory, relationship_signature


def load_temporal_history(db: Session, session_id: str | None) -> TemporalHistory | None:
    if session_id is None:
        return None
    statement = (
        select(SecurityEvidence)
        .where(SecurityEvidence.session_id == session_id)
        .where(SecurityEvidence.kind.in_(
            ["indicator_observed", "relationship_observed", "correlated_relationship"]
        ))
        .order_by(SecurityEvidence.observed_at.desc())
        .limit(20000)
    )
    rows = list(db.scalars(statement))
    entity_counts: Counter[str] = Counter()
    relationship_counts: Counter[str] = Counter()
    for row in rows:
        payload = row.payload or {}
        if row.kind == "indicator_observed":
            kind, value = payload.get("type"), payload.get("normalized_value")
            if kind and value:
                entity_counts[f"{kind}:{value}"] += 1
        elif row.kind == "relationship_observed":
            source_type, source_value = payload.get("source_type"), payload.get("source")
            target_type, target_value = payload.get("target_type"), payload.get("target")
            edge_kind = payload.get("kind")
            if source_type and source_value and target_type and target_value:
                entity_counts[f"{source_type}:{source_value}"] += 1
                entity_counts[f"{target_type}:{target_value}"] += 1
            if source_type and target_type and edge_kind:
                relationship_counts[relationship_signature(source_type, edge_kind, target_type)] += 1
        else:
            source, target = payload.get("source", {}), payload.get("target", {})
            edge_kind = payload.get("kind")
            source_type = (source.get("attributes") or {}).get("artifact_kind", source.get("type"))
            target_type = (target.get("attributes") or {}).get("artifact_kind", target.get("type"))
            if source.get("type") != "artifact" and source.get("type") and source.get("normalized_value"):
                entity_counts[f"{source['type']}:{source['normalized_value']}"] += 1
            if target.get("type") != "artifact" and target.get("type") and target.get("normalized_value"):
                entity_counts[f"{target['type']}:{target['normalized_value']}"] += 1
            if source_type and target_type and edge_kind and edge_kind.upper() != "CONTAINS":
                relationship_counts[relationship_signature(source_type, edge_kind, target_type)] += 1
    if not entity_counts and not relationship_counts:
        return None
    return TemporalHistory(dict(entity_counts), dict(relationship_counts))
