"""Alert history, lifecycle updates, and evidence-to-artifact investigation views."""

from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import Alert, AlertEvent, Artifact, EvidenceLink, Finding, SecurityEvidence

router = APIRouter(prefix="/alerts", tags=["alerts and investigations"])


class AlertStatusUpdate(BaseModel):
    status: Literal["open", "acknowledged", "resolved", "false_positive"]
    actor: str | None = Field(default=None, max_length=256)
    note: str | None = Field(default=None, max_length=4000)


@router.get("")
def list_alerts(session_id: str | None = None, status: str | None = None,
                limit: int = Query(default=50, ge=1, le=200),
                offset: int = Query(default=0, ge=0),
                db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    query = select(Alert)
    if session_id:
        query = query.where(Alert.session_id == session_id)
    if status:
        query = query.where(Alert.status == status)
    alerts = list(db.scalars(query.order_by(Alert.triggered_at.desc()).offset(offset).limit(limit)))
    return [{
        "id": item.id, "session_id": item.session_id, "finding_id": item.finding_id,
        "status": item.status, "severity": item.severity, "title": item.title,
        "summary": item.summary, "triggered_at": item.triggered_at,
        "updated_at": item.updated_at,
    } for item in alerts]


@router.patch("/{alert_id}")
def update_alert(alert_id: str, payload: AlertStatusUpdate,
                 db: Session = Depends(get_db)) -> dict[str, Any]:
    alert = db.get(Alert, alert_id)
    if alert is None:
        raise HTTPException(status_code=404, detail="Alert not found")
    previous = alert.status
    if previous == payload.status:
        return {"id": alert.id, "status": alert.status, "event_recorded": False}
    allowed = {
        "open": {"acknowledged", "resolved", "false_positive"},
        "acknowledged": {"open", "resolved", "false_positive"},
        "resolved": {"open"},
        "false_positive": {"open"},
    }
    if payload.status not in allowed.get(previous, set()):
        raise HTTPException(status_code=409, detail=f"Cannot change alert from {previous} to {payload.status}")
    alert.status = payload.status
    db.add(AlertEvent(
        alert_id=alert.id, event_type="status_changed", previous_status=previous,
        status=payload.status, actor=payload.actor, note=payload.note,
    ))
    db.commit()
    db.refresh(alert)
    return {"id": alert.id, "status": alert.status, "previous_status": previous,
            "event_recorded": True}


@router.get("/{alert_id}/investigation")
def get_alert_investigation(alert_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    alert = db.get(Alert, alert_id)
    if alert is None:
        raise HTTPException(status_code=404, detail="Alert not found")
    finding = db.get(Finding, alert.finding_id) if alert.finding_id else None
    if finding is None:
        raise HTTPException(status_code=404, detail="Alert finding is unavailable")
    chain_ids = list(dict.fromkeys(finding.evidence.get("evidence_record_ids", [])))
    evidence_rows = list(db.scalars(
        select(SecurityEvidence)
        .where(SecurityEvidence.session_id == alert.session_id)
        .where(SecurityEvidence.id.in_(chain_ids))
        .order_by(SecurityEvidence.observed_at)
    )) if chain_ids else []
    links = list(db.scalars(
        select(EvidenceLink)
        .where(EvidenceLink.session_id == alert.session_id)
        .where(or_(EvidenceLink.source_evidence_id.in_(chain_ids),
                   EvidenceLink.target_evidence_id.in_(chain_ids)))
        .order_by(EvidenceLink.created_at)
    )) if chain_ids else []
    artifact_ids = list(dict.fromkeys(finding.evidence.get("artifact_ids", [])))
    artifacts = list(db.scalars(
        select(Artifact).where(Artifact.session_id == alert.session_id, Artifact.id.in_(artifact_ids))
    )) if artifact_ids else []
    stored_by_id = {item.id: item for item in artifacts}
    graph_row = next((item for item in evidence_rows if item.kind == "graph_assessment"), None)
    graph_payload = graph_row.payload if graph_row else {}
    related_artifacts = []
    for entity in graph_payload.get("entities", []):
        if entity.get("type") != "artifact":
            continue
        normalized = str(entity.get("normalized_value", ""))
        kind, _, entity_id = normalized.partition(":")
        artifact_id = entity_id or entity.get("value")
        stored = stored_by_id.get(artifact_id)
        related_artifacts.append({
            "id": artifact_id,
            "kind": (entity.get("attributes") or {}).get("artifact_kind", kind),
            "persisted_artifact": stored is not None,
            "name": stored.name if stored else None,
            "media_type": stored.media_type if stored else None,
        })
    events = list(db.scalars(
        select(AlertEvent).where(AlertEvent.alert_id == alert.id).order_by(AlertEvent.created_at)
    ))
    return {
        "alert": {"id": alert.id, "session_id": alert.session_id, "status": alert.status,
                  "severity": alert.severity, "title": alert.title, "summary": alert.summary,
                  "triggered_at": alert.triggered_at, "updated_at": alert.updated_at},
        "finding": {
            "id": finding.id, "category": finding.category, "title": finding.title,
            "description": finding.description, "status": finding.status,
            "evidence": finding.evidence, "producer": finding.producer,
        },
        "alert_history": [{
            "id": item.id, "event_type": item.event_type,
            "previous_status": item.previous_status, "status": item.status,
            "actor": item.actor, "note": item.note, "created_at": item.created_at,
        } for item in events],
        "evidence_chain": [{
            "id": item.id, "kind": item.kind, "summary": item.summary,
            "producer": item.producer, "artifact_id": item.artifact_id,
            "observed_at": item.observed_at, "payload": item.payload,
        } for item in evidence_rows],
        "evidence_links": [{
            "id": item.id, "source_evidence_id": item.source_evidence_id,
            "target_evidence_id": item.target_evidence_id, "kind": item.kind,
            "attributes": item.attributes,
        } for item in links],
        "related_artifacts": related_artifacts,
        "graph": {"features": graph_payload.get("features", {}),
                  "relationships": graph_payload.get("relationships", []),
                  "paths": graph_payload.get("paths", [])},
    }
