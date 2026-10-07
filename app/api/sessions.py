from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import Alert, AlertEvent, AnalysisSession, Artifact, EvidenceLink, Finding, SecurityEvidence
from app.schemas.records import SecurityEvidenceRead
from app.schemas.sessions import ArtifactCreate, ArtifactRead, SessionCreate, SessionRead

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.post("", response_model=SessionRead, status_code=status.HTTP_201_CREATED)
def create_session(payload: SessionCreate, db: Session = Depends(get_db)) -> AnalysisSession:
    session = AnalysisSession(source=payload.source, submitted_by=payload.submitted_by, metadata_json=payload.metadata)
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


@router.get("", response_model=list[SessionRead])
def list_sessions(limit: int = 50, offset: int = 0, db: Session = Depends(get_db)) -> list[AnalysisSession]:
    if not 1 <= limit <= 100 or offset < 0:
        raise HTTPException(status_code=422, detail="limit must be 1-100 and offset must be non-negative")
    return list(db.scalars(select(AnalysisSession).order_by(AnalysisSession.created_at.desc()).offset(offset).limit(limit)))


@router.get("/{session_id}", response_model=SessionRead)
def get_session(session_id: str, db: Session = Depends(get_db)) -> AnalysisSession:
    session = db.get(AnalysisSession, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Analysis session not found")
    return session


@router.post("/{session_id}/artifacts", response_model=ArtifactRead, status_code=status.HTTP_201_CREATED)
def add_artifact(session_id: str, payload: ArtifactCreate, db: Session = Depends(get_db)) -> Artifact:
    if db.get(AnalysisSession, session_id) is None:
        raise HTTPException(status_code=404, detail="Analysis session not found")
    artifact = Artifact(session_id=session_id, **payload.model_dump())
    db.add(artifact)
    db.commit()
    db.refresh(artifact)
    return artifact


@router.get("/{session_id}/artifacts", response_model=list[ArtifactRead])
def list_artifacts(session_id: str, db: Session = Depends(get_db)) -> list[Artifact]:
    if db.get(AnalysisSession, session_id) is None:
        raise HTTPException(status_code=404, detail="Analysis session not found")
    return list(db.scalars(select(Artifact).where(Artifact.session_id == session_id).order_by(Artifact.created_at)))


@router.get("/{session_id}/evidence", response_model=list[SecurityEvidenceRead])
def list_security_evidence(session_id: str, db: Session = Depends(get_db)) -> list[SecurityEvidence]:
    if db.get(AnalysisSession, session_id) is None:
        raise HTTPException(status_code=404, detail="Analysis session not found")
    query = select(SecurityEvidence).where(SecurityEvidence.session_id == session_id).order_by(SecurityEvidence.observed_at)
    return list(db.scalars(query))


@router.get("/{session_id}/history")
def get_session_history(session_id: str, db: Session = Depends(get_db)) -> dict:
    if db.get(AnalysisSession, session_id) is None:
        raise HTTPException(status_code=404, detail="Analysis session not found")
    evidence = list(db.scalars(
        select(SecurityEvidence).where(SecurityEvidence.session_id == session_id)
        .order_by(SecurityEvidence.observed_at)
    ))
    findings = list(db.scalars(
        select(Finding).where(Finding.session_id == session_id).order_by(Finding.created_at)
    ))
    alerts = list(db.scalars(
        select(Alert).where(Alert.session_id == session_id).order_by(Alert.triggered_at)
    ))
    links = list(db.scalars(
        select(EvidenceLink).where(EvidenceLink.session_id == session_id).order_by(EvidenceLink.created_at)
    ))
    alert_ids = [item.id for item in alerts]
    events = list(db.scalars(
        select(AlertEvent).where(AlertEvent.alert_id.in_(alert_ids)).order_by(AlertEvent.created_at)
    )) if alert_ids else []
    return {
        "session_id": session_id,
        "evidence": [{"id": item.id, "kind": item.kind, "summary": item.summary,
                      "producer": item.producer, "artifact_id": item.artifact_id,
                      "observed_at": item.observed_at, "payload": item.payload} for item in evidence],
        "findings": [{"id": item.id, "category": item.category, "title": item.title,
                       "description": item.description, "status": item.status,
                       "evidence": item.evidence, "producer": item.producer} for item in findings],
        "alerts": [{"id": item.id, "finding_id": item.finding_id, "status": item.status,
                    "severity": item.severity, "title": item.title,
                    "triggered_at": item.triggered_at} for item in alerts],
        "alert_events": [{"id": item.id, "alert_id": item.alert_id,
                          "event_type": item.event_type, "previous_status": item.previous_status,
                          "status": item.status, "actor": item.actor, "note": item.note,
                          "created_at": item.created_at} for item in events],
        "evidence_links": [{"id": item.id, "source_evidence_id": item.source_evidence_id,
                            "target_evidence_id": item.target_evidence_id,
                            "kind": item.kind, "attributes": item.attributes} for item in links],
    }
