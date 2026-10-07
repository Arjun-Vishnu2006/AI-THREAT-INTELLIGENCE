"""Risk policy, evidence provenance chain, findings and alert creation."""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.correlation import CorrelationRequest
from app.api.exposure import predict_exposure
from app.db import get_db
from app.models import (
    Alert, AlertEvent, AnalysisSession, EvidenceLink, Finding, SecurityEvidence,
)
from app.services.risk_assessment import assess_evidence_risk

router = APIRouter(prefix="/risk", tags=["risk assessment and evidence chain"])


@router.post("/assess")
def assess_risk(payload: CorrelationRequest,
                model_version: str | None = Query(default=None),
                novelty_model_version: str | None = Query(default=None),
                max_depth: int = Query(default=6, ge=1, le=12),
                max_paths: int = Query(default=50, ge=1, le=500),
                db: Session = Depends(get_db)) -> dict[str, Any]:
    if payload.session_id is None:
        raise HTTPException(status_code=422, detail="Risk assessment requires a session_id for evidence and alert history")
    if db.get(AnalysisSession, payload.session_id) is None:
        raise HTTPException(status_code=404, detail="Analysis session not found")

    exposure = predict_exposure(
        payload, model_version=model_version,
        novelty_model_version=novelty_model_version,
        max_depth=max_depth, max_paths=max_paths, db=db,
    )
    risk = assess_evidence_risk(exposure)
    persisted_ids = exposure.get("evidence_persisted", {}).get("evidence_ids", [])
    current_evidence = list(db.scalars(
        select(SecurityEvidence).where(
            SecurityEvidence.session_id == payload.session_id,
            SecurityEvidence.id.in_(persisted_ids),
        )
    )) if persisted_ids else []
    source_evidence_ids = [row.id for row in current_evidence
                           if row.kind not in {"exposure_assessment", "known_threat_assessment"}]
    threat_evidence_ids = [row.id for row in current_evidence if row.kind == "known_threat_assessment"]
    exposure_rows = [row for row in current_evidence if row.kind == "exposure_assessment"]
    exposure_evidence_id = exposure_rows[-1].id if exposure_rows else None

    graph_evidence = SecurityEvidence(
        session_id=payload.session_id, kind="graph_assessment",
        summary="Cross-input graph and bounded paths derived from current analyzer evidence",
        producer="networkx-graph-processor",
        payload={
            "artifact_ids": [item.artifact_id for item in payload.artifacts if item.artifact_id],
            "entities": exposure["graph"].get("entities", []),
            "relationships": exposure["graph"].get("relationships", []),
            "features": exposure["graph"].get("features", {}),
            "paths": exposure["graph"].get("paths", []),
        },
    )
    novelty_evidence = SecurityEvidence(
        session_id=payload.session_id, kind="novelty_assessment",
        summary=f"Novelty assessment: {exposure['novelty_status']}",
        producer=f"novelty-fusion:{exposure['novelty_model']['version']}",
        payload=exposure["novelty_assessment"],
    )
    risk_evidence = SecurityEvidence(
        session_id=payload.session_id, kind="risk_assessment",
        summary=f"Evidence risk policy result: {risk['risk_level']}",
        producer=risk["policy_version"],
        payload={
            **risk,
            "exposure_model": exposure["model"],
            "novelty_model": exposure["novelty_model"],
            "exposure_prediction": exposure["prediction"],
            "knowledge_status": exposure["knowledge_status"],
            "anomaly_status": exposure["anomaly_status"],
            "novelty_status": exposure["novelty_status"],
        },
    )
    db.add_all([graph_evidence, novelty_evidence, risk_evidence])
    db.flush()

    links = [
        EvidenceLink(
            session_id=payload.session_id,
            source_evidence_id=source_id,
            target_evidence_id=graph_evidence.id,
            kind="DERIVES_GRAPH_FROM",
        )
        for source_id in source_evidence_ids
    ]
    links.append(EvidenceLink(
        session_id=payload.session_id, source_evidence_id=graph_evidence.id,
        target_evidence_id=novelty_evidence.id, kind="GRAPH_EVALUATED_FOR_NOVELTY",
        attributes={"model_version": exposure["novelty_model"]["version"]},
    ))
    if exposure_evidence_id:
        links.append(EvidenceLink(
            session_id=payload.session_id, source_evidence_id=novelty_evidence.id,
            target_evidence_id=exposure_evidence_id, kind="NOVELTY_FEATURES_FOR_EXPOSURE",
            attributes={"model_version": exposure["model"]["version"]},
        ))
        links.extend(EvidenceLink(
            session_id=payload.session_id, source_evidence_id=threat_id,
            target_evidence_id=exposure_evidence_id, kind="IOC_FEATURES_FOR_EXPOSURE",
        ) for threat_id in threat_evidence_ids)
    risk_sources = [graph_evidence.id, novelty_evidence.id]
    if exposure_evidence_id:
        risk_sources.append(exposure_evidence_id)
    risk_sources.extend(
        row.id for row in current_evidence
        if row.kind in {"indicator_observed", "known_threat_lookup", "known_threat_assessment",
                        "correlated_relationship"}
    )
    links.extend(EvidenceLink(
        session_id=payload.session_id, source_evidence_id=source_id,
        target_evidence_id=risk_evidence.id, kind="CONTRIBUTES_TO_RISK",
    ) for source_id in dict.fromkeys(risk_sources))
    db.add_all(links)

    artifact_ids = list(dict.fromkeys(
        [item.artifact_id for item in payload.artifacts if item.artifact_id]
        + [row.artifact_id for row in current_evidence if row.artifact_id]
    ))
    evidence_ids = list(dict.fromkeys(
        source_evidence_ids + threat_evidence_ids
        + [graph_evidence.id, novelty_evidence.id, risk_evidence.id]
        + ([exposure_evidence_id] if exposure_evidence_id else [])
    ))
    finding = Finding(
        session_id=payload.session_id, category="evidence_risk_assessment",
        title=f"Evidence risk level: {risk['risk_level']}",
        description=risk["rationale"],
        status="alerted" if risk["alert_recommended"] else "observed",
        evidence={
            "risk_evidence_id": risk_evidence.id,
            "evidence_record_ids": evidence_ids,
            "artifact_ids": artifact_ids,
            "risk_level": risk["risk_level"],
            "policy_version": risk["policy_version"],
            "factors": risk["factors"],
        },
        producer=risk["policy_version"],
    )
    db.add(finding)
    db.flush()

    alert = None
    if risk["alert_recommended"]:
        alert = Alert(
            session_id=payload.session_id, finding_id=finding.id,
            status="open", severity="high" if risk["risk_level"] == "HIGH" else "medium",
            title=f"{risk['risk_level']} evidence risk requires review",
            summary=risk["rationale"],
        )
        db.add(alert)
        db.flush()
        db.add(AlertEvent(
            alert_id=alert.id, event_type="created", previous_status=None,
            status="open", actor="risk_policy", note=risk["rationale"],
        ))
    db.commit()
    return {
        "session_id": payload.session_id,
        "risk_assessment": risk,
        "exposure_prediction": exposure["prediction"],
        "statuses": {"knowledge": exposure["knowledge_status"],
                     "anomaly": exposure["anomaly_status"],
                     "novelty": exposure["novelty_status"]},
        "finding_id": finding.id,
        "alert_id": alert.id if alert else None,
        "evidence_chain": {
            "source_evidence_ids": source_evidence_ids,
            "threat_evidence_ids": threat_evidence_ids,
            "graph_evidence_id": graph_evidence.id,
            "novelty_evidence_id": novelty_evidence.id,
            "exposure_evidence_id": exposure_evidence_id,
            "risk_evidence_id": risk_evidence.id,
            "link_count": len(links),
        },
    }
