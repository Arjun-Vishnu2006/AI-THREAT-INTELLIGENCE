from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.correlation import CorrelationRequest, analyze_submitted_artifacts
from app.db import get_db
from app.ml.registry import load_registered_artifact
from app.models import AnalysisSession, ModelMetadata, SecurityEvidence
from app.novelty.engine import NoveltyEngine
from app.novelty.history import load_temporal_history
from app.novelty.types import snapshot_from_analysis
from app.services.correlation import CrossInputCorrelator
from app.services.evidence_store import persist_security_evidence
from app.services.threat_graph import NetworkXGraphProcessor

router = APIRouter(prefix="/novelty", tags=["novelty detection"])


@router.post("/analyze")
def analyze_novelty(payload: CorrelationRequest,
                    model_version: str | None = Query(default=None),
                    max_depth: int = Query(default=6, ge=1, le=12),
                    max_paths: int = Query(default=50, ge=1, le=500),
                    db: Session = Depends(get_db)) -> dict[str, Any]:
    if payload.session_id and db.get(AnalysisSession, payload.session_id) is None:
        raise HTTPException(status_code=404, detail="Analysis session not found")
    statement = select(ModelMetadata).where(
        ModelMetadata.name == "novelty-fusion",
        ModelMetadata.task == "multi_level_novelty_detection",
    )
    if model_version:
        statement = statement.where(ModelMetadata.version == model_version)
    statement = statement.order_by(ModelMetadata.created_at.desc())
    registered_model = db.scalar(statement)
    if registered_model is None:
        raise HTTPException(status_code=409, detail="No registered novelty-fusion model. Train one with app.novelty.train first.")
    try:
        engine = load_registered_artifact(registered_model)
    except (FileNotFoundError, ValueError, TypeError) as exc:
        raise HTTPException(status_code=409, detail=f"Registered novelty model cannot be loaded: {exc}") from exc
    if not isinstance(engine, NoveltyEngine):
        raise HTTPException(status_code=409, detail="Registered artifact is not a calibrated novelty model")

    bundles = analyze_submitted_artifacts(payload.artifacts)
    correlation = CrossInputCorrelator().correlate(bundles)
    graph = NetworkXGraphProcessor().process_correlation(correlation)
    history = load_temporal_history(db, payload.session_id)
    snapshot = snapshot_from_analysis(bundles, correlation, graph, history)
    assessment = engine.assess(snapshot)
    response = {
        "model": {"name": registered_model.name, "version": registered_model.version},
        "assessment": assessment,
        "graph": graph.to_dict(max_depth=max_depth, max_paths=max_paths),
        "temporal_reference_available": history is not None,
    }
    if payload.session_id:
        count = persist_security_evidence(db, payload.session_id, bundles, correlation)
        db.add(SecurityEvidence(
            session_id=payload.session_id, kind="novelty_assessment",
            summary=f"Novelty assessment result: {assessment['novelty_status']}",
            producer=f"novelty-fusion:{registered_model.version}", payload=assessment,
        ))
        db.commit()
        response["evidence_persisted"] = {"session_id": payload.session_id, "record_count": count + 1}
    return response
