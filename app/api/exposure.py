"""Calibrated exposure category inference backed by extracted evidence."""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.correlation import CorrelationRequest, analyze_submitted_artifacts
from app.db import get_db
from app.exposure.engine import CalibratedExposureModel
from app.exposure.features import build_exposure_features, explain_exposure_features
from app.ml.registry import load_registered_artifact
from app.models import AnalysisSession, ModelMetadata, SecurityEvidence
from app.novelty.engine import NoveltyEngine
from app.novelty.history import load_temporal_history
from app.novelty.types import snapshot_from_analysis
from app.services.configured_intel import get_threat_intel_provider
from app.services.contracts import IndicatorResult
from app.services.correlation import CrossInputCorrelator
from app.services.evidence_store import persist_security_evidence
from app.services.threat_graph import NetworkXGraphProcessor
from app.services.threat_intel import KnownThreatBaseline

router = APIRouter(prefix="/exposure", tags=["exposure prediction"])


def _registered_model(db: Session, *, name: str, task: str,
                      version: str | None) -> ModelMetadata:
    query = select(ModelMetadata).where(ModelMetadata.name == name, ModelMetadata.task == task)
    if version:
        query = query.where(ModelMetadata.version == version)
    model = db.scalar(query.order_by(ModelMetadata.created_at.desc()))
    if model is None:
        raise HTTPException(status_code=409, detail=f"No registered {name} model is available")
    return model


@router.post("/predict")
def predict_exposure(payload: CorrelationRequest,
                     model_version: str | None = Query(default=None),
                     novelty_model_version: str | None = Query(default=None),
                     max_depth: int = Query(default=6, ge=1, le=12),
                     max_paths: int = Query(default=50, ge=1, le=500),
                     db: Session = Depends(get_db)) -> dict[str, Any]:
    if payload.session_id and db.get(AnalysisSession, payload.session_id) is None:
        raise HTTPException(status_code=404, detail="Analysis session not found")
    exposure_record = _registered_model(
        db, name="exposure-predictor", task="exposure_category_prediction", version=model_version,
    )
    novelty_record = _registered_model(
        db, name="novelty-fusion", task="multi_level_novelty_detection", version=novelty_model_version,
    )
    try:
        exposure_model = load_registered_artifact(exposure_record)
        novelty_model = load_registered_artifact(novelty_record)
    except (FileNotFoundError, ValueError, TypeError) as exc:
        raise HTTPException(status_code=409, detail=f"Registered model artifact cannot be loaded: {exc}") from exc
    if not isinstance(exposure_model, CalibratedExposureModel):
        raise HTTPException(status_code=409, detail="Registered exposure artifact is not a calibrated category model")
    if not isinstance(novelty_model, NoveltyEngine):
        raise HTTPException(status_code=409, detail="Registered novelty artifact is not a calibrated novelty model")

    bundles = analyze_submitted_artifacts(payload.artifacts)
    correlation = CrossInputCorrelator().correlate(bundles)
    graph = NetworkXGraphProcessor().process_correlation(correlation)
    history = load_temporal_history(db, payload.session_id)
    snapshot = snapshot_from_analysis(bundles, correlation, graph, history)
    novelty = novelty_model.assess(snapshot)

    baseline = KnownThreatBaseline(get_threat_intel_provider())
    assessments = []
    for bundle in bundles:
        for item in bundle.indicators:
            assessment = baseline.assess(IndicatorResult(item.type, item.value, item.normalized_value))
            assessments.append(assessment)
    known = [item for item in assessments if item.knowledge_status == "KNOWN"]
    knowledge_status = "KNOWN" if known else ("UNKNOWN" if assessments else "NOT_ASSESSED")
    features = build_exposure_features(
        snapshot, {**novelty, "knowledge_status": knowledge_status},
        known_ioc_hit_count=len(known), indicator_count=len(assessments),
    )
    probabilities = exposure_model.predict_proba([features])[0]
    classes = list(exposure_model.probability_calibrator.classes_)
    selected_index = int(probabilities.argmax())
    category = classes[selected_index]
    linear_explanation = exposure_model.explain(features, category)
    evidence_explanation = explain_exposure_features(features, novelty)
    response = {
        "model": {"name": exposure_record.name, "version": exposure_record.version,
                  "calibration_method": exposure_record.parameters.get("calibration_method")},
        "novelty_model": {"name": novelty_record.name, "version": novelty_record.version},
        "prediction": {
            "exposure_category": category,
            "probabilities": {name: float(probabilities[index]) for index, name in enumerate(classes)},
            "confidence": float(probabilities[selected_index]),
        },
        "knowledge_status": knowledge_status,
        "anomaly_status": novelty["anomaly_status"],
        "novelty_status": novelty["novelty_status"],
        "known_matches": [{"indicator_type": item.indicator_type,
                           "record_id": item.record_id,
                           "matched_value": item.matched_value,
                           "source": item.source,
                           "classification": item.classification,
                           "tags": list(item.tags)} for item in (match.match for match in known) if item],
        "explanation": {
            "model_feature_contributions": linear_explanation,
            "evidence": evidence_explanation,
            "interpretation": (
                "Contributions describe features the fitted model associated with this synthetic exposure category; "
                "they are not causal evidence or a measure of real-world impact."
            ),
        },
        "novelty_assessment": novelty,
        "graph": graph.to_dict(max_depth=max_depth, max_paths=max_paths),
        "temporal_reference_available": history is not None,
    }
    if payload.session_id:
        persisted = persist_security_evidence(
            db, payload.session_id, bundles, correlation, include_evidence_ids=True,
        )
        assert isinstance(persisted, dict)
        known_record = SecurityEvidence(
            session_id=payload.session_id, kind="known_threat_assessment",
            summary=f"Exact-match IOC assessment: {knowledge_status}",
            producer=get_threat_intel_provider().name,
            payload={
                "knowledge_status": knowledge_status,
                "indicator_count": len(assessments),
                "known_match_count": len(known),
                "matches": [{"indicator_type": item.indicator_type,
                             "record_id": item.match.record_id,
                             "source": item.match.source,
                             "classification": item.match.classification,
                             "tags": list(item.match.tags)}
                            for item in known if item.match],
            },
        )
        exposure_record_row = SecurityEvidence(
            session_id=payload.session_id, kind="exposure_assessment",
            summary=f"Predicted exposure category: {category}",
            producer=f"exposure-predictor:{exposure_record.version}",
            payload={"category": category, "probabilities": response["prediction"]["probabilities"],
                     "knowledge_status": knowledge_status,
                     "anomaly_status": novelty["anomaly_status"],
                     "novelty_status": novelty["novelty_status"],
                     "explanation": response["explanation"],
                     "novelty_assessment": novelty},
        )
        db.add_all([known_record, exposure_record_row])
        db.flush()
        evidence_ids = list(persisted["evidence_ids"]) + [known_record.id, exposure_record_row.id]
        db.commit()
        response["evidence_persisted"] = {
            "session_id": payload.session_id,
            "record_count": int(persisted["record_count"]) + 2,
            "evidence_ids": evidence_ids,
        }
    return response
