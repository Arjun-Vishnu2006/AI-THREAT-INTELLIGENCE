from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import AnalysisSession, SecurityEvidence
from app.schemas.threat_intel import IOCCheckResponse
from app.services.contracts import IndicatorResult
from app.services.configured_intel import get_threat_intel_provider
from app.services.threat_intel import KnownThreatBaseline

router = APIRouter(prefix="/intel", tags=["threat intelligence"])


class IndicatorInput(BaseModel):
    type: str = Field(min_length=1, max_length=48)
    value: str = Field(min_length=1, max_length=8192)


class IOCCheckRequest(BaseModel):
    indicators: list[IndicatorInput] = Field(min_length=1, max_length=500)


@router.get("/dataset")
def dataset_info() -> dict[str, Any]:
    return get_threat_intel_provider().info()


@router.post("/check", response_model=IOCCheckResponse)
def check_indicators(payload: IOCCheckRequest,
                     session_id: str | None = Query(default=None),
                     db: Session = Depends(get_db)) -> dict[str, Any]:
    if session_id and db.get(AnalysisSession, session_id) is None:
        raise HTTPException(status_code=404, detail="Analysis session not found")
    baseline = KnownThreatBaseline(get_threat_intel_provider())
    assessments = []
    for item in payload.indicators:
        result = baseline.assess(IndicatorResult(item.type, item.value, item.value))
        match = None
        if result.match:
            match = {
                "record_id": result.match.record_id,
                "indicator_type": result.match.indicator_type,
                "matched_value": result.match.matched_value,
                "source": result.match.source,
                "classification": result.match.classification,
                "tags": list(result.match.tags),
            }
        assessments.append({
            "indicator_type": result.indicator_type,
            "value": result.value,
            "knowledge_status": result.knowledge_status,
            "behavior_status": result.behavior_status,
            "match": match,
        })
    if session_id:
        for assessment in assessments:
            db.add(SecurityEvidence(
                session_id=session_id,
                kind="known_threat_lookup",
                summary=f"IOC lookup result: {assessment['knowledge_status']}",
                producer=get_threat_intel_provider().name,
                payload=assessment,
            ))
        db.commit()
    return {"provider": get_threat_intel_provider().info(), "assessments": assessments}
