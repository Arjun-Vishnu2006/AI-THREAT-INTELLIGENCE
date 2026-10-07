from typing import Any, Literal

from pydantic import BaseModel


class IOCMatchRead(BaseModel):
    record_id: str
    indicator_type: str
    matched_value: str
    source: str
    classification: str
    tags: list[str]


class IndicatorAssessmentRead(BaseModel):
    indicator_type: str
    value: str
    knowledge_status: Literal["KNOWN", "UNKNOWN"]
    behavior_status: Literal["NOT_ASSESSED", "ANOMALOUS", "NOVEL"]
    match: IOCMatchRead | None


class IOCCheckResponse(BaseModel):
    provider: dict[str, Any]
    assessments: list[IndicatorAssessmentRead]
