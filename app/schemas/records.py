from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class IndicatorCreate(BaseModel):
    type: str = Field(min_length=1, max_length=48)
    value: str = Field(min_length=1)
    normalized_value: str = Field(min_length=1)
    attributes: dict[str, Any] = Field(default_factory=dict)


class IndicatorRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    type: str
    value: str
    normalized_value: str
    first_seen_at: datetime | None
    last_seen_at: datetime | None
    attributes: dict[str, Any]


class RelationshipCreate(BaseModel):
    source_indicator_id: str
    target_indicator_id: str
    kind: str = Field(min_length=1, max_length=64)
    session_id: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)


class FindingRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    session_id: str
    category: str
    title: str
    description: str | None
    status: str
    confidence: float | None
    evidence: dict[str, Any]
    producer: str | None
    created_at: datetime


class AlertRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    session_id: str
    finding_id: str | None
    status: str
    severity: str | None
    title: str
    summary: str | None
    triggered_at: datetime


class ModelMetadataRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    name: str
    version: str
    task: str
    artifact_ref: str | None
    parameters: dict[str, Any]
    provenance: dict[str, Any]


class ExperimentMetadataRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    name: str
    version: str | None
    status: str
    configuration: dict[str, Any]
    results: dict[str, Any]


class SecurityEvidenceRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    session_id: str
    artifact_id: str | None
    kind: str
    summary: str
    producer: str | None
    observed_at: datetime
    payload: dict[str, Any]
    created_at: datetime
