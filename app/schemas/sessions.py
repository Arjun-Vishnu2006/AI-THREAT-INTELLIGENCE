from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class SessionCreate(BaseModel):
    source: str | None = Field(default=None, max_length=128)
    submitted_by: str | None = Field(default=None, max_length=256)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SessionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    status: str
    source: str | None
    submitted_by: str | None
    metadata: dict[str, Any] = Field(validation_alias="metadata_json")
    created_at: datetime
    updated_at: datetime


class ArtifactCreate(BaseModel):
    kind: Literal["url", "email", "qr", "file"]
    name: str | None = Field(default=None, max_length=512)
    media_type: str | None = Field(default=None, max_length=255)
    content_hash: str | None = Field(default=None, max_length=128)
    storage_ref: str | None = None
    submitted_value: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)


class ArtifactRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    session_id: str
    kind: str
    name: str | None
    media_type: str | None
    content_hash: str | None
    storage_ref: str | None
    submitted_value: str | None
    attributes: dict[str, Any]
    created_at: datetime
    updated_at: datetime
