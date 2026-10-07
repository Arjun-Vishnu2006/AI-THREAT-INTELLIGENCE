import base64
import binascii
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field, model_validator
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import AnalysisSession
from app.services.analyzers import EmailAnalyzer, FileAnalyzer, QRAnalyzer, URLAnalyzer
from app.services.analyzers.common import MAX_ARTIFACT_BYTES
from app.services.evidence_store import persist_security_evidence

router = APIRouter(prefix="/analyze", tags=["analysis"])
_ANALYZERS = {"url": URLAnalyzer(), "email": EmailAnalyzer(), "qr": QRAnalyzer(), "file": FileAnalyzer()}


class AnalyzeRequest(BaseModel):
    artifact_id: str | None = Field(default=None, max_length=128)
    submitted_value: str | None = None
    content_base64: str | None = None
    name: str | None = Field(default=None, max_length=512)
    media_type: str | None = Field(default=None, max_length=255)

    @model_validator(mode="after")
    def require_input(self) -> "AnalyzeRequest":
        if self.submitted_value is None and self.content_base64 is None:
            raise ValueError("Provide submitted_value for URL/email or content_base64 for QR/file")
        return self


@router.post("/{kind}")
def analyze(kind: Literal["url", "email", "qr", "file"], payload: AnalyzeRequest,
            session_id: str | None = Query(default=None),
            db: Session = Depends(get_db)) -> dict[str, Any]:
    analyzer = _ANALYZERS[kind]
    artifact: dict[str, Any] = {"id": payload.artifact_id, "kind": kind,
                                "name": payload.name, "media_type": payload.media_type}
    if kind in {"url", "email"}:
        if payload.submitted_value is None:
            raise HTTPException(status_code=422, detail=f"{kind} analysis requires submitted_value")
        artifact["submitted_value"] = payload.submitted_value
        if kind == "email":
            artifact["content"] = payload.submitted_value
    else:
        if payload.content_base64 is None:
            raise HTTPException(status_code=422, detail=f"{kind} analysis requires content_base64")
        max_encoded_size = 4 * ((MAX_ARTIFACT_BYTES + 2) // 3)
        if len(payload.content_base64) > max_encoded_size:
            raise HTTPException(status_code=413, detail=f"Encoded content exceeds the {MAX_ARTIFACT_BYTES}-byte limit")
        try:
            content = base64.b64decode(payload.content_base64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise HTTPException(status_code=422, detail="content_base64 must be valid standard base64") from exc
        if len(content) > MAX_ARTIFACT_BYTES:
            raise HTTPException(status_code=413, detail=f"Decoded content exceeds {MAX_ARTIFACT_BYTES} bytes")
        artifact["content"] = content
    try:
        bundle = analyzer.analyze(artifact)
        response = bundle.to_dict()
        if session_id:
            if db.get(AnalysisSession, session_id) is None:
                raise HTTPException(status_code=404, detail="Analysis session not found")
            response["evidence_persisted"] = {
                "session_id": session_id,
                "record_count": persist_security_evidence(db, session_id, [bundle]),
            }
        return response
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
