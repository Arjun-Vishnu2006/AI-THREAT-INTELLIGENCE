import base64
import binascii
import hashlib
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field, model_validator
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import AnalysisSession
from app.services.analyzers import EmailAnalyzer, FileAnalyzer, QRAnalyzer, URLAnalyzer
from app.services.analyzers.common import MAX_ARTIFACT_BYTES
from app.services.correlation import CrossInputCorrelator
from app.services.evidence_store import persist_security_evidence
from app.services.synthetic_cases import build_synthetic_correlation_cases
from app.services.threat_graph import NetworkXGraphProcessor

router = APIRouter(prefix="/correlate", tags=["correlation and graph"])
_ANALYZERS = {"url": URLAnalyzer(), "email": EmailAnalyzer(), "qr": QRAnalyzer(), "file": FileAnalyzer()}


class CorrelationArtifact(BaseModel):
    kind: Literal["url", "email", "qr", "file"]
    artifact_id: str | None = Field(default=None, max_length=128)
    submitted_value: str | None = None
    content_base64: str | None = None
    name: str | None = Field(default=None, max_length=512)
    media_type: str | None = Field(default=None, max_length=255)

    @model_validator(mode="after")
    def validate_input(self) -> "CorrelationArtifact":
        if self.kind in {"url", "email"} and self.submitted_value is None:
            raise ValueError(f"{self.kind} artifacts require submitted_value")
        if self.kind in {"qr", "file"} and self.content_base64 is None:
            raise ValueError(f"{self.kind} artifacts require content_base64")
        if self.kind in {"url", "email"} and self.content_base64 is not None:
            raise ValueError(f"{self.kind} artifacts do not accept content_base64")
        return self


class CorrelationRequest(BaseModel):
    artifacts: list[CorrelationArtifact] = Field(min_length=1, max_length=50)
    session_id: str | None = None


def analyze_submitted_artifacts(submitted_artifacts: list[CorrelationArtifact]):
    bundles = []
    max_encoded_size = 4 * ((MAX_ARTIFACT_BYTES + 2) // 3)
    for submitted in submitted_artifacts:
        artifact: dict[str, Any] = {
            "id": submitted.artifact_id,
            "kind": submitted.kind,
            "name": submitted.name,
            "media_type": submitted.media_type,
        }
        if submitted.kind in {"url", "email"}:
            artifact["submitted_value"] = submitted.submitted_value
            if submitted.kind == "email":
                artifact["content"] = submitted.submitted_value
        else:
            encoded = submitted.content_base64 or ""
            if len(encoded) > max_encoded_size:
                raise HTTPException(status_code=413, detail=f"Encoded content exceeds the {MAX_ARTIFACT_BYTES}-byte limit")
            try:
                content = base64.b64decode(encoded, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise HTTPException(status_code=422, detail="content_base64 must be valid standard base64") from exc
            if len(content) > MAX_ARTIFACT_BYTES:
                raise HTTPException(status_code=413, detail=f"Decoded content exceeds {MAX_ARTIFACT_BYTES} bytes")
            artifact["content"] = content
        try:
            bundle = _ANALYZERS[submitted.kind].analyze(artifact)
            bundles.append(bundle)
            # Route URLs extracted from QR, email, and file inputs through the
            # same URL analyzer used for directly submitted URL artifacts.
            embedded_urls = (
                bundle.features.get("qr.url_payloads", [])
                or bundle.features.get("email.urls", [])
                or bundle.features.get("file.contains_urls", [])
            )
            for index, value in enumerate(dict.fromkeys(embedded_urls)):
                url = str(value).strip()
                if not url.casefold().startswith(("http://", "https://", "www.")):
                    continue
                digest = hashlib.sha256(
                    f"{submitted.kind}:{submitted.artifact_id or ''}:{index}:{url}".encode("utf-8")
                ).hexdigest()[:32]
                bundles.append(URLAnalyzer().analyze({
                    "id": f"embedded-url-{digest}",
                    "kind": "url",
                    "submitted_value": url,
                }))
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    return bundles


@router.post("")
def correlate(payload: CorrelationRequest,
              max_depth: int = Query(default=6, ge=1, le=12),
              max_paths: int = Query(default=50, ge=1, le=500),
              db: Session = Depends(get_db)) -> dict[str, Any]:
    if payload.session_id and db.get(AnalysisSession, payload.session_id) is None:
        raise HTTPException(status_code=404, detail="Analysis session not found")
    bundles = analyze_submitted_artifacts(payload.artifacts)
    result = CrossInputCorrelator().correlate(bundles)
    graph = NetworkXGraphProcessor().process_correlation(result)
    response = graph.to_dict(max_depth=max_depth, max_paths=max_paths)
    if payload.session_id:
        response["evidence_persisted"] = {
            "session_id": payload.session_id,
            "record_count": persist_security_evidence(db, payload.session_id, bundles, result),
        }
    return response


@router.get("/synthetic-cases")
def synthetic_cases() -> dict[str, Any]:
    return {"cases": build_synthetic_correlation_cases()}
