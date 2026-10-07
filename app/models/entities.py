from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import DateTime, ForeignKey, Index, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


def new_id() -> str:
    return str(uuid4())


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)


class AnalysisSession(TimestampMixin, Base):
    __tablename__ = "analysis_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    status: Mapped[str] = mapped_column(String(32), default="created", index=True)
    source: Mapped[str | None] = mapped_column(String(128), nullable=True)
    submitted_by: Mapped[str | None] = mapped_column(String(256), nullable=True)
    metadata_json: Mapped[dict] = mapped_column("metadata", JSON, default=dict)
    artifacts: Mapped[list["Artifact"]] = relationship(back_populates="session", cascade="all, delete-orphan")
    findings: Mapped[list["Finding"]] = relationship(back_populates="session", cascade="all, delete-orphan")
    alerts: Mapped[list["Alert"]] = relationship(back_populates="session", cascade="all, delete-orphan")


class Artifact(TimestampMixin, Base):
    __tablename__ = "artifacts"
    __table_args__ = (Index("ix_artifacts_session_kind", "session_id", "kind"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    session_id: Mapped[str] = mapped_column(ForeignKey("analysis_sessions.id", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column(String(32), index=True)  # url, email, qr, file
    name: Mapped[str | None] = mapped_column(String(512), nullable=True)
    media_type: Mapped[str | None] = mapped_column(String(255), nullable=True)
    content_hash: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    storage_ref: Mapped[str | None] = mapped_column(Text, nullable=True)
    submitted_value: Mapped[str | None] = mapped_column(Text, nullable=True)
    attributes: Mapped[dict] = mapped_column(JSON, default=dict)
    session: Mapped[AnalysisSession] = relationship(back_populates="artifacts")


class Indicator(TimestampMixin, Base):
    __tablename__ = "indicators"
    __table_args__ = (UniqueConstraint("type", "normalized_value", name="uq_indicator_type_value"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    type: Mapped[str] = mapped_column(String(48), index=True)  # domain, ip, hash, email, url, ...
    value: Mapped[str] = mapped_column(Text)
    normalized_value: Mapped[str] = mapped_column(Text, index=True)
    first_seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    attributes: Mapped[dict] = mapped_column(JSON, default=dict)


class Relationship(TimestampMixin, Base):
    __tablename__ = "relationships"
    __table_args__ = (
        UniqueConstraint("source_indicator_id", "target_indicator_id", "kind", name="uq_relationship_edge"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    source_indicator_id: Mapped[str] = mapped_column(ForeignKey("indicators.id", ondelete="CASCADE"), index=True)
    target_indicator_id: Mapped[str] = mapped_column(ForeignKey("indicators.id", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column(String(64), index=True)
    session_id: Mapped[str | None] = mapped_column(ForeignKey("analysis_sessions.id", ondelete="SET NULL"), nullable=True, index=True)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    attributes: Mapped[dict] = mapped_column(JSON, default=dict)


class Finding(TimestampMixin, Base):
    __tablename__ = "findings"
    __table_args__ = (Index("ix_findings_session_category", "session_id", "category"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    session_id: Mapped[str] = mapped_column(ForeignKey("analysis_sessions.id", ondelete="CASCADE"), index=True)
    category: Mapped[str] = mapped_column(String(64), index=True)
    title: Mapped[str] = mapped_column(String(512))
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="observed")
    confidence: Mapped[float | None] = mapped_column(nullable=True)
    evidence: Mapped[dict] = mapped_column(JSON, default=dict)
    producer: Mapped[str | None] = mapped_column(String(128), nullable=True)
    session: Mapped[AnalysisSession] = relationship(back_populates="findings")


class Alert(TimestampMixin, Base):
    __tablename__ = "alerts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    session_id: Mapped[str] = mapped_column(ForeignKey("analysis_sessions.id", ondelete="CASCADE"), index=True)
    finding_id: Mapped[str | None] = mapped_column(ForeignKey("findings.id", ondelete="SET NULL"), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="open", index=True)
    severity: Mapped[str | None] = mapped_column(String(32), nullable=True)
    title: Mapped[str] = mapped_column(String(512))
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    triggered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    session: Mapped[AnalysisSession] = relationship(back_populates="alerts")


class EvidenceLink(Base):
    """Directed provenance edge connecting source evidence to derived evidence."""

    __tablename__ = "evidence_links"
    __table_args__ = (
        Index("ix_evidence_links_session_source", "session_id", "source_evidence_id"),
        Index("ix_evidence_links_session_target", "session_id", "target_evidence_id"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    session_id: Mapped[str] = mapped_column(ForeignKey("analysis_sessions.id", ondelete="CASCADE"), index=True)
    source_evidence_id: Mapped[str] = mapped_column(ForeignKey("security_evidence.id", ondelete="CASCADE"), index=True)
    target_evidence_id: Mapped[str] = mapped_column(ForeignKey("security_evidence.id", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column(String(64), index=True)
    attributes: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class AlertEvent(Base):
    """Append-only alert lifecycle event history."""

    __tablename__ = "alert_events"
    __table_args__ = (Index("ix_alert_events_alert_created", "alert_id", "created_at"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    alert_id: Mapped[str] = mapped_column(ForeignKey("alerts.id", ondelete="CASCADE"), index=True)
    event_type: Mapped[str] = mapped_column(String(64), index=True)
    previous_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    status: Mapped[str] = mapped_column(String(32), index=True)
    actor: Mapped[str | None] = mapped_column(String(256), nullable=True)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class ModelMetadata(TimestampMixin, Base):
    __tablename__ = "model_metadata"
    __table_args__ = (UniqueConstraint("name", "version", name="uq_model_name_version"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(String(256), index=True)
    version: Mapped[str] = mapped_column(String(128))
    task: Mapped[str] = mapped_column(String(128), index=True)
    artifact_ref: Mapped[str | None] = mapped_column(Text, nullable=True)
    parameters: Mapped[dict] = mapped_column(JSON, default=dict)
    provenance: Mapped[dict] = mapped_column(JSON, default=dict)


class ExperimentMetadata(TimestampMixin, Base):
    __tablename__ = "experiment_metadata"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(String(256), index=True)
    version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="registered")
    configuration: Mapped[dict] = mapped_column(JSON, default=dict)
    results: Mapped[dict] = mapped_column(JSON, default=dict)


class SecurityEvidence(TimestampMixin, Base):
    """Durable, structured evidence; raw submitted artifact bytes/content are never stored here."""

    __tablename__ = "security_evidence"
    __table_args__ = (Index("ix_security_evidence_session_kind", "session_id", "kind"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    session_id: Mapped[str] = mapped_column(ForeignKey("analysis_sessions.id", ondelete="CASCADE"), index=True)
    artifact_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    kind: Mapped[str] = mapped_column(String(64), index=True)
    summary: Mapped[str] = mapped_column(Text)
    producer: Mapped[str | None] = mapped_column(String(128), nullable=True)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
