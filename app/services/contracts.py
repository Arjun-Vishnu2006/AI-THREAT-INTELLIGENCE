"""Stable extension points for later analysis phases. Implementations are intentionally absent."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol, Sequence

if TYPE_CHECKING:
    from app.services.feature_types import FeatureBundle


@dataclass(frozen=True)
class AnalysisContext:
    session_id: str
    observed_at: datetime
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Evidence:
    kind: str
    summary: str
    references: tuple[str, ...] = ()
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FindingResult:
    category: str
    title: str
    description: str | None = None
    confidence: float | None = None
    evidence: tuple[Evidence, ...] = ()
    producer: str | None = None


@dataclass(frozen=True)
class IndicatorResult:
    type: str
    value: str
    normalized_value: str
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RelationshipResult:
    source: IndicatorResult
    target: IndicatorResult
    kind: str
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExposureResult:
    category: str
    description: str
    confidence: float | None = None
    evidence: tuple[Evidence, ...] = ()


class Analyzer(Protocol):
    name: str
    supported_kinds: frozenset[str]

    def analyze(self, artifact: dict[str, Any], context: AnalysisContext) -> "FeatureBundle": ...


class CrossInputCorrelator(Protocol):
    def correlate(self, artifacts: Sequence["FeatureBundle"], context: AnalysisContext) -> Any: ...


class GraphProcessor(Protocol):
    def process(self, indicators: Sequence[IndicatorResult], relationships: Sequence[RelationshipResult], context: AnalysisContext) -> Any: ...


class AnomalyDetector(Protocol):
    def detect(self, graph: Any, context: AnalysisContext) -> Sequence[FindingResult]: ...


class NoveltyFuser(Protocol):
    def fuse(self, findings: Sequence[FindingResult], context: AnalysisContext) -> Sequence[FindingResult]: ...


class ExposurePredictor(Protocol):
    def predict(self, findings: Sequence[FindingResult], context: AnalysisContext) -> Sequence[ExposureResult]: ...


class EvidenceGenerator(Protocol):
    def generate(self, finding: FindingResult, context: AnalysisContext) -> Sequence[Evidence]: ...
