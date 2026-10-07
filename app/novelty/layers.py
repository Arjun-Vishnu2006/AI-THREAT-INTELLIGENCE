"""Interpretable nonconformity layers used by the novelty fusion engine."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from statistics import median
from typing import Any, Protocol, Sequence

from app.novelty.types import NoveltySnapshot


@dataclass(frozen=True)
class LayerScore:
    name: str
    nonconformity: float
    evidence: tuple[dict[str, Any], ...]


class NoveltyLayer(Protocol):
    name: str

    def fit(self, baseline: Sequence[NoveltySnapshot]) -> None: ...

    def score(self, snapshot: NoveltySnapshot) -> LayerScore | None: ...


class IndividualAnomalyLayer:
    name = "individual_anomaly"

    def __init__(self) -> None:
        self.numeric_profiles: dict[tuple[str, str], tuple[float, float, float, float]] = {}
        self.categorical_values: dict[tuple[str, str], set[str]] = {}

    def fit(self, baseline: Sequence[NoveltySnapshot]) -> None:
        values: dict[tuple[str, str], list[float]] = {}
        categories: dict[tuple[str, str], set[str]] = {}
        for sample in baseline:
            for vector in sample.feature_vectors:
                kind = str(vector.get("artifact.kind", "unknown"))
                for key, value in vector.items():
                    if key == "artifact.kind":
                        continue
                    identity = (kind, key)
                    if isinstance(value, (int, float)):
                        values.setdefault(identity, []).append(float(value))
                    else:
                        categories.setdefault(identity, set()).add(str(value))
        for identity, observations in values.items():
            center = median(observations)
            deviations = [abs(value - center) for value in observations]
            mad = median(deviations)
            ordered = sorted(observations)
            q1 = ordered[len(ordered) // 4]
            q3 = ordered[(3 * len(ordered)) // 4]
            scale = max(1.4826 * mad, (q3 - q1) / 1.349, 1.0)
            self.numeric_profiles[identity] = (center, scale, min(observations), max(observations))
        self.categorical_values = categories

    def score(self, snapshot: NoveltySnapshot) -> LayerScore | None:
        deviations = []
        for vector in snapshot.feature_vectors:
            kind = str(vector.get("artifact.kind", "unknown"))
            for key, value in vector.items():
                if key == "artifact.kind":
                    continue
                identity = (kind, key)
                if isinstance(value, (int, float)):
                    profile = self.numeric_profiles.get(identity)
                    if profile is None:
                        deviations.append((4.0, {"feature": key, "artifact_kind": kind,
                                                  "observed": value, "reference": "feature not present in baseline"}))
                        continue
                    center, scale, low, high = profile
                    z = abs(float(value) - center) / scale
                    if z > 0:
                        deviations.append((z, {"feature": key, "artifact_kind": kind,
                                               "observed": value, "reference_median": center,
                                               "reference_range": [low, high], "robust_deviation": z}))
                else:
                    allowed = self.categorical_values.get(identity, set())
                    if str(value) not in allowed:
                        deviations.append((4.0, {"feature": key, "artifact_kind": kind,
                                                  "observed": value,
                                                  "reference": "category not present in baseline"}))
        if not snapshot.feature_vectors:
            return None
        deviations.sort(key=lambda row: row[0], reverse=True)
        score = deviations[0][0] if deviations else 0.0
        evidence = tuple(item for _, item in deviations[:5])
        if not evidence:
            evidence = ({"summary": "Allowlisted features are within the fitted reference ranges."},)
        return LayerScore(self.name, float(score), evidence)


class RelationshipAnomalyLayer:
    name = "relationship_anomaly"

    def __init__(self) -> None:
        self.counts: Counter[str] = Counter()
        self.total = 0

    def fit(self, baseline: Sequence[NoveltySnapshot]) -> None:
        self.counts.clear()
        for sample in baseline:
            self.counts.update(sample.relationship_signatures)
        self.total = sum(self.counts.values())

    def score(self, snapshot: NoveltySnapshot) -> LayerScore | None:
        if not snapshot.relationship_signatures:
            return None
        vocab = max(1, len(self.counts))
        records = []
        for signature in set(snapshot.relationship_signatures):
            count = self.counts[signature]
            score = -math.log((count + 0.5) / (self.total + 0.5 * (vocab + 1)))
            records.append((score, {"relationship_signature": signature,
                                   "baseline_occurrences": count,
                                   "observation": "unseen relationship" if count == 0 else "observed relationship"}))
        records.sort(key=lambda row: row[0], reverse=True)
        return LayerScore(self.name, float(records[0][0]), tuple(item for _, item in records[:10]))


class GraphStructureNoveltyLayer:
    name = "graph_structure"

    def __init__(self) -> None:
        self.counts: Counter[str] = Counter()
        self.total = 0

    def fit(self, baseline: Sequence[NoveltySnapshot]) -> None:
        self.counts = Counter(sample.graph_signature for sample in baseline)
        self.total = sum(self.counts.values())

    def score(self, snapshot: NoveltySnapshot) -> LayerScore | None:
        if not snapshot.graph_signature:
            return None
        count = self.counts[snapshot.graph_signature]
        score = -math.log((count + 0.5) / (self.total + 0.5 * (len(self.counts) + 1)))
        return LayerScore(self.name, float(score), ({
            "graph_signature_seen_in_baseline": count > 0,
            "baseline_occurrences": count,
            "graph_features": snapshot.graph_features,
        },))


class AttackPathNoveltyLayer:
    name = "attack_chain_path"

    def __init__(self) -> None:
        self.counts: Counter[str] = Counter()
        self.total = 0

    def fit(self, baseline: Sequence[NoveltySnapshot]) -> None:
        self.counts.clear()
        for sample in baseline:
            self.counts.update(sample.path_signatures)
        self.total = sum(self.counts.values())

    def score(self, snapshot: NoveltySnapshot) -> LayerScore | None:
        if not snapshot.path_signatures:
            return None
        vocab = max(1, len(self.counts))
        records = []
        for signature in set(snapshot.path_signatures):
            count = self.counts[signature]
            score = -math.log((count + 0.5) / (self.total + 0.5 * (vocab + 1)))
            records.append((score, {"path_signature": signature,
                                   "baseline_occurrences": count,
                                   "observation": "unseen path sequence" if count == 0 else "observed path sequence"}))
        records.sort(key=lambda row: row[0], reverse=True)
        return LayerScore(self.name, float(records[0][0]), tuple(item for _, item in records[:10]))


class TemporalNoveltyLayer:
    name = "temporal"

    def score(self, snapshot: NoveltySnapshot) -> LayerScore | None:
        if snapshot.temporal_features is None:
            return None
        features = snapshot.temporal_features
        nonconformity = max(
            features.get("new_entity_fraction", 0.0),
            features.get("new_relationship_fraction", 0.0),
            features.get("relationship_frequency_increase_fraction", 0.0),
        )
        return LayerScore(self.name, float(nonconformity), snapshot.temporal_evidence or ({
            "temporal_features": features,
        },))
