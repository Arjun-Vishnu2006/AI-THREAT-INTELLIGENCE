"""Project novelty, IOC and graph evidence into a content-safe exposure feature set."""

from __future__ import annotations

from collections import Counter
from typing import Any

from app.novelty.types import NoveltySnapshot


def build_exposure_features(snapshot: NoveltySnapshot, assessment: dict[str, Any],
                            *, known_ioc_hit_count: int = 0,
                            indicator_count: int = 0) -> dict[str, str | int | float]:
    features: dict[str, str | int | float] = {
        "artifact.count": len(snapshot.feature_vectors),
        "relationship.count": len(snapshot.relationship_signatures),
        "relationship.unique_count": len(set(snapshot.relationship_signatures)),
        "path.count": len(snapshot.path_signatures),
        "indicator.count": indicator_count,
        "known_ioc.hit_count": known_ioc_hit_count,
        "knowledge.known": int(assessment.get("knowledge_status") == "KNOWN"),
        "knowledge.unknown": int(assessment.get("knowledge_status") == "UNKNOWN"),
        "anomaly.anomalous": int(assessment.get("anomaly_status") == "ANOMALOUS"),
        "novelty.novel": int(assessment.get("novelty_status") == "NOVEL"),
        "novelty.p_value": float((assessment.get("fusion") or {}).get("combined_p_value") or 1.0),
    }
    artifact_counts: Counter[str] = Counter()
    for vector in snapshot.feature_vectors:
        kind = str(vector.get("artifact.kind", "unknown"))
        artifact_counts[kind] += 1
        for key, value in vector.items():
            if key == "artifact.kind":
                continue
            if isinstance(value, (int, float)):
                features[f"artifact.max.{kind}.{key}"] = max(
                    float(value), float(features.get(f"artifact.max.{kind}.{key}", float("-inf")))
                )
            else:
                category = str(value).casefold()[:80]
                features[f"artifact.category.{kind}.{key}.{category}"] = 1
    for kind, count in artifact_counts.items():
        features[f"artifact.kind_count.{kind}"] = count

    relation_counts = Counter(snapshot.relationship_signatures)
    for signature, count in relation_counts.items():
        features[f"relationship.signature.{signature.casefold()}"] = count
    graph = snapshot.graph_features
    for key in (
        "graph.node_count", "graph.edge_count", "graph.weak_component_count",
        "graph.isolated_node_count", "graph.max_degree", "graph.average_degree",
        "graph.directed_density",
    ):
        value = graph.get(key)
        if isinstance(value, (int, float)):
            features[key] = float(value)
    for kind, count in (graph.get("graph.node_counts_by_type") or {}).items():
        features[f"graph.node_type.{str(kind).casefold()}"] = int(count)
    for kind, count in (graph.get("graph.edge_counts_by_type") or {}).items():
        features[f"graph.edge_type.{str(kind).casefold()}"] = int(count)
    for path in snapshot.path_signatures:
        features[f"path.signature.{path.casefold()}"] = 1

    temporal = snapshot.temporal_features or {}
    features["temporal.reference_available"] = int(snapshot.temporal_features is not None)
    for key, value in temporal.items():
        features[f"temporal.{key}"] = float(value)
    for name, layer in (assessment.get("layers") or {}).items():
        features[f"novelty.layer.{name}.p_value"] = float(layer.get("p_value", 1.0))
        features[f"novelty.layer.{name}.nonconformity"] = float(layer.get("nonconformity", 0.0))
        evidence = layer.get("evidence") or []
        features[f"novelty.layer.{name}.evidence_count"] = len(evidence)
        features[f"novelty.layer.{name}.unseen_evidence_count"] = sum(
            1 for item in evidence
            if item.get("baseline_occurrences") == 0
            or item.get("graph_signature_seen_in_baseline") is False
            or "unseen" in str(item.get("observation", ""))
            or "not present in baseline" in str(item.get("reference", ""))
        )
    return features


def explain_exposure_features(features: dict[str, str | int | float],
                              assessment: dict[str, Any], limit: int = 12) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for name, layer in (assessment.get("layers") or {}).items():
        for item in (layer.get("evidence") or [])[:2]:
            evidence.append({"source_layer": name, "evidence": item})
    if len(evidence) >= limit:
        return evidence[:limit]
    for key, value in sorted(features.items()):
        if key.startswith(("knowledge.", "novelty.", "anomaly.", "temporal.")) and value:
            evidence.append({"feature": key, "value": value})
        elif key.startswith(("relationship.signature.", "path.signature.")):
            evidence.append({"feature": key, "value": value})
        elif key.startswith("graph.") and value:
            evidence.append({"feature": key, "value": value})
    return evidence[:limit]
