from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from app.services.correlation import CorrelationResult
from app.services.feature_types import FeatureBundle
from app.services.threat_graph import ThreatGraph
from app.ml.features import extract_ml_features
from app.ml.android_feature_schema import ANDROID_FEATURE_KEY, extract_android_feature_row


@dataclass(frozen=True)
class TemporalHistory:
    entity_counts: Mapping[str, int]
    relationship_counts: Mapping[str, int]


@dataclass(frozen=True)
class NoveltySnapshot:
    feature_vectors: tuple[dict[str, str | int | float], ...]
    relationship_signatures: tuple[str, ...]
    graph_signature: str
    graph_features: dict[str, Any]
    path_signatures: tuple[str, ...]
    entity_keys: tuple[str, ...]
    temporal_features: dict[str, float] | None = None
    temporal_evidence: tuple[dict[str, Any], ...] = ()
    evidence_sources: dict[str, Any] = field(default_factory=dict)
    android_feature_vector: dict[str, float] | None = None
    android_feature_extraction: dict[str, Any] | None = None


def relationship_signature(source_type: str, kind: str, target_type: str) -> str:
    return f"{source_type.upper()}|{kind.upper()}|{target_type.upper()}"


def snapshot_from_analysis(bundles: Sequence[FeatureBundle], correlation: CorrelationResult,
                           graph: ThreatGraph, history: TemporalHistory | None = None,
                           max_paths: int = 100) -> NoveltySnapshot:
    vectors = tuple(extract_ml_features(bundle) for bundle in bundles)
    android_rows = [
        extract_android_feature_row(bundle.features[ANDROID_FEATURE_KEY])
        for bundle in bundles
        if ANDROID_FEATURE_KEY in bundle.features
    ]
    if len(android_rows) > 1:
        raise ValueError("A NoveltySnapshot accepts Android autoencoder features from one APK at a time")
    android_extractions = [
        bundle.features["android.apk_analysis"]
        for bundle in bundles if "android.apk_analysis" in bundle.features
    ]
    if len(android_extractions) > 1:
        raise ValueError("A NoveltySnapshot accepts APK extraction details from one APK at a time")
    entity_index = {entity.id: entity for entity in correlation.entities}
    relation_sigs: list[str] = []
    for edge in correlation.relationships:
        source, target = entity_index[edge.source_id], entity_index[edge.target_id]
        # CONTAINS edges are ubiquitous artifact bookkeeping and add little signal.
        # For semantic edges, represent artifact nodes by their concrete input kind.
        if edge.kind.upper() == "CONTAINS":
            continue
        source_type = source.attributes.get("artifact_kind", source.type)
        target_type = target.attributes.get("artifact_kind", target.type)
        relation_sigs.append(relationship_signature(source_type, edge.kind, target_type))
    path_signatures: list[str] = []
    path_records = graph.paths(max_depth=8, max_paths=max_paths)
    for path in path_records:
        node_labels = [
            str(node.get("attributes", {}).get("artifact_kind", node.get("type", "entity"))).upper()
            for node in path["nodes"]
        ]
        edge_labels = ["/".join(edge.get("relationships", [])) for edge in path["edges"]]
        pieces: list[str] = [node_labels[0]] if node_labels else []
        for rels, label in zip(edge_labels, node_labels[1:]):
            pieces.extend((f"-{rels}->", label))
        path_signatures.append("".join(pieces))
    graph_features = graph.features()
    node_types = graph_features.get("graph.node_counts_by_type", {})
    edge_types = graph_features.get("graph.edge_counts_by_type", {})
    graph_signature = json.dumps({
        "nodes": node_types,
        "edges": edge_types,
        "components": graph_features.get("graph.weak_component_count", 0),
    }, sort_keys=True, separators=(",", ":"))
    entity_keys = tuple(sorted(
        f"{entity.type}:{entity.normalized_value}"
        for entity in correlation.entities if entity.type != "artifact"
    ))
    temporal_features = None
    temporal_evidence: list[dict[str, Any]] = []
    if history is not None:
        current_entities = set(entity_keys)
        current_relations = set(relation_sigs)
        new_entities = sorted(key for key in current_entities if key not in history.entity_counts)
        new_relations = sorted(key for key in current_relations if key not in history.relationship_counts)
        entity_fraction = len(new_entities) / max(1, len(current_entities))
        relation_fraction = len(new_relations) / max(1, len(current_relations))
        frequency_increases = [
            key for key in current_relations
            if relation_sigs.count(key) > history.relationship_counts.get(key, 0)
        ]
        frequency_increase_fraction = len(frequency_increases) / max(1, len(current_relations))
        temporal_features = {
            "new_entity_fraction": entity_fraction,
            "new_relationship_fraction": relation_fraction,
            "relationship_frequency_increase_fraction": frequency_increase_fraction,
            "new_entity_count": float(len(new_entities)),
            "new_relationship_count": float(len(new_relations)),
        }
        temporal_evidence = [
            {"kind": "first_seen_entity", "key": key, "historical_count": 0}
            for key in new_entities
        ] + [
            {"kind": "first_seen_relationship", "key": key, "historical_count": 0}
            for key in new_relations
        ]
        # Compare current edge frequency with persisted prior observations.
        for key in sorted(current_relations & history.relationship_counts.keys()):
            temporal_evidence.append({
                "kind": "relationship_frequency", "key": key,
                "historical_count": history.relationship_counts[key],
                "current_count": relation_sigs.count(key),
            })
    return NoveltySnapshot(
        feature_vectors=vectors,
        relationship_signatures=tuple(relation_sigs),
        graph_signature=graph_signature,
        graph_features=graph_features,
        path_signatures=tuple(path_signatures),
        entity_keys=entity_keys,
        temporal_features=temporal_features,
        temporal_evidence=tuple(temporal_evidence),
        evidence_sources={
            "artifact_ids": [bundle.artifact_id for bundle in bundles if bundle.artifact_id],
            "path_count": len(path_records),
            "relationship_count": len(relation_sigs),
            "autoencoder_feature_status": (
                "available" if android_rows else (
                    "unavailable_incomplete_feature_schema"
                    if android_extractions and not android_extractions[0].get(
                        "autoencoder_vector_available", False
                    ) and android_extractions[0].get("status") == "partial"
                    else "unavailable_apk_analysis"
                    if android_extractions
                    else "unavailable_missing_android_feature_schema"
                )
            ),
        },
        android_feature_vector=android_rows[0] if android_rows else None,
        android_feature_extraction=android_extractions[0] if android_extractions else None,
    )
