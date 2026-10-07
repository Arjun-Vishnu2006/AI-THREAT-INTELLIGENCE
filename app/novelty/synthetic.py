"""Deterministic synthetic baseline and deliberately unseen novelty snapshots."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from app.novelty.types import NoveltySnapshot

_SEEN_PATTERNS = (
    "url_domain", "email_url_domain", "file_url_domain", "qr_url_domain",
    "email_file_qr_url_domain",
)


@dataclass(frozen=True)
class SyntheticNoveltyCase:
    sample_id: str
    campaign_id: str
    label: str
    snapshot: NoveltySnapshot
    description: str


def _seen_snapshot(pattern: str, index: int) -> NoveltySnapshot:
    suffix = f"{index:03d}"
    vectors: list[dict[str, str | int | float]] = []
    relations: list[str] = []
    path = ""
    additional_paths: tuple[str, ...] = ()
    graph_nodes: dict[str, int]
    graph_edges: dict[str, int]
    if pattern == "url_domain":
        vectors = [{"artifact.kind": "url", "url.scheme": "https", "url.length": 30 + index % 7,
                    "url.host_label_count": 3, "url.subdomain_label_count": 1,
                    "url.path_length": 8 + index % 5, "url.query_parameter_count": index % 2,
                    "url.percent_escape_count": 0, "url.has_userinfo": 0}]
        relations = ["URL|BELONGS_TO|DOMAIN"]
        path = ""
        graph_nodes, graph_edges = {"artifact": 1, "url": 1, "domain": 1}, {"CONTAINS": 2, "BELONGS_TO": 1}
    elif pattern == "email_url_domain":
        vectors = [
            {"artifact.kind": "email", "email.links.count": 1 + index % 2,
             "email.attachments.count": 0, "email.urgency_matches.count": 0,
             "email.reply_domain_differs": 0, "email.body_character_count": 90 + index % 11},
            {"artifact.kind": "url", "url.scheme": "https", "url.length": 35 + index % 6,
             "url.host_label_count": 3, "url.subdomain_label_count": 1,
             "url.path_length": 9, "url.query_parameter_count": 0,
             "url.percent_escape_count": 0, "url.has_userinfo": 0},
        ]
        relations = ["EMAIL|LINKS_TO|URL", "EMAIL|BELONGS_TO|DOMAIN", "URL|BELONGS_TO|DOMAIN"]
        path = "EMAIL-[LINKS_TO]->URL"
        graph_nodes, graph_edges = {"artifact": 2, "email": 1, "url": 1, "domain": 2}, {"CONTAINS": 4, "LINKS_TO": 2, "BELONGS_TO": 2}
    elif pattern == "file_url_domain":
        vectors = [
            {"artifact.kind": "file", "file.detected_type": "text", "file.suffix": ".txt",
             "file.size_bytes": 350 + index % 17, "file.declared_mime_mismatch": 0,
             "file.script_like_extension": 0},
            {"artifact.kind": "url", "url.scheme": "https", "url.length": 32 + index % 8,
             "url.host_label_count": 3, "url.subdomain_label_count": 1,
             "url.path_length": 7, "url.query_parameter_count": 0,
             "url.percent_escape_count": 0, "url.has_userinfo": 0},
        ]
        relations = ["FILE|HAS_HASH|FILE_HASH_SHA256", "FILE|LINKS_TO|URL", "URL|BELONGS_TO|DOMAIN"]
        path = ""
        graph_nodes, graph_edges = {"artifact": 2, "file_hash_sha256": 1, "url": 1, "domain": 1}, {"HAS_HASH": 1, "CONTAINS": 4, "LINKS_TO": 1, "BELONGS_TO": 1}
    elif pattern == "qr_url_domain":
        vectors = [
            {"artifact.kind": "qr", "qr.decoded_count": 1, "qr.has_url_payload": 1},
            {"artifact.kind": "url", "url.scheme": "https", "url.length": 34 + index % 8,
             "url.host_label_count": 3, "url.subdomain_label_count": 1,
             "url.path_length": 8, "url.query_parameter_count": 0,
             "url.percent_escape_count": 0, "url.has_userinfo": 0},
        ]
        relations = ["QR|DECODED_TO|URL", "QR_PAYLOAD|DECODED_TO|URL", "URL|BELONGS_TO|DOMAIN"]
        path = "QR-[DECODED_TO]->URL"
        graph_nodes, graph_edges = {"artifact": 2, "qr_payload": 1, "url": 1, "domain": 1}, {"CONTAINS": 5, "DECODED_TO": 2, "BELONGS_TO": 1}
    else:
        vectors = [
            {"artifact.kind": "email", "email.links.count": 0,
             "email.attachments.count": 1, "email.urgency_matches.count": 0,
             "email.reply_domain_differs": 0, "email.body_character_count": 100},
            {"artifact.kind": "file", "file.detected_type": "unknown", "file.suffix": ".bin",
             "file.size_bytes": 80 + index % 9, "file.declared_mime_mismatch": 0,
             "file.script_like_extension": 0},
            {"artifact.kind": "qr", "qr.decoded_count": 1, "qr.has_url_payload": 1},
            {"artifact.kind": "url", "url.scheme": "https", "url.length": 40 + index % 9,
             "url.host_label_count": 3, "url.subdomain_label_count": 1,
             "url.path_length": 12, "url.query_parameter_count": 1,
             "url.percent_escape_count": 0, "url.has_userinfo": 0},
        ]
        relations = ["EMAIL|ATTACHED_TO|FILE", "EMAIL|ATTACHED_TO|QR",
                     "FILE|HAS_HASH|FILE_HASH_SHA256", "FILE|RELATED_TO|QR",
                     "QR|DECODED_TO|URL", "QR_PAYLOAD|DECODED_TO|URL",
                     "EMAIL|BELONGS_TO|DOMAIN", "URL|BELONGS_TO|DOMAIN"]
        path = "EMAIL-[ATTACHED_TO]->FILE-[RELATED_TO]->QR-[DECODED_TO]->URL"
        additional_paths = ("EMAIL-[ATTACHED_TO]->QR-[DECODED_TO]->URL",)
        graph_nodes, graph_edges = {"artifact": 4, "email": 1, "file_hash_sha256": 1, "qr_payload": 1,
                                    "url": 1, "domain": 2}, {"CONTAINS": 7, "ATTACHED_TO": 3,
                                    "RELATED_TO": 1, "DECODED_TO": 3, "BELONGS_TO": 2, "HAS_HASH": 1}
    graph_features = {
        "graph.node_counts_by_type": graph_nodes,
        "graph.edge_counts_by_type": graph_edges,
        "graph.node_count": sum(graph_nodes.values()),
        "graph.edge_count": sum(graph_edges.values()),
        "graph.weak_component_count": 1,
        "graph.max_degree": min(6, 2 + len(relations)),
    }
    graph_signature = json.dumps({"nodes": graph_nodes, "edges": graph_edges, "components": 1},
                                 sort_keys=True, separators=(",", ":"))
    entities = tuple(f"domain:site-{suffix}-{n}.example.test" for n in range(1))
    return NoveltySnapshot(
        feature_vectors=tuple(vectors), relationship_signatures=tuple(relations),
        graph_signature=graph_signature, graph_features=graph_features,
        path_signatures=((path,) + additional_paths) if path else (), entity_keys=entities,
        temporal_features={"new_entity_fraction": (index % 3) / 20,
                           "new_relationship_fraction": (index % 2) / 20,
                           "relationship_frequency_increase_fraction": 0.0},
    )


def _unseen_snapshot(index: int, pattern: str) -> NoveltySnapshot:
    # The unseen case is a coherent four-artifact chain with an additional
    # resolution edge and extreme feature values on existing artifacts.
    base = _seen_snapshot("email_file_qr_url_domain", index)
    vectors = list(base.feature_vectors)
    for vector in vectors:
        if vector.get("artifact.kind") == "url":
            vector.update({"url.scheme": "custom-protocol", "url.length": 900 + index,
                           "url.host_label_count": 11, "url.subdomain_label_count": 9,
                           "url.path_length": 420, "url.query_parameter_count": 35,
                           "url.percent_escape_count": 44, "url.has_userinfo": 1})
        elif vector.get("artifact.kind") == "email":
            vector.update({"email.links.count": 18, "email.attachments.count": 4,
                           "email.urgency_matches.count": 13, "email.reply_domain_differs": 1,
                           "email.body_character_count": 15000})
    new_relation = "DOMAIN|RESOLVES_TO|IP"
    relations = tuple(base.relationship_signatures) + (
        new_relation,
    )
    nodes = dict(base.graph_features["graph.node_counts_by_type"])
    nodes["ip"] = 1
    edges = dict(base.graph_features["graph.edge_counts_by_type"])
    edges["RESOLVES_TO"] = 1
    graph_features = {
        "graph.node_counts_by_type": nodes, "graph.edge_counts_by_type": edges,
        "graph.node_count": sum(nodes.values()), "graph.edge_count": sum(edges.values()),
        "graph.weak_component_count": 1, "graph.max_degree": 11,
    }
    graph_signature = json.dumps({"nodes": nodes, "edges": edges, "components": 1},
                                 sort_keys=True, separators=(",", ":"))
    return NoveltySnapshot(
        feature_vectors=tuple(vectors), relationship_signatures=relations,
        graph_signature=graph_signature, graph_features=graph_features,
        path_signatures=base.path_signatures,
        entity_keys=(f"ip:192.0.2.{index % 200 + 1}", f"domain:new-{index}.example.test"),
        temporal_features={"new_entity_fraction": 0.95,
                           "new_relationship_fraction": 1.0,
                           "relationship_frequency_increase_fraction": 1.0},
        temporal_evidence=({"kind": "synthetic_first_seen_entity", "key": f"ip:192.0.2.{index % 200 + 1}"},),
    )


def build_synthetic_novelty_corpus(seed: int = 2026) -> dict[str, list[SyntheticNoveltyCase]]:
    # Prefixes are disjoint across partitions; no campaign/group can leak between them.
    result: dict[str, list[SyntheticNoveltyCase]] = {"train": [], "calibration": [], "test_seen": [], "test_unseen": []}
    sizes = {"train": 100, "calibration": 120, "test_seen": 60, "test_unseen": 60}
    for split, size in sizes.items():
        for index in range(size):
            if split == "test_unseen":
                pattern = "new_multi_input_chain_and_extreme_features"
                snapshot = _unseen_snapshot(index, pattern)
                label = "unseen"
                description = "Known multi-input chain with an unseen domain-to-IP resolution edge and feature extremes."
            else:
                pattern = _SEEN_PATTERNS[index % len(_SEEN_PATTERNS)]
                snapshot = _seen_snapshot(pattern, index)
                label = "baseline"
                description = f"Held-out instance of reference motif {pattern}."
            prefix = f"{split}-seed-{seed}"
            result[split].append(SyntheticNoveltyCase(
                sample_id=f"{prefix}-sample-{index:04d}",
                campaign_id=f"{prefix}-campaign-{pattern}-{index:04d}",
                label=label, snapshot=snapshot, description=description,
            ))
    return result


def corpus_fingerprint(corpus: dict[str, list[SyntheticNoveltyCase]]) -> str:
    records = []
    for split, cases in corpus.items():
        records.extend({"split": split, "sample_id": item.sample_id,
                        "campaign_id": item.campaign_id, "label": item.label,
                        "features": item.snapshot.feature_vectors,
                        "relationships": item.snapshot.relationship_signatures,
                        "graph_signature": item.snapshot.graph_signature,
                        "paths": item.snapshot.path_signatures,
                        "temporal": item.snapshot.temporal_features}
                       for item in cases)
    blob = json.dumps(records, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()
