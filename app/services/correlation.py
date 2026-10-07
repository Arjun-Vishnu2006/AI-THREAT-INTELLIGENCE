"""Deterministic cross-input normalization and relationship construction."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Sequence

from app.services.contracts import IndicatorResult, RelationshipResult
from app.services.entities import NormalizedEntity, artifact_entity, normalize_indicator
from app.services.feature_types import FeatureBundle

_ALLOWED_RELATIONSHIPS = {
    "CONTAINS", "ATTACHED_TO", "DECODED_TO", "LINKS_TO", "BELONGS_TO",
    "RESOLVES_TO", "HAS_HASH", "RELATED_TO",
}
_RELATIONSHIP_ALIASES = {
    "ENCODES": "DECODED_TO", "DECODED_TO": "DECODED_TO", "LINKS_TO": "LINKS_TO",
    "CONTAINS": "CONTAINS", "ATTACHED_TO": "ATTACHED_TO", "BELONGS_TO": "BELONGS_TO",
    "RESOLVES_TO": "RESOLVES_TO", "HAS_HASH": "HAS_HASH", "RELATED_TO": "RELATED_TO",
}


@dataclass(frozen=True)
class ThreatRelationship:
    source_id: str
    target_id: str
    kind: str
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"source_id": self.source_id, "target_id": self.target_id,
                "kind": self.kind, "attributes": self.attributes}


@dataclass(frozen=True)
class CorrelationResult:
    entities: tuple[NormalizedEntity, ...]
    relationships: tuple[ThreatRelationship, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"entities": [entity.to_dict() for entity in self.entities],
                "relationships": [rel.to_dict() for rel in self.relationships]}


class CrossInputCorrelator:
    """Merge equal normalized indicators and add evidence-backed typed edges."""

    def correlate(self, artifacts: Sequence[FeatureBundle], context: Any = None) -> CorrelationResult:
        entities: dict[str, NormalizedEntity] = {}
        relationships: dict[tuple[str, str, str], ThreatRelationship] = {}
        artifact_nodes: list[tuple[FeatureBundle, NormalizedEntity]] = []
        bundle_indicators: dict[int, dict[tuple[str, str], NormalizedEntity]] = {}
        hash_to_artifacts: dict[str, list[NormalizedEntity]] = {}
        url_to_artifacts: dict[str, list[NormalizedEntity]] = {}

        def remember(entity: NormalizedEntity) -> NormalizedEntity:
            entities.setdefault(entity.id, entity)
            return entities[entity.id]

        def add(source: NormalizedEntity, target: NormalizedEntity, kind: str,
                attributes: dict[str, Any] | None = None) -> None:
            kind = _RELATIONSHIP_ALIASES.get(kind.upper().replace("-", "_"), "RELATED_TO")
            if kind not in _ALLOWED_RELATIONSHIPS:
                kind = "RELATED_TO"
            source, target = remember(source), remember(target)
            key = (source.id, target.id, kind)
            relationships.setdefault(key, ThreatRelationship(source.id, target.id, kind, attributes or {}))

        # Create stable artifact and canonical indicator nodes, then retain membership edges.
        for bundle in artifacts:
            source_id = bundle.artifact_id
            if not source_id:
                serialized = json.dumps(bundle.to_dict(), sort_keys=True, ensure_ascii=True, default=str)
                source_id = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
            artifact = remember(artifact_entity(bundle.artifact_kind, source_id))
            artifact_nodes.append((bundle, artifact))
            indexed: dict[tuple[str, str], NormalizedEntity] = {}
            for item in bundle.indicators:
                entity = remember(normalize_indicator(item))
                indexed[(entity.type, entity.normalized_value)] = entity
                add(artifact, entity, "CONTAINS", {"producer": bundle.analyzer})
                if entity.type == "file_hash_sha256":
                    hash_to_artifacts.setdefault(entity.normalized_value, []).append(artifact)
                elif entity.type == "url":
                    url_to_artifacts.setdefault(entity.normalized_value, []).append(artifact)
            bundle_indicators[id(bundle)] = indexed

        for bundle, artifact in artifact_nodes:
            indexed = bundle_indicators[id(bundle)]
            by_type = lambda kind: [entity for entity in indexed.values() if entity.type == kind]
            if bundle.artifact_kind == "url":
                for url in by_type("url"):
                    for host in by_type("domain") + by_type("ip"):
                        add(url, host, "BELONGS_TO", {"source": "url_host"})
            elif bundle.artifact_kind == "email":
                for address in bundle.features.get("email.from_addresses", []):
                    email_entity = remember(normalize_indicator(IndicatorResult("email", address, address.casefold())))
                    if "@" in address:
                        domain = address.rsplit("@", 1)[-1]
                        domain_entity = remember(normalize_indicator(IndicatorResult("domain", domain, domain.casefold())))
                        add(email_entity, domain_entity, "BELONGS_TO", {"source": "email_from_header"})
                for link in bundle.features.get("email.urls", []):
                    url_entity = remember(normalize_indicator(IndicatorResult("url", link, link)))
                    add(artifact, url_entity, "LINKS_TO", {"source": "email_body_or_html"})
                for attachment in bundle.features.get("email.attachments", []):
                    digest = attachment.get("sha256")
                    if digest:
                        hash_entity = remember(normalize_indicator(IndicatorResult("file_hash_sha256", digest, digest)))
                        add(artifact, hash_entity, "ATTACHED_TO", {"filename": attachment.get("filename"), "sha256": digest})
                        hash_to_artifacts.setdefault(digest.casefold(), []).append(artifact)
            elif bundle.artifact_kind == "qr":
                for relation in bundle.relationships:
                    if relation.kind.casefold() in {"encodes", "decoded_to"}:
                        target = remember(normalize_indicator(relation.target))
                        add(artifact, target, "DECODED_TO", {"source": "qr_payload"})
                for url in bundle.features.get("qr.url_payloads", []):
                    url_entity = remember(normalize_indicator(IndicatorResult("url", url, url)))
                    add(artifact, url_entity, "DECODED_TO", {"source": "qr_payload"})
            elif bundle.artifact_kind == "file":
                for digest in by_type("file_hash_sha256"):
                    add(artifact, digest, "HAS_HASH", {"algorithm": "sha256"})
                for url in bundle.features.get("file.contains_urls", []):
                    url_entity = remember(normalize_indicator(IndicatorResult("url", url, url)))
                    add(artifact, url_entity, "LINKS_TO", {"source": "static_string"})

            # Preserve any analyzer-provided, evidence-backed edge not already expressed above.
            for relation in bundle.relationships:
                source = remember(normalize_indicator(relation.source))
                target = remember(normalize_indicator(relation.target))
                add(source, target, relation.kind, {**relation.attributes, "producer": bundle.analyzer})

            # RESOLVES_TO is only emitted when the submitted evidence already includes resolutions.
            resolved_ips = bundle.features.get("url.resolved_ips", [])
            for host in by_type("domain"):
                for ip_value in resolved_ips:
                    ip_entity = remember(normalize_indicator(IndicatorResult("ip", str(ip_value), str(ip_value))))
                    add(host, ip_entity, "RESOLVES_TO", {"source": "provided_resolution_evidence"})

        # Connect separate submitted artifacts when their canonical indicators match.
        for bundle, artifact in artifact_nodes:
            if bundle.artifact_kind == "email":
                for link in bundle.features.get("email.urls", []):
                    url_entity = normalize_indicator(IndicatorResult("url", link, link))
                    for target in url_to_artifacts.get(url_entity.normalized_value, []):
                        if artifact.id != target.id:
                            add(artifact, target, "LINKS_TO", {"source": "shared_url_indicator"})
                for attachment in bundle.features.get("email.attachments", []):
                    digest = str(attachment.get("sha256", "")).casefold()
                    for target in hash_to_artifacts.get(digest, []):
                        if target.id != artifact.id:
                            kind = target.attributes.get("artifact_kind")
                            if kind in {"file", "qr"}:
                                add(artifact, target, "ATTACHED_TO", {"source": "shared_attachment_sha256", "sha256": digest})
            elif bundle.artifact_kind == "qr":
                for value in bundle.features.get("qr.url_payloads", []):
                    url_entity = normalize_indicator(IndicatorResult("url", value, value))
                    for target in url_to_artifacts.get(url_entity.normalized_value, []):
                        if artifact.id != target.id:
                            add(artifact, target, "DECODED_TO", {"source": "shared_url_indicator"})

        # A file and QR analyzer can independently inspect the same submitted bytes.
        # A shared SHA-256 then gives a reproducible File -> QR bridge without decoding
        # or extracting content during correlation.
        for digest, related_artifacts in hash_to_artifacts.items():
            file_nodes = [node for node in related_artifacts if node.attributes.get("artifact_kind") == "file"]
            qr_nodes = [node for node in related_artifacts if node.attributes.get("artifact_kind") == "qr"]
            for file_node in file_nodes:
                for qr_node in qr_nodes:
                    if file_node.id != qr_node.id:
                        add(file_node, qr_node, "RELATED_TO", {
                            "source": "shared_sha256_across_file_and_qr_analysis",
                            "sha256": digest,
                        })

        return CorrelationResult(tuple(entities.values()), tuple(relationships.values()))
