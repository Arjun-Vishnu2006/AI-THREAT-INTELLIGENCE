"""Initial NetworkX threat graph and bounded relationship-path extraction."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

import networkx as nx

from app.services.correlation import CorrelationResult, ThreatRelationship
from app.services.entities import NormalizedEntity, normalize_indicator
from app.services.contracts import IndicatorResult, RelationshipResult


@dataclass
class ThreatGraph:
    graph: nx.MultiDiGraph

    @classmethod
    def from_correlation(cls, result: CorrelationResult) -> "ThreatGraph":
        graph = nx.MultiDiGraph()
        for entity in result.entities:
            graph.add_node(entity.id, **entity.to_dict())
        for index, relation in enumerate(result.relationships):
            graph.add_edge(relation.source_id, relation.target_id, key=f"{relation.kind}:{index}",
                           kind=relation.kind, attributes=relation.attributes)
        return cls(graph)

    @classmethod
    def from_records(cls, indicators: Sequence[IndicatorResult],
                     relationships: Sequence[RelationshipResult]) -> "ThreatGraph":
        nodes: dict[str, NormalizedEntity] = {}
        edges: list[ThreatRelationship] = []
        for item in indicators:
            entity = normalize_indicator(item)
            nodes[entity.id] = entity
        for relation in relationships:
            source, target = normalize_indicator(relation.source), normalize_indicator(relation.target)
            nodes[source.id] = source
            nodes[target.id] = target
            edges.append(ThreatRelationship(source.id, target.id, relation.kind.upper(), relation.attributes))
        result = CorrelationResult(tuple(nodes.values()), tuple(edges))
        return cls.from_correlation(result)

    def features(self) -> dict[str, Any]:
        graph = self.graph
        node_types = Counter(data.get("type", "unknown") for _, data in graph.nodes(data=True))
        edge_types = Counter(data.get("kind", "RELATED_TO") for _, _, data in graph.edges(data=True))
        simple = nx.DiGraph(graph)
        weak_components = list(nx.weakly_connected_components(graph)) if graph.number_of_nodes() else []
        degrees = [degree for _, degree in graph.degree()]
        return {
            "graph.node_count": graph.number_of_nodes(),
            "graph.edge_count": graph.number_of_edges(),
            "graph.node_counts_by_type": dict(sorted(node_types.items())),
            "graph.edge_counts_by_type": dict(sorted(edge_types.items())),
            "graph.weak_component_count": len(weak_components),
            "graph.isolated_node_count": sum(1 for _, degree in graph.degree() if degree == 0),
            "graph.max_degree": max(degrees, default=0),
            "graph.average_degree": (sum(degrees) / len(degrees)) if degrees else 0.0,
            "graph.directed_density": nx.density(simple) if simple.number_of_nodes() > 1 else 0.0,
        }

    def paths(self, *, max_depth: int = 6, max_paths: int = 50) -> list[dict[str, Any]]:
        if not 1 <= max_depth <= 12 or not 1 <= max_paths <= 500:
            raise ValueError("max_depth must be 1-12 and max_paths must be 1-500")
        graph = self.graph
        artifacts = [node for node, data in graph.nodes(data=True) if data.get("type") == "artifact"]
        paths: list[dict[str, Any]] = []
        for source in artifacts:
            for target in artifacts:
                if source == target or not nx.has_path(graph, source, target):
                    continue
                try:
                    found = nx.all_simple_paths(graph, source, target, cutoff=max_depth)
                    for node_path in found:
                        if len(node_path) < 2:
                            continue
                        edges = []
                        for left, right in zip(node_path, node_path[1:]):
                            choices = graph.get_edge_data(left, right) or {}
                            rels = sorted({edge_data.get("kind", "RELATED_TO") for edge_data in choices.values()})
                            edges.append({"source_id": left, "target_id": right, "relationships": rels})
                        paths.append({
                            "path_type": "relationship_path",
                            "source_artifact": source,
                            "target_artifact": target,
                            "nodes": [graph.nodes[node] for node in node_path],
                            "edges": edges,
                            "hop_count": len(node_path) - 1,
                        })
                        if len(paths) >= max_paths:
                            return paths
                except nx.NetworkXNoPath:
                    continue
        return paths

    def to_dict(self, *, include_paths: bool = True, max_depth: int = 6,
                max_paths: int = 50) -> dict[str, Any]:
        nodes = [dict(data) for _, data in self.graph.nodes(data=True)]
        edges = [
            {"source_id": source, "target_id": target, "kind": data["kind"],
             "attributes": data.get("attributes", {})}
            for source, target, data in self.graph.edges(data=True)
        ]
        result = {"entities": nodes, "relationships": edges, "features": self.features()}
        if include_paths:
            result["paths"] = self.paths(max_depth=max_depth, max_paths=max_paths)
        return result


class NetworkXGraphProcessor:
    """GraphProcessor contract implementation; contains no anomaly or risk scoring."""

    def process(self, indicators: Sequence[IndicatorResult],
                relationships: Sequence[RelationshipResult], context: Any = None) -> ThreatGraph:
        return ThreatGraph.from_records(indicators, relationships)

    def process_correlation(self, result: CorrelationResult) -> ThreatGraph:
        return ThreatGraph.from_correlation(result)
