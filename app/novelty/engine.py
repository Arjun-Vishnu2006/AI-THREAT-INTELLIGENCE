
"""Empirically calibrated, multi-level novelty fusion with explicit evidence."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from app.novelty.layers import (
    AttackPathNoveltyLayer,
    GraphStructureNoveltyLayer,
    IndividualAnomalyLayer,
    RelationshipAnomalyLayer,
    TemporalNoveltyLayer,
)
from app.novelty.ae_layers.autoencoder_novelty_layer import AutoencoderNoveltyLayer
from app.novelty.types import NoveltySnapshot


@dataclass
class NoveltyEngine:
    alpha: float = 0.05
    layers: dict[str, Any] = field(default_factory=dict)
    calibration_scores: dict[str, list[float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.layers:
            layer_objects = (
                IndividualAnomalyLayer(),
                RelationshipAnomalyLayer(),
                GraphStructureNoveltyLayer(),
                AttackPathNoveltyLayer(),
                TemporalNoveltyLayer(),
                AutoencoderNoveltyLayer(),
            )
            self.layers = {layer.name: layer for layer in layer_objects}

        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be between 0 and 1")

    def fit(
        self,
        baseline: Sequence[NoveltySnapshot],
        calibration: Sequence[NoveltySnapshot],
    ) -> "NoveltyEngine":
        if len(baseline) < 10 or len(calibration) < 20:
            raise ValueError(
                "Need at least 10 baseline and 20 independent calibration snapshots"
            )

        self.calibration_scores.clear()

        for name, layer in self.layers.items():
            fit = getattr(layer, "fit", None)
            if fit is not None:
                fit(baseline)

            if isinstance(layer, AutoencoderNoveltyLayer):
                raw_scores = layer.score_many(list(calibration))
                scores = [
                    raw.nonconformity
                    for raw in raw_scores
                    if raw is not None
                ]

                # Skip the autoencoder when the calibration data does not
                # contain enough valid Android feature vectors.
                if len(scores) < 20:
                    if len(scores) == 19:
                        raise ValueError(
                            f"Autoencoder requires at least 20 scored Android "
                            f"calibration samples; got {len(scores)}"
                        )
                    continue
            else:
                scores = []
                for item in calibration:
                    score = layer.score(item)
                    if score is not None:
                        scores.append(score.nonconformity)

                if len(scores) < 20:
                    raise ValueError(
                        f"Layer {name!r} has fewer than 20 calibration observations"
                    )

            self.calibration_scores[name] = sorted(
                float(value) for value in scores
            )

        if not self.calibration_scores:
            raise ValueError(
                "No novelty layers could be calibrated: fewer than 20 matching-schema "
                "calibration observations may be available"
            )

        return self

    @staticmethod
    def _p_value(calibration: Sequence[float], observed: float) -> float:
        # Finite-sample conformal tail probability with a +1 correction; ties are conservative.
        return (1 + sum(value >= observed for value in calibration)) / (len(calibration) + 1)

    def score_layers(self, snapshot: NoveltySnapshot) -> dict[str, dict[str, Any]]:
        scores: dict[str, dict[str, Any]] = {}

        for name, layer in self.layers.items():
            if name not in self.calibration_scores:
                continue

            raw = layer.score(snapshot)
            if raw is None:
                continue

            calibration = self.calibration_scores[name]
            scores[name] = {
                "nonconformity": raw.nonconformity,
                "p_value": self._p_value(calibration, raw.nonconformity),
                "calibration_count": len(calibration),
                "evidence": list(raw.evidence),
            }

        return scores

    def score_many(
        self, snapshots: Sequence[NoveltySnapshot]
    ) -> list[dict[str, dict[str, Any]]]:
        """Score a batch of snapshots, batching the TensorFlow autoencoder pass."""
        results: list[dict[str, dict[str, Any]]] = [
            dict() for _ in snapshots
        ]

        for name, layer in self.layers.items():
            if name not in self.calibration_scores:
                continue

            if isinstance(layer, AutoencoderNoveltyLayer):
                raw_scores = layer.score_many(list(snapshots))
            else:
                raw_scores = [layer.score(snapshot) for snapshot in snapshots]

            calibration = self.calibration_scores[name]

            for result, raw in zip(results, raw_scores):
                if raw is None:
                    continue

                result[name] = {
                    "nonconformity": raw.nonconformity,
                    "p_value": self._p_value(calibration, raw.nonconformity),
                    "calibration_count": len(calibration),
                    "evidence": list(raw.evidence),
                }

        return results

    def fuse(
        self,
        layer_scores: dict[str, dict[str, Any]],
        *,
        omit: str | None = None,
    ) -> dict[str, Any]:
        available = {
            key: value
            for key, value in layer_scores.items()
            if key != omit
        }

        if not available:
            return {
                "method": "bonferroni_min_p",
                "available_layers": [],
                "combined_p_value": None,
                "novelty_status": "NOT_ASSESSED",
            }

        minimum = min(value["p_value"] for value in available.values())
        combined = min(1.0, len(available) * minimum)

        return {
            "method": "bonferroni_min_p",
            "familywise_alpha": self.alpha,
            "available_layers": list(available),
            "combined_p_value": combined,
            "novelty_status": (
                "NOVEL" if combined <= self.alpha else "NOT_NOVEL"
            ),
            "significant_layer": min(
                available, key=lambda key: available[key]["p_value"]
            ),
            "correction_factor": len(available),
        }

    def assess(
        self,
        snapshot: NoveltySnapshot,
        knowledge_status: str = "NOT_ASSESSED",
    ) -> dict[str, Any]:
        layer_scores = self.score_layers(snapshot)
        fused = self.fuse(layer_scores)
        individual = layer_scores.get("individual_anomaly")

        if individual is None:
            anomaly_status = "NOT_ASSESSED"
        else:
            anomaly_status = (
                "ANOMALOUS"
                if individual["p_value"] <= self.alpha
                else "NOT_ANOMALOUS"
            )

        return {
            "knowledge_status": knowledge_status,
            "anomaly_status": anomaly_status,
            "novelty_status": fused["novelty_status"],
            "fusion": fused,
            "layers": layer_scores,
            "interpretation": (
                "Knowledge match, individual anomaly, and fused novelty are "
                "separate assessments. A non-significant result means "
                "insufficient calibrated evidence at this threshold, not proof "
                "of safety."
            ),
        }
