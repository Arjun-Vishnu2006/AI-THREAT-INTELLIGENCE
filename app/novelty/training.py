"""Leakage-aware fit, calibration, evaluation, and layer-ablation study."""

from __future__ import annotations

from itertools import combinations
from math import ceil
from statistics import mean, median
from time import perf_counter
from typing import Any

from sklearn.metrics import average_precision_score, roc_auc_score

from app.novelty.engine import NoveltyEngine
from app.novelty.synthetic import SyntheticNoveltyCase, build_synthetic_novelty_corpus, corpus_fingerprint


def _binary_metrics(labels: list[bool], predicted: list[bool],
                    scores: list[float] | None = None) -> dict[str, float | int | None]:
    tp = sum(actual and guess for actual, guess in zip(labels, predicted))
    fp = sum((not actual) and guess for actual, guess in zip(labels, predicted))
    fn = sum(actual and (not guess) for actual, guess in zip(labels, predicted))
    tn = sum((not actual) and (not guess) for actual, guess in zip(labels, predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    positives, negatives = tp + fn, tn + fp
    ranking: dict[str, float | None] = {"roc_auc": None, "pr_auc": None}
    if scores is not None and len(set(labels)) == 2:
        ranking = {
            "roc_auc": float(roc_auc_score(labels, scores)),
            "pr_auc": float(average_precision_score(labels, scores)),
        }
    return {"sample_count": len(labels), "true_positive": tp, "false_positive": fp,
            "true_negative": tn, "false_negative": fn, "precision": precision,
            "recall": recall, "f1": f1,
            "false_positive_rate": fp / negatives if negatives else 0.0,
            "false_negative_rate": fn / positives if positives else 0.0,
            **ranking}


def _evaluate(engine: NoveltyEngine, cases: list[SyntheticNoveltyCase],
              omit: str | None = None) -> dict[str, float | int | None]:
    labels: list[bool] = []
    predictions: list[bool] = []
    ranking_scores: list[float] = []
    scored_cases = engine.score_many([case.snapshot for case in cases])
    for case, layer_scores in zip(cases, scored_cases):
        labels.append(case.label == "unseen")
        fused = engine.fuse(layer_scores, omit=omit)
        predictions.append(fused["novelty_status"] == "NOVEL")
        combined = fused["combined_p_value"]
        ranking_scores.append(1.0 - float(combined if combined is not None else 1.0))
    return _binary_metrics(labels, predictions, ranking_scores)


def _evaluate_stage(engine: NoveltyEngine, cases: list[SyntheticNoveltyCase],
                    layer_name: str | None) -> tuple[dict[str, Any], dict[str, list[Any]]]:
    labels: list[bool] = []
    predictions: list[bool] = []
    scores: list[float] = []
    covered = 0
    errors: dict[str, list[Any]] = {"false_positive_cases": [], "false_negative_cases": []}
    scored_cases = engine.score_many([case.snapshot for case in cases])
    for case, layer_scores in zip(cases, scored_cases):
        actual = case.label == "unseen"
        if layer_name is None:
            combined = engine.fuse(layer_scores)
            p_value = combined["combined_p_value"]
            score = 1.0 - float(p_value if p_value is not None else 1.0)
            predicted = combined["novelty_status"] == "NOVEL"
            covered += bool(combined["available_layers"])
        else:
            layer = layer_scores.get(layer_name)
            p_value = float(layer["p_value"]) if layer else 1.0
            score = float(layer["nonconformity"]) if layer else 0.0
            predicted = bool(layer and p_value <= engine.alpha)
            covered += layer is not None
        labels.append(actual)
        predictions.append(predicted)
        scores.append(score)
        category = "false_positive_cases" if predicted and not actual else (
            "false_negative_cases" if actual and not predicted else None
        )
        if category and len(errors[category]) < 25:
            errors[category].append({
                "sample_id": case.sample_id,
                "description": case.description,
                "observed_layer_scores": layer_scores,
                "stage": layer_name or "full_multi_level_fusion",
            })
    metrics = _binary_metrics(labels, predictions, scores)
    metrics["coverage_count"] = covered
    metrics["coverage_fraction"] = covered / len(cases) if cases else 0.0
    return metrics, errors


def _latency_summary(samples: list[float]) -> dict[str, float | int]:
    ordered = sorted(samples)
    return {
        "sample_count": len(ordered),
        "mean_ms": mean(ordered) * 1000 if ordered else 0.0,
        "median_ms": median(ordered) * 1000 if ordered else 0.0,
        "p95_ms": ordered[max(0, ceil(0.95 * len(ordered)) - 1)] * 1000 if ordered else 0.0,
    }


def train_and_evaluate_novelty(seed: int = 2026, alpha: float = 0.05) -> tuple[NoveltyEngine, dict[str, Any]]:
    corpus = build_synthetic_novelty_corpus(seed)
    train = corpus["train"]
    calibration = corpus["calibration"]
    test = corpus["test_seen"] + corpus["test_unseen"]
    split_groups = {name: {case.campaign_id for case in cases} for name, cases in corpus.items()}
    for left, right in combinations(split_groups, 2):
        if split_groups[left] & split_groups[right]:
            raise ValueError(f"Synthetic campaign leakage detected between {left} and {right}")

    engine = NoveltyEngine(alpha=alpha).fit(
        [case.snapshot for case in train],
        [case.snapshot for case in calibration],
    )
    layer_names = list(engine.layers)
    stage_names = {
        "baseline_individual_anomaly": "individual_anomaly",
        "relationship_anomaly": "relationship_anomaly",
        "graph_structure_novelty": "graph_structure",
        "attack_chain_novelty": "attack_chain_path",
        "temporal_novelty": "temporal",
        "autoencoder_novelty": "autoencoder",
    }
    stage_metrics: dict[str, Any] = {}
    error_analysis: dict[str, Any] = {}
    for stage, layer_name in stage_names.items():
        stage_metrics[stage], error_analysis[stage] = _evaluate_stage(engine, test, layer_name)
    stage_metrics["full_multi_level_fusion"], error_analysis["full_multi_level_fusion"] = _evaluate_stage(engine, test, None)
    layer_metrics = {
        stage_names[stage]: stage_metrics[stage]
        for stage in stage_names
    }
    full_metrics = stage_metrics["full_multi_level_fusion"]
    ablations = {
        f"without_{layer_name}": _evaluate(engine, test, omit=layer_name)
        for layer_name in layer_names
    }
    novelty_only = [case for case in test if case.label == "unseen"]
    baseline_only = [case for case in test if case.label == "baseline"]
    inference_seconds = []
    for case in test:
        started = perf_counter()
        layer_scores = engine.score_layers(case.snapshot)
        engine.fuse(layer_scores)
        inference_seconds.append(perf_counter() - started)
    metadata = {
        "task": "multi_level_novelty_detection",
        "versioned_layers": layer_names,
        "alpha": alpha,
        "fusion_method": "Bonferroni-corrected minimum empirical conformal p-value",
        "conformal_calibration": "p=(1 + calibration scores >= observed)/(n + 1)",
        "training_sample_count": len(train),
        "calibration_sample_count": len(calibration),
        "held_out_test_count": len(test),
        "held_out_unseen_count": len(novelty_only),
        "held_out_baseline_count": len(baseline_only),
        "seed": seed,
        "dataset_id": "synthetic-multilevel-novelty-v1",
        "dataset_fingerprint_sha256": corpus_fingerprint(corpus),
        "split_campaigns": {name: sorted(ids) for name, ids in split_groups.items()},
        "group_leakage_checked": True,
        "held_out_metrics": full_metrics,
        "stage_comparison": stage_metrics,
        "single_layer_metrics": layer_metrics,
        "ablation_metrics": ablations,
        "error_analysis": error_analysis,
        "inference_latency": _latency_summary(inference_seconds),
        "deliberately_unseen_pattern": "Email -> File -> QR -> URL with an unseen Domain -> IP resolution edge and feature extremes",
        "synthetic_only": True,
        "risk_score_produced": False,
    }
    return engine, metadata
if __name__ == "__main__":
    import json

    print("=" * 70)
    print("Training Novelty Engine...")
    print("=" * 70)

    engine, metadata = train_and_evaluate_novelty()

    print("\nTraining completed successfully!\n")

    print(json.dumps(metadata, indent=2))
