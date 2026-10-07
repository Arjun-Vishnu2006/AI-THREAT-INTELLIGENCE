"""Grouped synthetic exposure-category training, calibration, and evaluation."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from itertools import combinations
from math import ceil
from statistics import mean, median
from time import perf_counter
from typing import Any

import numpy as np
import sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score, log_loss, roc_auc_score,
)
from sklearn.pipeline import Pipeline

from app.exposure.engine import CalibratedExposureModel, calibrate_exposure_model
from app.exposure.features import build_exposure_features
from app.novelty.types import NoveltySnapshot

EXPOSURE_CATEGORIES = ("LOW", "GUARDED", "HIGH")
_LAYERS = ("individual_anomaly", "relationship_anomaly", "graph_structure", "attack_chain_path", "temporal")


@dataclass(frozen=True)
class ExposureCase:
    sample_id: str
    campaign_id: str
    category: str
    features: dict[str, str | int | float]


def _scenario_features(category: str, group_index: int, repeat: int, seed: int) -> dict[str, str | int | float]:
    rng = random.Random((seed * 1_000_003) + (group_index * 101) + repeat)
    if category == "LOW":
        kinds = ("url",)
        relationship_signatures = ("URL|BELONGS_TO|DOMAIN",)
        paths: tuple[str, ...] = ()
        graph_features = {
            "graph.node_count": 3, "graph.edge_count": 3, "graph.weak_component_count": 1,
            "graph.isolated_node_count": 0, "graph.max_degree": 2,
            "graph.average_degree": 2.0, "graph.directed_density": 0.5,
            "graph.node_counts_by_type": {"artifact": 1, "url": 1, "domain": 1},
            "graph.edge_counts_by_type": {"CONTAINS": 2, "BELONGS_TO": 1},
        }
        knowledge, hit_count, anomaly, novelty, p_range, temporal_available = (
            "UNKNOWN", 0, "NOT_ANOMALOUS", "NOT_NOVEL", (0.35, 0.95), False,
        )
        vectors = ({"artifact.kind": "url", "url.scheme": "https", "url.length": 25 + rng.randrange(8),
                    "url.host_label_count": 2, "url.subdomain_label_count": 0,
                    "url.path_length": 5 + rng.randrange(5), "url.query_parameter_count": 0,
                    "url.percent_escape_count": 0, "url.has_userinfo": 0},)
    elif category == "GUARDED":
        kinds = ("email", "url", "qr")
        relationship_signatures = (
            "EMAIL|LINKS_TO|URL", "EMAIL|BELONGS_TO|DOMAIN", "QR|DECODED_TO|URL",
        )
        paths = ("EMAIL-[LINKS_TO]->URL", "QR-[DECODED_TO]->URL")
        graph_features = {
            "graph.node_count": 8, "graph.edge_count": 10, "graph.weak_component_count": 1,
            "graph.isolated_node_count": 0, "graph.max_degree": 4,
            "graph.average_degree": 2.5, "graph.directed_density": 0.2,
            "graph.node_counts_by_type": {"artifact": 3, "email": 1, "url": 1, "qr_payload": 1, "domain": 2},
            "graph.edge_counts_by_type": {"CONTAINS": 7, "LINKS_TO": 1, "DECODED_TO": 2},
        }
        knowledge, hit_count, anomaly, novelty, p_range, temporal_available = (
            "UNKNOWN", 0, "NOT_ANOMALOUS", "NOT_NOVEL", (0.08, 0.35), True,
        )
        vectors = (
            {"artifact.kind": "email", "email.links.count": 1 + rng.randrange(3),
             "email.attachments.count": 0, "email.urgency_matches.count": rng.randrange(2),
             "email.reply_domain_differs": 0, "email.body_character_count": 150 + rng.randrange(200)},
            {"artifact.kind": "url", "url.scheme": "https", "url.length": 40 + rng.randrange(25),
             "url.host_label_count": 3, "url.subdomain_label_count": 1,
             "url.path_length": 8 + rng.randrange(10), "url.query_parameter_count": 1,
             "url.percent_escape_count": 0, "url.has_userinfo": 0},
            {"artifact.kind": "qr", "qr.decoded_count": 1, "qr.has_url_payload": 1},
        )
    else:
        kinds = ("email", "file", "qr", "url")
        relationship_signatures = (
            "EMAIL|ATTACHED_TO|FILE", "FILE|RELATED_TO|QR", "QR|DECODED_TO|URL",
            "DOMAIN|RESOLVES_TO|IP",
        )
        paths = (
            "EMAIL-[ATTACHED_TO]->FILE-[RELATED_TO]->QR-[DECODED_TO]->URL",
            "EMAIL-[ATTACHED_TO]->QR-[DECODED_TO]->URL",
        )
        graph_features = {
            "graph.node_count": 12, "graph.edge_count": 19, "graph.weak_component_count": 1,
            "graph.isolated_node_count": 0, "graph.max_degree": 7,
            "graph.average_degree": 3.1, "graph.directed_density": 0.3,
            "graph.node_counts_by_type": {"artifact": 4, "email": 1, "file_hash_sha256": 1,
                                           "qr_payload": 1, "url": 1, "domain": 3, "ip": 1},
            "graph.edge_counts_by_type": {"CONTAINS": 7, "ATTACHED_TO": 3, "RELATED_TO": 1,
                                           "DECODED_TO": 3, "BELONGS_TO": 3, "HAS_HASH": 1, "RESOLVES_TO": 1},
        }
        known = group_index % 2 == 0
        knowledge, hit_count, anomaly, novelty, p_range, temporal_available = (
            ("KNOWN", 1) if known else ("UNKNOWN", 0)
        ) + ("ANOMALOUS", "NOVEL", (0.0005, 0.025), True)
        vectors = (
            {"artifact.kind": "email", "email.links.count": 8 + rng.randrange(12),
             "email.attachments.count": 1 + rng.randrange(3),
             "email.urgency_matches.count": 3 + rng.randrange(6),
             "email.reply_domain_differs": 1, "email.body_character_count": 700 + rng.randrange(800)},
            {"artifact.kind": "file", "file.detected_type": "pe", "file.suffix": ".bin",
             "file.size_bytes": 12000 + rng.randrange(50000), "file.declared_mime_mismatch": 1,
             "file.script_like_extension": 1},
            {"artifact.kind": "qr", "qr.decoded_count": 2, "qr.has_url_payload": 1},
            {"artifact.kind": "url", "url.scheme": "http", "url.length": 120 + rng.randrange(500),
             "url.host_label_count": 5, "url.subdomain_label_count": 3,
             "url.path_length": 60 + rng.randrange(200), "url.query_parameter_count": 5,
             "url.percent_escape_count": 4, "url.has_userinfo": 1},
        )
    layer_assessments = {}
    for name in _LAYERS:
        if name == "temporal" and not temporal_available:
            continue
        p_value = rng.uniform(*p_range)
        layer_assessments[name] = {
            "p_value": p_value,
            "nonconformity": -np.log(max(p_value, 1e-6)),
            "evidence": [{"baseline_occurrences": 0 if category == "HIGH" else 4}],
        }
    combined_p = min(1.0, min((v["p_value"] for v in layer_assessments.values()), default=1.0)
                     * len(layer_assessments))
    snapshot = NoveltySnapshot(
        feature_vectors=tuple(dict(vector) for vector in vectors),
        relationship_signatures=relationship_signatures,
        graph_signature="synthetic-" + category.lower(), graph_features=graph_features,
        path_signatures=paths, entity_keys=(),
        temporal_features=(
            {"new_entity_fraction": 0.7 if category == "HIGH" else 0.12,
             "new_relationship_fraction": 0.8 if category == "HIGH" else 0.1,
             "relationship_frequency_increase_fraction": 0.65 if category == "HIGH" else 0.0}
            if temporal_available else None
        ),
    )
    assessment = {
        "knowledge_status": knowledge,
        "anomaly_status": anomaly,
        "novelty_status": novelty,
        "fusion": {"combined_p_value": combined_p},
        "layers": layer_assessments,
    }
    return build_exposure_features(
        snapshot, assessment, known_ioc_hit_count=hit_count,
        indicator_count=max(2, len(kinds) * 2),
    )


def build_synthetic_exposure_cases(seed: int = 2026) -> list[ExposureCase]:
    # Category definitions are explicit evidence scenarios, not real-world labels or risk scores.
    cases: list[ExposureCase] = []
    for category in EXPOSURE_CATEGORIES:
        for group_index in range(60):
            campaign_id = f"{category.lower()}-scenario-{group_index:03d}"
            for repeat in range(2):
                features = _scenario_features(category, group_index, repeat, seed)
                cases.append(ExposureCase(
                    sample_id=f"{campaign_id}-sample-{repeat}", campaign_id=campaign_id,
                    category=category, features=features,
                ))
    return cases


def _split(cases: list[ExposureCase], seed: int) -> dict[str, list[ExposureCase]]:
    grouped: dict[str, list[ExposureCase]] = {}
    label_for_group: dict[str, str] = {}
    for case in cases:
        if case.campaign_id in label_for_group and label_for_group[case.campaign_id] != case.category:
            raise ValueError("An exposure campaign contains conflicting categories")
        label_for_group[case.campaign_id] = case.category
        grouped.setdefault(case.campaign_id, []).append(case)
    rng = random.Random(seed)
    selected: dict[str, set[str]] = {name: set() for name in ("train", "calibration", "test")}
    for category in EXPOSURE_CATEGORIES:
        groups = sorted(key for key, label in label_for_group.items() if label == category)
        rng.shuffle(groups)
        selected["train"].update(groups[:36])
        selected["calibration"].update(groups[36:48])
        selected["test"].update(groups[48:])
    result = {
        name: [case for case in cases if case.campaign_id in group_ids]
        for name, group_ids in selected.items()
    }
    for left, right in combinations(result, 2):
        left_groups = {case.campaign_id for case in result[left]}
        right_groups = {case.campaign_id for case in result[right]}
        if left_groups & right_groups:
            raise ValueError(f"Exposure campaign leakage detected between {left} and {right}")
    return result


def _probability_metrics(labels: list[str], classes: list[str], probabilities: np.ndarray) -> dict[str, Any]:
    predictions = [classes[index] for index in probabilities.argmax(axis=1)]
    one_hot = np.asarray([[int(label == category) for category in classes] for label in labels])
    confidence = probabilities.max(axis=1)
    correct = np.asarray(predictions) == np.asarray(labels)
    bins = []
    bin_indices = np.minimum((confidence * 10).astype(int), 9)
    for bin_index in range(10):
        lower = bin_index / 10
        upper = (bin_index + 1) / 10
        selected = bin_indices == bin_index
        if selected.any():
            bins.append({"lower": float(lower), "upper": float(upper), "count": int(selected.sum()),
                         "mean_confidence": float(confidence[selected].mean()),
                         "accuracy": float(correct[selected].mean())})
    ece = sum(item["count"] / len(labels) * abs(item["mean_confidence"] - item["accuracy"])
              for item in bins)
    per_category = {}
    for index, category in enumerate(classes):
        actual = one_hot[:, index].astype(bool)
        predicted = np.asarray(predictions) == category
        tp = int(np.sum(actual & predicted))
        fp = int(np.sum(~actual & predicted))
        fn = int(np.sum(actual & ~predicted))
        tn = int(np.sum(~actual & ~predicted))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_category[category] = {
            "support": int(actual.sum()), "true_positive": tp, "false_positive": fp,
            "true_negative": tn, "false_negative": fn, "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
            "false_negative_rate": fn / (fn + tp) if fn + tp else 0.0,
            "brier_score": float(np.mean((probabilities[:, index] - actual) ** 2)),
            "roc_auc": float(roc_auc_score(actual, probabilities[:, index])) if len(set(actual)) == 2 else None,
            "pr_auc": float(average_precision_score(actual, probabilities[:, index])) if len(set(actual)) == 2 else None,
        }
    try:
        macro_roc_auc = float(roc_auc_score(one_hot, probabilities, average="macro", multi_class="ovr"))
    except ValueError:
        macro_roc_auc = None
    try:
        macro_pr_auc = float(average_precision_score(one_hot, probabilities, average="macro"))
    except ValueError:
        macro_pr_auc = None
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, labels=classes, average="macro", zero_division=0)),
        "multiclass_brier_score": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "log_loss": float(log_loss(labels, probabilities, labels=classes)),
        "top_label_expected_calibration_error": float(ece),
        "roc_auc_ovr_macro": macro_roc_auc,
        "pr_auc_macro": macro_pr_auc,
        "reliability_bins": bins,
        "per_category": per_category,
    }


def _metrics(model: CalibratedExposureModel, cases: list[ExposureCase]) -> dict[str, Any]:
    labels = [case.category for case in cases]
    rows = [case.features for case in cases]
    probabilities = model.predict_proba(rows)
    classes = list(model.probability_calibrator.classes_)
    calibrated_metrics = _probability_metrics(labels, classes, probabilities)
    raw_probabilities = model.classifier.predict_proba(rows)
    uncalibrated_metrics = _probability_metrics(labels, classes, raw_probabilities)
    latency_samples = []
    for row in rows:
        started = perf_counter()
        model.predict_proba([row])
        latency_samples.append(perf_counter() - started)
    ordered = sorted(latency_samples)
    return {
        "sample_count": len(cases),
        "calibrated": calibrated_metrics,
        "uncalibrated_reference": uncalibrated_metrics,
        "single_example_inference_latency": {
            "sample_count": len(ordered),
            "mean_ms": mean(ordered) * 1000 if ordered else 0.0,
            "median_ms": median(ordered) * 1000 if ordered else 0.0,
            "p95_ms": ordered[max(0, ceil(0.95 * len(ordered)) - 1)] * 1000 if ordered else 0.0,
        },
        "confusion_matrix_labels": classes,
        "confusion_matrix": confusion_matrix(
            labels, [classes[index] for index in probabilities.argmax(axis=1)], labels=classes
        ).tolist(),
    }


def train_and_evaluate_exposure(seed: int = 2026) -> tuple[CalibratedExposureModel, dict[str, Any]]:
    cases = build_synthetic_exposure_cases(seed)
    splits = _split(cases, seed)
    train, calibration, test = (splits[key] for key in ("train", "calibration", "test"))
    classifier = Pipeline([
        ("vectorizer", DictVectorizer(sparse=True)),
        ("classifier", LogisticRegression(max_iter=600, class_weight="balanced", random_state=seed)),
    ])
    classifier.fit([item.features for item in train], [item.category for item in train])
    model = calibrate_exposure_model(
        classifier, [item.features for item in calibration], [item.category for item in calibration],
    )
    split_campaigns = {
        name: sorted({item.campaign_id for item in rows}) for name, rows in splits.items()
    }
    fingerprint_source = json.dumps(
        [{"id": case.sample_id, "campaign": case.campaign_id, "category": case.category,
          "features": case.features} for case in cases], sort_keys=True, separators=(",", ":"),
    )
    metadata = {
        "task": "exposure_category_prediction",
        "categories": list(EXPOSURE_CATEGORIES),
        "dataset_id": "synthetic-evidence-exposure-v1",
        "dataset_fingerprint_sha256": hashlib.sha256(fingerprint_source.encode()).hexdigest(),
        "dataset_sample_count": len(cases),
        "random_seed": seed,
        "split_strategy": "campaign-grouped 60/20/20 fit/calibration/test; campaign IDs disjoint",
        "split_campaigns": split_campaigns,
        "group_leakage_checked": True,
        "fit_sample_count": len(train),
        "calibration_sample_count": len(calibration),
        "held_out_metrics": _metrics(model, test),
        "calibration_method": "multinomial logistic calibration map fitted on separate calibration groups",
        "features": ["individual analyzer features", "relationship signatures", "graph features",
                     "path signatures", "temporal evidence", "exact-match known-threat status",
                     "per-layer novelty p-values and evidence"],
        "label_semantics": "synthetic evidence exposure categories; not maliciousness, impact, or risk scores",
        "scikit_learn_version": sklearn.__version__,
    }
    return model, metadata
