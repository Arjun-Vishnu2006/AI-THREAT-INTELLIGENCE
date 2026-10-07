"""Reproducible synthetic baseline with campaign-grouped train/validation/test splits."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any, Sequence

import sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline

from app.ml.features import extract_ml_features
from app.services.feature_types import FeatureBundle

LABELS = (
    "routine",
    "synthetic_credential_lure_pattern",
    "synthetic_script_attachment_pattern",
)


@dataclass(frozen=True)
class TrainingRow:
    row_id: str
    group_id: str
    label: str
    features: dict[str, str | int | float]


@dataclass(frozen=True)
class DatasetSplit:
    train: tuple[TrainingRow, ...]
    validation: tuple[TrainingRow, ...]
    test: tuple[TrainingRow, ...]

    def manifest(self) -> dict[str, Any]:
        return {
            name: {"row_ids": [r.row_id for r in rows],
                   "group_ids": sorted({r.group_id for r in rows})}
            for name, rows in (("train", self.train), ("validation", self.validation), ("test", self.test))
        }


@dataclass(frozen=True)
class BaselineTrainingResult:
    pipeline: Pipeline
    metadata: dict[str, Any]


def make_synthetic_training_rows(groups_per_label: int = 12) -> tuple[TrainingRow, ...]:
    if groups_per_label < 5:
        raise ValueError("At least five independent groups per class are required for three-way splitting")
    rows: list[TrainingRow] = []
    for index in range(groups_per_label):
        delta = index % 4
        group = f"synthetic-routine-{index:02d}"
        examples = [
            {"artifact.kind": "url", "url.scheme": "https", "url.length": 22 + delta,
             "url.host_label_count": 2, "url.subdomain_label_count": 0,
             "url.path_length": 4 + delta, "url.query_parameter_count": delta % 2,
             "url.percent_escape_count": 0, "url.has_userinfo": 0},
            {"artifact.kind": "email", "email.links.count": delta % 2,
             "email.attachments.count": 0, "email.urgency_matches.count": 0,
             "email.reply_domain_differs": 0, "email.body_character_count": 70 + delta},
            {"artifact.kind": "file", "file.detected_type": "text", "file.suffix": ".txt",
             "file.size_bytes": 500 + delta * 20, "file.declared_mime_mismatch": 0,
             "file.script_like_extension": 0},
        ]
        rows.extend(TrainingRow(f"{group}-row-{n}", group, "routine", values)
                    for n, values in enumerate(examples))

        group = f"synthetic-credential-{index:02d}"
        examples = [
            {"artifact.kind": "url", "url.scheme": "http", "url.length": 70 + delta * 3,
             "url.host_label_count": 4, "url.subdomain_label_count": 2,
             "url.path_length": 28 + delta, "url.query_parameter_count": 3 + delta,
             "url.percent_escape_count": 2 + delta, "url.has_userinfo": int(index % 3 == 0)},
            {"artifact.kind": "email", "email.links.count": 2 + delta,
             "email.attachments.count": 0, "email.urgency_matches.count": 2 + delta,
             "email.reply_domain_differs": 1, "email.body_character_count": 180 + delta * 10},
            {"artifact.kind": "qr", "qr.decoded_count": 1,
             "qr.has_url_payload": 1},
        ]
        rows.extend(TrainingRow(f"{group}-row-{n}", group, LABELS[1], values)
                    for n, values in enumerate(examples))

        group = f"synthetic-script-{index:02d}"
        examples = [
            {"artifact.kind": "file", "file.detected_type": "pe", "file.suffix": ".scr",
             "file.size_bytes": 16000 + delta * 100, "file.declared_mime_mismatch": 1,
             "file.script_like_extension": 1},
            {"artifact.kind": "file", "file.detected_type": "ole", "file.suffix": ".js",
             "file.size_bytes": 9000 + delta * 200, "file.declared_mime_mismatch": 1,
             "file.script_like_extension": 1, "file.archive_entry_count": 0},
            {"artifact.kind": "email", "email.links.count": delta,
             "email.attachments.count": 1 + delta, "email.urgency_matches.count": 0,
             "email.reply_domain_differs": 0, "email.body_character_count": 100 + delta},
        ]
        rows.extend(TrainingRow(f"{group}-row-{n}", group, LABELS[2], values)
                    for n, values in enumerate(examples))
    return tuple(rows)


def stratified_group_split(rows: Sequence[TrainingRow], seed: int = 2026) -> DatasetSplit:
    groups: dict[str, list[TrainingRow]] = {}
    group_labels: dict[str, str] = {}
    for row in rows:
        if row.group_id in group_labels and group_labels[row.group_id] != row.label:
            raise ValueError(f"Group {row.group_id!r} contains more than one label")
        group_labels[row.group_id] = row.label
        groups.setdefault(row.group_id, []).append(row)
    rng = random.Random(seed)
    train_groups: set[str] = set()
    validation_groups: set[str] = set()
    test_groups: set[str] = set()
    for label in sorted({row.label for row in rows}):
        class_groups = sorted(group for group, group_label in group_labels.items() if group_label == label)
        if len(class_groups) < 5:
            raise ValueError(f"Label {label!r} needs at least five independent groups")
        rng.shuffle(class_groups)
        train_count = max(1, round(len(class_groups) * 0.6))
        validation_count = max(1, round(len(class_groups) * 0.2))
        if train_count + validation_count >= len(class_groups):
            train_count = len(class_groups) - 2
            validation_count = 1
        train_groups.update(class_groups[:train_count])
        validation_groups.update(class_groups[train_count:train_count + validation_count])
        test_groups.update(class_groups[train_count + validation_count:])

    def collect(selected: set[str]) -> tuple[TrainingRow, ...]:
        return tuple(row for row in rows if row.group_id in selected)

    split = DatasetSplit(collect(train_groups), collect(validation_groups), collect(test_groups))
    if not split.train or not split.validation or not split.test:
        raise ValueError("Train, validation, and test partitions must all contain rows")
    return split


def _metrics(pipeline: Pipeline, rows: Sequence[TrainingRow]) -> dict[str, Any]:
    y_true = [row.label for row in rows]
    feature_rows = [row.features for row in rows]
    y_pred = pipeline.predict(feature_rows).tolist()
    classes = list(pipeline.named_steps["classifier"].classes_)
    probabilities = pipeline.predict_proba(feature_rows)
    per_class = {}
    for index, label in enumerate(classes):
        actual = [item == label for item in y_true]
        predicted = [item == label for item in y_pred]
        tp = sum(a and p for a, p in zip(actual, predicted))
        fp = sum((not a) and p for a, p in zip(actual, predicted))
        fn = sum(a and (not p) for a, p in zip(actual, predicted))
        tn = sum((not a) and (not p) for a, p in zip(actual, predicted))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_class[label] = {
            "support": sum(actual), "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
            "false_negative_rate": fn / (fn + tp) if fn + tp else 0.0,
            "roc_auc": float(roc_auc_score(actual, probabilities[:, index])) if len(set(actual)) == 2 else None,
            "pr_auc": float(average_precision_score(actual, probabilities[:, index])) if len(set(actual)) == 2 else None,
        }
    one_hot = [[int(item == label) for label in classes] for item in y_true]
    return {
        "row_count": len(rows),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "labels": classes,
        "per_class": per_class,
        "roc_auc_ovr_macro": float(roc_auc_score(one_hot, probabilities, average="macro", multi_class="ovr")),
        "pr_auc_macro": float(average_precision_score(one_hot, probabilities, average="macro")),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
    }


def train_baseline(rows: Sequence[TrainingRow] | None = None, seed: int = 2026,
                   max_iter: int = 500) -> BaselineTrainingResult:
    if max_iter < 1:
        raise ValueError("max_iter must be positive")
    rows = tuple(make_synthetic_training_rows() if rows is None else rows)
    split = stratified_group_split(rows, seed=seed)
    pipeline = Pipeline([
        ("vectorizer", DictVectorizer(sparse=True)),
        ("classifier", LogisticRegression(max_iter=max_iter, random_state=seed, class_weight="balanced")),
    ])
    # Fit only on training groups. Validation/test groups never enter fit or preprocessing.
    pipeline.fit([row.features for row in split.train], [row.label for row in split.train])
    validation_metrics = _metrics(pipeline, split.validation)
    test_metrics = _metrics(pipeline, split.test)
    dataset_json = json.dumps(
        [{"row_id": r.row_id, "group_id": r.group_id, "label": r.label, "features": r.features}
         for r in sorted(rows, key=lambda item: item.row_id)],
        sort_keys=True, separators=(",", ":"),
    )
    metadata = {
        "task": "synthetic_behavior_classification",
        "labels": list(LABELS),
        "dataset_id": "synthetic-feature-baseline-v1",
        "dataset_fingerprint_sha256": hashlib.sha256(dataset_json.encode()).hexdigest(),
        "dataset_row_count": len(rows),
        "random_seed": seed,
        "split_strategy": "stratified_by_label_group; group ids are disjoint across train/validation/test",
        "split": split.manifest(),
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "model_parameters": {"estimator": "LogisticRegression", "max_iter": max_iter,
                             "class_weight": "balanced", "random_state": seed},
        "feature_pipeline": ["allowlisted analyzer feature extraction", "DictVectorizer", "LogisticRegression"],
        "training_data_is_synthetic_only": True,
        "security_verdicts_or_risk_scores_produced": False,
        "scikit_learn_version": sklearn.__version__,
    }
    return BaselineTrainingResult(pipeline, metadata)


def predict_behavior_class(pipeline: Pipeline, bundle: FeatureBundle) -> dict[str, str]:
    features = extract_ml_features(bundle)
    return predict_behavior_features(pipeline, features)


def predict_behavior_features(pipeline: Pipeline,
                              features: dict[str, str | int | float]) -> dict[str, str]:
    label = str(pipeline.predict([features])[0])
    return {"behavior_class": label, "knowledge_status": "NOT_ASSESSED",
            "anomaly_status": "NOT_ASSESSED", "novelty_status": "NOT_ASSESSED"}
