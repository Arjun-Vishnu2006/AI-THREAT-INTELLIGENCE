import json
import time
from pathlib import Path
from collections import Counter

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from lightgbm import LGBMClassifier
import joblib

from ember.features import PEFeatureExtractor

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "file_dataset" / "ember2018"
OUTPUT_DIR = ROOT / "artifacts" / "file_model"

# Start conservatively because available RAM was about 4.3 GB.
SAMPLES_PER_CLASS = 10_000
RANDOM_STATE = 42

def main():
    started = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(DATA_DIR.glob("train_features_*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No training shards found in {DATA_DIR}")

    # Reservoir sampling keeps memory use bounded while scanning all shards.
    rng = np.random.default_rng(RANDOM_STATE)
    reservoirs = {0: [], 1: []}
    seen = Counter()

    print("Scanning EMBER shards and selecting a balanced sample...", flush=True)

    for path in files:
        print(f"Reading {path.name}", flush=True)
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue

                record = json.loads(line)
                label = record.get("label")
                if label not in (0, 1):
                    continue

                seen[label] += 1
                bucket = reservoirs[label]
                item = line

                if len(bucket) < SAMPLES_PER_CLASS:
                    bucket.append(item)
                else:
                    index = int(rng.integers(0, seen[label]))
                    if index < SAMPLES_PER_CLASS:
                        bucket[index] = item

    print("Available labeled records:", dict(seen), flush=True)
    print("Selected per class:", {k: len(v) for k, v in reservoirs.items()}, flush=True)

    if any(len(reservoirs[label]) < 100 for label in (0, 1)):
        raise RuntimeError("Insufficient labeled records for training.")

    extractor = PEFeatureExtractor(feature_version=2)
    vectors = []
    labels = []

    print("Converting selected records to EMBER v2 feature vectors...", flush=True)
    for label in (0, 1):
        for i, line in enumerate(reservoirs[label], start=1):
            record = json.loads(line)

            for key in ("label", "md5", "sha256", "appeared", "avclass"):
                record.pop(key, None)

            vector = np.asarray(
                extractor.process_raw_features(record), dtype=np.float32
            ).reshape(-1)

            if vector.size != 2381 or not np.isfinite(vector).all():
                raise ValueError(f"Invalid feature vector in class {label}, item {i}")

            vectors.append(vector)
            labels.append(label)

            if i % 2000 == 0:
                print(f"Class {label}: converted {i:,}", flush=True)

    X = np.vstack(vectors)
    y = np.asarray(labels, dtype=np.int8)
    del vectors, labels, reservoirs

    print("Matrix shape:", X.shape, flush=True)
    print("Splitting train and validation sets...", flush=True)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    del X, y

    model = LGBMClassifier(
        objective="binary",
        n_estimators=250,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        subsample=0.9,
        colsample_bytree=0.9,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=4,
        verbosity=-1,
    )

    print("Training LightGBM classifier...", flush=True)
    model.fit(X_train, y_train)

    print("Evaluating validation set...", flush=True)
    predictions = model.predict(X_val)
    probabilities = model.predict_proba(X_val)[:, 1]

    cm = confusion_matrix(y_val, predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    report = {
        "dataset": "EMBER 2018",
        "feature_version": 2,
        "feature_dimensions": 2381,
        "samples_per_class": SAMPLES_PER_CLASS,
        "random_state": RANDOM_STATE,
        "validation_fraction": 0.20,
        "metrics": {
            "accuracy": float(accuracy_score(y_val, predictions)),
            "malware_precision": float(precision_score(y_val, predictions, zero_division=0)),
            "malware_recall": float(recall_score(y_val, predictions, zero_division=0)),
            "malware_f1": float(f1_score(y_val, predictions, zero_division=0)),
            "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
            "confusion_matrix": {
                "true_benign_predicted_benign": int(tn),
                "true_benign_predicted_malicious": int(fp),
                "true_malicious_predicted_benign": int(fn),
                "true_malicious_predicted_malicious": int(tp),
            },
        },
        "classification_report": classification_report(
            y_val, predictions, target_names=["benign", "malicious"], output_dict=True
        ),
        "limitations": [
            "EMBER 2018 is a historical dataset; performance may not represent current malware.",
            "Metrics use a random split of a sampled training set, not the official test set.",
            "LIEF version differs from the version used to compute the original EMBER v2 features.",
            "Predictions are static PE malware assessments, not live reputation checks.",
        ],
    }

    model_path = OUTPUT_DIR / "file_malware_model.joblib"
    report_path = OUTPUT_DIR / "evaluation_report.json"

    joblib.dump(model, model_path)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n--- Validation metrics ---")
    print(json.dumps(report["metrics"], indent=2))
    print("\nModel saved:", model_path)
    print("Report saved:", report_path)
    print(f"Elapsed minutes: {(time.time() - started) / 60:.1f}")

if __name__ == "__main__":
    main()
