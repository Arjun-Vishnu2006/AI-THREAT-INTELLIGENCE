import json
from pathlib import Path
import numpy as np
import joblib
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from ember.features import PEFeatureExtractor

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "file_dataset" / "ember2018" / "test_features.jsonl"
MODEL = ROOT / "artifacts" / "file_model" / "file_malware_model.joblib"
OUT = ROOT / "artifacts" / "file_model" / "test_evaluation_report.json"

print("Loading existing model...")
model = joblib.load(MODEL)
extractor = PEFeatureExtractor(feature_version=2)

y_true = []
y_pred = []
skipped = 0

print("Evaluating labeled EMBER test records...")
with DATA.open("r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Test records"):
        if not line.strip():
            continue

        record = json.loads(line)
        label = record.get("label")

        if label not in (0, 1):
            skipped += 1
            continue

        try:
            vector = np.asarray(
                extractor.process_raw_features(record),
                dtype=np.float32
            ).reshape(1, -1)

            if vector.shape[1] != 2381 or not np.isfinite(vector).all():
                skipped += 1
                continue

            prediction = int(model.predict(vector)[0])
            y_true.append(int(label))
            y_pred.append(prediction)

        except Exception as exc:
            skipped += 1
            print(f"\nSkipped one record: {type(exc).__name__}: {exc}")

y_true = np.asarray(y_true)
y_pred = np.asarray(y_pred)

if len(y_true) == 0:
    raise RuntimeError("No test records were successfully evaluated.")

tn, fp, fn, tp = confusion_matrix(
    y_true, y_pred, labels=[0, 1]
).ravel()

report = {
    "dataset": "EMBER 2018 labeled test_features.jsonl",
    "model": str(MODEL.name),
    "evaluated_records": int(len(y_true)),
    "skipped_records": int(skipped),
    "accuracy": float(accuracy_score(y_true, y_pred)),
    "malware_precision": float(precision_score(y_true, y_pred, zero_division=0)),
    "malware_recall": float(recall_score(y_true, y_pred, zero_division=0)),
    "malware_f1": float(f1_score(y_true, y_pred, zero_division=0)),
    "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
    "confusion_matrix": {
        "true_benign_predicted_benign": int(tn),
        "true_benign_predicted_malicious": int(fp),
        "true_malicious_predicted_benign": int(fn),
        "true_malicious_predicted_malicious": int(tp),
    },
    "classification_report": classification_report(
        y_true, y_pred,
        labels=[0, 1],
        target_names=["benign", "malicious"],
        output_dict=True,
        zero_division=0,
    ),
    "limitations": [
        "Verify the provenance and separation of the test records from training data.",
        "EMBER 2018 is historical and may not represent current malware.",
        "Feature extraction uses the installed LIEF version, which differs from the original EMBER v2 environment."
    ],
}

OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")

print("\n=== EMBER TEST EVALUATION ===")
print(f"Evaluated records: {len(y_true):,}")
print(f"Skipped records:   {skipped:,}")
print(f"Accuracy:          {report['accuracy']:.4%}")
print(f"Malware precision: {report['malware_precision']:.4%}")
print(f"Malware recall:    {report['malware_recall']:.4%}")
print(f"Malware F1-score:  {report['malware_f1']:.4%}")
print(f"False positive rate: {report['false_positive_rate']:.4%}")
print("\nConfusion matrix (TN, FP, FN, TP):", tn, fp, fn, tp)
print(f"\nDetailed report saved to: {OUT}")
