from __future__ import annotations

import json
import re
import hashlib
from collections import Counter
from email import policy
from email.parser import BytesParser
from html.parser import HTMLParser
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "email_dataset" / "extracted"
OUT_DIR = ROOT / "artifacts" / "email_model"
MODEL_PATH = OUT_DIR / "email_spam_model.joblib"
REPORT_PATH = OUT_DIR / "evaluation_report.json"

RANDOM_STATE = 42


class HTMLTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


def extract_message(path: Path) -> str:
    """Extract subject and readable body without executing attachments."""
    try:
        raw = path.read_bytes()
        msg = BytesParser(policy=policy.default).parsebytes(raw)

        subject = str(msg.get("Subject", "") or "")
        body_parts = []

        for part in msg.walk():
            if part.is_multipart():
                continue

            if part.get_content_disposition() == "attachment":
                continue

            content_type = part.get_content_type().lower()
            if content_type not in {"text/plain", "text/html"}:
                continue

            try:
                payload = part.get_content()
                if not isinstance(payload, str):
                    continue
            except Exception:
                payload_bytes = part.get_payload(decode=True) or b""
                charset = part.get_content_charset() or "utf-8"
                payload = payload_bytes.decode(charset, errors="replace")

            if content_type == "text/html":
                parser = HTMLTextExtractor()
                try:
                    parser.feed(payload)
                    payload = " ".join(parser.parts)
                except Exception:
                    payload = re.sub(r"<[^>]*>", " ", payload)

            body_parts.append(payload)

        text = subject + "\n" + "\n".join(body_parts)

        # Normalize whitespace; keep URLs and useful text features.
        text = re.sub(r"\s+", " ", text).strip()
        return text

    except Exception as exc:
        print(f"Skipped unreadable file {path.name}: {exc}")
        return ""


def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Dataset directory not found: {DATA_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    seen = {}

    for folder, label in [("easy_ham", 0), ("spam", 1)]:
        class_dir = DATA_DIR / folder

        if not class_dir.is_dir():
            raise FileNotFoundError(f"Expected class directory: {class_dir}")

        for path in sorted(class_dir.iterdir()):
            if not path.is_file():
                continue

            text = extract_message(path)
            if len(text) < 20:
                continue

            digest = hashlib.sha256(text.casefold().encode("utf-8")).hexdigest()

            # Avoid exact duplicate text crossing the train/test split.
            if digest in seen:
                if seen[digest] != label:
                    print(f"Skipped conflicting duplicate: {path.name}")
                continue

            seen[digest] = label
            records.append({
                "text": text,
                "label": label,
                "source": folder,
            })

    if not records:
        raise RuntimeError("No usable emails found. Check the extracted dataset.")

    df = pd.DataFrame(records)

    print("\nDataset after parsing and deduplication:")
    print("Total emails:", len(df))
    print("Class counts (0=ham, 1=spam):")
    print(df["label"].value_counts().sort_index().to_string())

    if df["label"].nunique() != 2:
        raise RuntimeError("Both ham and spam classes are required.")

    if df["label"].value_counts().min() < 4:
        raise RuntimeError("Too few samples in one class for a reliable split.")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=2,
            max_features=100000,
            sublinear_tf=True,
            dtype=float,
        )),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ])

    print("\nTraining TF-IDF + Logistic Regression...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    # Labels: 0 = ham, 1 = spam.
    cm = confusion_matrix(y_test, predictions, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in cm.ravel())

    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    report = {
        "task": "Spam versus ham email classification",
        "positive_class": "spam",
        "label_mapping": {
            "0": "ham (legitimate according to source dataset)",
            "1": "spam (according to source dataset)",
        },
        "dataset": {
            "source": "SpamAssassin public corpus",
            "input_directory": str(DATA_DIR.relative_to(ROOT)),
            "usable_unique_samples": int(len(df)),
            "class_counts": {
                "ham": int((df["label"] == 0).sum()),
                "spam": int((df["label"] == 1).sum()),
            },
        },
        "split": {
            "method": "stratified train_test_split",
            "test_size": 0.25,
            "random_state": RANDOM_STATE,
            "training_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
        },
        "model": {
            "algorithm": "TF-IDF + Logistic Regression",
            "tfidf_ngram_range": [1, 2],
            "tfidf_max_features": 100000,
            "class_weight": "balanced",
        },
        "metrics": {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "spam_precision": float(precision_score(
                y_test, predictions, pos_label=1, zero_division=0
            )),
            "spam_recall": float(recall_score(
                y_test, predictions, pos_label=1, zero_division=0
            )),
            "spam_f1": float(f1_score(
                y_test, predictions, pos_label=1, zero_division=0
            )),
            "false_positive_rate": float(fpr),
            "confusion_matrix_labels": ["ham", "spam"],
            "confusion_matrix": cm.tolist(),
            "classification_report": classification_report(
                y_test,
                predictions,
                labels=[0, 1],
                target_names=["ham", "spam"],
                output_dict=True,
                zero_division=0,
            ),
        },
        "software": {
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
        },
        "limitations": [
            "Spam detection is not the same as phishing detection.",
            "The corpus is historical and may not represent current email threats.",
            "A random held-out split does not establish future or external generalization.",
            "Predictions must be combined with other evidence and must not be treated as proof of safety.",
        ],
    }

    joblib.dump(model, MODEL_PATH)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nEvaluation results:")
    print("Accuracy:", round(report["metrics"]["accuracy"], 4))
    print("Spam precision:", round(report["metrics"]["spam_precision"], 4))
    print("Spam recall:", round(report["metrics"]["spam_recall"], 4))
    print("Spam F1:", round(report["metrics"]["spam_f1"], 4))
    print("False-positive rate:", round(fpr, 4))
    print("\nConfusion matrix (rows=actual, columns=predicted; ham, spam):")
    print(cm)
    print("\nModel saved to:", MODEL_PATH)
    print("Evaluation report saved to:", REPORT_PATH)
    print("\nNote: this model predicts spam, not verified phishing.")


if __name__ == "__main__":
    main()
