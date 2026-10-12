from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer

DATA = Path("data/real_url_dataset/data_bal - 20000.xlsx")
OUT = Path("artifacts/real_url_model")
OUT.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42

def url_text(x):
    return np.asarray(x).astype(str)

def main():
    df = pd.read_excel(DATA)
    df = df[["URLs", "Labels"]].dropna()
    df["URLs"] = df["URLs"].astype(str).str.strip()
    df["Labels"] = pd.to_numeric(df["Labels"], errors="coerce")
    df = df.dropna(subset=["Labels"])
    df = df[df["Labels"].isin([0, 1])]
    df = df[df["URLs"].str.len() > 0]
    df = df.drop_duplicates(subset=["URLs"])
    df["Labels"] = df["Labels"].astype(int)

    print("Rows after cleaning:", len(df))
    print("Label counts (0=legitimate, 1=phishing):")
    print(df["Labels"].value_counts().sort_index().to_dict())

    X = df["URLs"].to_numpy()
    y = df["Labels"].to_numpy()

    # 60% training, 20% validation, 20% test.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, stratify=y,
        random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp,
        random_state=RANDOM_STATE
    )

    # Character n-grams learn patterns in URLs without fetching them.
    def make_features():
        return TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            min_df=2,
            max_features=250000,
            sublinear_tf=True,
            dtype=np.float32
        )

    models = {
        "logistic_regression": Pipeline([
            ("tfidf", make_features()),
            ("classifier", LogisticRegression(
                max_iter=1500,
                class_weight="balanced",
                random_state=RANDOM_STATE
            ))
        ]),
        "random_forest": Pipeline([
            ("features", FeatureUnion([
                ("char_tfidf", make_features())
            ])),
            ("classifier", RandomForestClassifier(
                n_estimators=200,
                max_depth=30,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=RANDOM_STATE
            ))
        ])
    }

    results = {}
    trained = {}

    for name, model in models.items():
        print(f"\nTraining {name} ...")
        started = time.perf_counter()
        model.fit(X_train, y_train)
        train_seconds = time.perf_counter() - started

        val_pred = model.predict(X_val)
        val_f1 = f1_score(y_val, val_pred, zero_division=0)
        results[name] = {
            "validation_f1": float(val_f1),
            "validation_precision": float(
                precision_score(y_val, val_pred, zero_division=0)
            ),
            "validation_recall": float(
                recall_score(y_val, val_pred, zero_division=0)
            ),
            "training_seconds": round(train_seconds, 3)
        }
        trained[name] = model
        print("Validation metrics:", results[name])

    best_name = max(results, key=lambda n: results[n]["validation_f1"])
    best_model = trained[best_name]

    # Evaluate the selected model once on the held-out test set.
    test_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, test_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    report = {
        "dataset": str(DATA),
        "dataset_age_note": "Source describes URLs collected in 2021",
        "label_mapping": {"0": "legitimate", "1": "phishing"},
        "deduplicated_rows": int(len(df)),
        "split_sizes": {
            "train": int(len(X_train)),
            "validation": int(len(X_val)),
            "test": int(len(X_test))
        },
        "class_counts": {
            str(k): int(v) for k, v in df["Labels"].value_counts().items()
        },
        "validation_comparison": results,
        "selected_model": best_name,
        "test_metrics": {
            "accuracy": float(accuracy_score(y_test, test_pred)),
            "precision_phishing": float(
                precision_score(y_test, test_pred, zero_division=0)
            ),
            "recall_phishing": float(
                recall_score(y_test, test_pred, zero_division=0)
            ),
            "f1_phishing": float(f1_score(y_test, test_pred, zero_division=0)),
            "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
            "confusion_matrix_labels": ["legitimate", "phishing"],
            "confusion_matrix": cm.tolist(),
            "classification_report": classification_report(
                y_test, test_pred, labels=[0, 1],
                target_names=["legitimate", "phishing"],
                output_dict=True, zero_division=0
            )
        },
        "limitations": [
            "Historical dataset collected in 2021",
            "Random split does not establish performance on future URLs",
            "Dataset may contain collection bias and duplicate domains",
            "This model does not fetch URLs or verify live reputation"
        ]
    }

    joblib.dump(best_model, OUT / "url_phishing_model.joblib")
    (OUT / "evaluation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print("\nSELECTED MODEL:", best_name)
    print("TEST METRICS:")
    print(json.dumps(report["test_metrics"], indent=2))
    print("\nSaved model:", OUT / "url_phishing_model.joblib")
    print("Saved report:", OUT / "evaluation_report.json")

if __name__ == "__main__":
    main()
