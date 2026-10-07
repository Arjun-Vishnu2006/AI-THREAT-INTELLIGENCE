"""
Autoencoder-based novelty layer.

Loads the trained TensorFlow autoencoder and computes reconstruction
errors for feature vectors supplied by NoveltySnapshot.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from app.novelty.layers import LayerScore
from app.novelty.types import NoveltySnapshot
from app.ml.android_feature_schema import ANDROID_FEATURE_NAMES, validate_android_features


class AutoencoderNoveltyLayer:
    """
    Autoencoder novelty detector.

    Uses the trained autoencoder to determine whether a feature vector
    differs significantly from the learned distribution of known samples.
    """

    name = "autoencoder"

    def __getstate__(self) -> dict[str, Any]:
        """Keep the registry artifact small; the immutable Keras model is separate."""
        state = self.__dict__.copy()
        state["model"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        if self.model is None:
            model_dir = Path(__file__).resolve().parent.parent / "models"
            self.model = tf.keras.models.load_model(model_dir / "autoencoder.keras")

    def __init__(self) -> None:

        base_dir = Path(__file__).resolve().parent.parent

        model_dir = base_dir / "models"

        self.model = tf.keras.models.load_model(
            model_dir / "autoencoder.keras"
        )

        self.scaler = joblib.load(
            model_dir / "scaler.pkl"
        )

        with open(
            model_dir / "threshold.json",
            "r",
            encoding="utf-8",
        ) as fp:
            threshold_data = json.load(fp)

        self.threshold = float(
            threshold_data["threshold"]
        )

        with open(
            model_dir / "feature_columns.json",
            "r",
            encoding="utf-8",
        ) as fp:
            self.feature_columns = json.load(fp)

        # StandardScaler records these when it was trained from a DataFrame.
        # Keep the persisted feature order as a fallback for older artifacts.
        self.feature_names = list(ANDROID_FEATURE_NAMES)
        scaler_names = getattr(self.scaler, "feature_names_in_", None)
        if scaler_names is not None and list(scaler_names) != self.feature_names:
            raise ValueError("Saved scaler feature names do not match the canonical Android schema")
        self.baseline_scaled = np.empty((0, len(self.feature_names)), dtype=np.float32)

    def fit(self, baseline) -> None:
        """
        Autoencoder is already trained.

        No additional fitting is required by the novelty engine.
        """
        vectors = [validate_android_features(snapshot.android_feature_vector)
                   for snapshot in baseline if snapshot.android_feature_vector is not None]
        if vectors:
            baseline_frame = pd.DataFrame(vectors, columns=self.feature_names)
            self.baseline_scaled = self.scaler.transform(baseline_frame)

    def _ordered_values(self, feature_vector: dict[str, Any]) -> np.ndarray:
        """
        Convert one feature dictionary into a correctly
        ordered NumPy array.
        """

        return validate_android_features(feature_vector)

    def _vector_to_array(self, feature_vector: dict[str, Any]) -> np.ndarray:
        """Transform a single feature vector without sklearn feature-name warnings."""
        frame = pd.DataFrame([self._ordered_values(feature_vector)], columns=self.feature_names)
        return self.scaler.transform(frame)

    def score_many(
        self,
        snapshots: list[NoveltySnapshot],
    ) -> list[LayerScore | None]:
        """Score many snapshots in one scaler transform and TensorFlow forward pass."""
        rows: list[np.ndarray] = []
        owners: list[int] = []
        for snapshot_index, snapshot in enumerate(snapshots):
            if snapshot.android_feature_vector is not None:
                rows.append(self._ordered_values(snapshot.android_feature_vector))
                owners.append(snapshot_index)

        if not rows:
            return [None] * len(snapshots)

        frame = pd.DataFrame(rows, columns=self.feature_names)
        X = self.scaler.transform(frame)
        reconstructed = self.model(X, training=False).numpy()
        errors = np.mean(np.square(X - reconstructed), axis=1)

        best_errors = np.full(len(snapshots), -1.0, dtype=float)
        for owner, error in zip(owners, errors):
            best_errors[owner] = max(best_errors[owner], float(error))

        results: list[LayerScore | None] = []
        for error in best_errors:
            if error < 0:
                results.append(None)
                continue
            results.append(LayerScore(
                name=self.name,
                nonconformity=float(error),
                evidence=({
                    "reconstruction_error": float(error),
                    "threshold": self.threshold,
                    "novel": float(error) > self.threshold,
                    "margin": float(error) - self.threshold,
                },),
            ))
        return results
    
    def score(
        self,
        snapshot: NoveltySnapshot,
    ) -> LayerScore | None:
        """
        Compute the autoencoder reconstruction error for the supplied
        NoveltySnapshot and convert it into a LayerScore.
        """

        return self.score_many([snapshot])[0]
