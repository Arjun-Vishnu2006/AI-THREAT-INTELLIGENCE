"""
Threshold management for Android malware novelty detection.

Responsibilities:
- Calculate reconstruction errors
- Compute novelty threshold
- Save/load threshold
- Classify samples as Known or Novel
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import json
import numpy as np


class NoveltyThreshold:
    """
    Computes and manages novelty detection thresholds.
    """

    def __init__(self, percentile: float = 95.0):
        self.percentile = percentile
        self.threshold: float | None = None

    def reconstruction_errors(
        self,
        original: np.ndarray,
        reconstructed: np.ndarray,
    ) -> np.ndarray:
        """
        Compute Mean Squared Error (MSE)
        for each sample.
        """

        return np.mean(
            np.square(original - reconstructed),
            axis=1,
        )

    def fit(
        self,
        errors: np.ndarray,
    ) -> float:
        """
        Calculate threshold from
        reconstruction errors.
        """

        self.threshold = float(
            np.percentile(errors, self.percentile)
        )

        return self.threshold

    def predict(
        self,
        errors: np.ndarray,
    ) -> np.ndarray:
        """
        Predict novelty.

        Returns:
            0 = Known
            1 = Novel
        """

        if self.threshold is None:
            raise RuntimeError(
                "Threshold has not been initialized."
            )

        return (errors > self.threshold).astype(int)

    def novelty_score(
        self,
        errors: np.ndarray,
    ) -> np.ndarray:
        """
        Normalize reconstruction errors
        into [0,1] novelty scores.
        """

        if self.threshold is None:
            raise RuntimeError(
                "Threshold has not been initialized."
            )

        scores = errors / self.threshold

        return np.clip(scores, 0.0, 1.0)

    def save(
        self,
        path: Union[str, Path],
    ):
        """
        Save threshold to JSON.
        """

        if self.threshold is None:
            raise RuntimeError(
                "Threshold has not been initialized."
            )

        data = {
            "threshold": self.threshold,
            "percentile": self.percentile,
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=4)

    def load(
        self,
        path: Union[str, Path],
    ):
        """
        Load threshold from JSON.
        """

        with open(path, "r") as f:
            data = json.load(f)

        self.threshold = data["threshold"]
        self.percentile = data["percentile"]


if __name__ == "__main__":

    rng = np.random.default_rng(42)

    errors = rng.normal(
        loc=0.05,
        scale=0.01,
        size=1000,
    )

    detector = NoveltyThreshold(percentile=95)

    threshold = detector.fit(errors)

    detector.save("../models/threshold.json")

    print("Threshold:", threshold)

    detector.load("../models/threshold.json")

    sample_errors = np.array([
        0.02,
        0.05,
        0.08,
        0.15,
    ])

    print("Novelty Scores:")
    print(detector.novelty_score(sample_errors))

    print("Predictions:")
    print(detector.predict(sample_errors))