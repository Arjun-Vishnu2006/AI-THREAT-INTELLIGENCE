"""Linear exposure-category model with an independent multinomial calibration map."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


@dataclass
class CalibratedExposureModel:
    classifier: Pipeline
    probability_calibrator: LogisticRegression

    @property
    def classes_(self) -> np.ndarray:
        return self.classifier.named_steps["classifier"].classes_

    def logits(self, rows: Sequence[dict[str, str | int | float]]) -> np.ndarray:
        return np.asarray(self.classifier.decision_function(list(rows)), dtype=float)

    def predict_proba(self, rows: Sequence[dict[str, str | int | float]]) -> np.ndarray:
        return self.probability_calibrator.predict_proba(self.logits(rows))

    def explain(self, features: dict[str, str | int | float], category: str,
                limit: int = 8) -> list[dict[str, object]]:
        vectorizer = self.classifier.named_steps["vectorizer"]
        estimator = self.classifier.named_steps["classifier"]
        transformed = vectorizer.transform([features])
        calibrated_index = list(self.probability_calibrator.classes_).index(category)
        # Explain the calibrated class logit as a linear combination of base logits.
        effective_weights = self.probability_calibrator.coef_[calibrated_index] @ estimator.coef_
        contributions = np.asarray(transformed.multiply(effective_weights).toarray()).ravel()
        names = vectorizer.get_feature_names_out()
        ranked = sorted(range(len(names)), key=lambda idx: abs(contributions[idx]), reverse=True)
        output = []
        for index in ranked:
            if contributions[index] == 0:
                continue
            name = str(names[index])
            output.append({
                "feature": name,
                "value": features.get(name),
                "linear_contribution": float(contributions[index]),
            })
            if len(output) >= limit:
                break
        return output


def calibrate_exposure_model(classifier: Pipeline,
                             calibration_features: Sequence[dict[str, str | int | float]],
                             calibration_labels: Sequence[str]) -> CalibratedExposureModel:
    """Fit a low-capacity multinomial calibration map on held-out calibration groups."""
    if len(calibration_features) < 30 or len(set(calibration_labels)) < 3:
        raise ValueError("Exposure probability calibration needs 30 examples spanning all categories")
    raw_logits = np.asarray(classifier.decision_function(list(calibration_features)), dtype=float)
    calibrator = LogisticRegression(C=1.0, max_iter=500, random_state=2026)
    calibrator.fit(raw_logits, list(calibration_labels))
    return CalibratedExposureModel(classifier, calibrator)
