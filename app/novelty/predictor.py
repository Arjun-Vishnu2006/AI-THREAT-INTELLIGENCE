"""
Prediction module for Android malware novelty detection.

Responsibilities:
- Load trained autoencoder
- Load scaler
- Load threshold
- Predict novelty
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Union

import numpy as np
import pandas as pd
import tensorflow as tf

from .preprocessing import FeaturePreprocessor
from .threshold import NoveltyThreshold


class NoveltyPredictor:
    """
    Uses the trained autoencoder to detect novel Android threats.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        scaler_path: Union[str, Path],
        threshold_path: Union[str, Path],
        feature_columns_path: Union[str, Path],
    ):

        self.model = tf.keras.models.load_model(model_path)

        self.preprocessor = FeaturePreprocessor()
        self.preprocessor.load_scaler(scaler_path)
        self.preprocessor.load_feature_columns(feature_columns_path)

        self.threshold = NoveltyThreshold()
        self.threshold.load(threshold_path)

    def predict(
        self,
        features: pd.DataFrame,
    ) -> Dict:
        """
        Predict novelty for one or more samples.

        Returns:
            reconstruction_error
            novelty_score
            is_novel
        """

        # Ensure correct feature order
        features = self.preprocessor.reorder_columns(features)

        # Scale features
        X = self.preprocessor.transform(features)

        # Reconstruction
        reconstructed = self.model.predict(
            X,
            verbose=0
        )

        # Reconstruction error
        errors = np.mean(
            np.square(X - reconstructed),
            axis=1
        )

        scores = self.threshold.novelty_score(errors)

        predictions = self.threshold.predict(errors)

        results = []

        for i in range(len(errors)):

            results.append({

                "reconstruction_error":
                    float(errors[i]),

                "novelty_score":
                    float(scores[i]),

                "is_novel":
                    bool(predictions[i])

            })

        return {
            "samples": results
        }


if __name__ == "__main__":

    predictor = NoveltyPredictor(

        model_path="../models/autoencoder.keras",

        scaler_path="../models/scaler.pkl",

        threshold_path="../models/threshold.json",

        feature_columns_path="../models/feature_columns.json",

    )

    sample = pd.DataFrame(
        np.random.rand(1, 470)
    )

    sample.columns = predictor.preprocessor.feature_columns

    result = predictor.predict(sample)

    print(result)