"""
Preprocessing utilities for Android malware novelty detection.

Responsibilities:
- Load feature datasets
- Filter required classes
- Separate features and labels
- Scale features
- Save/load scaler
- Preserve feature column order
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler


class FeaturePreprocessor:
    """
    Handles feature preprocessing for novelty detection.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns: List[str] = []

    def load_dataset(
        self,
        csv_path: str | Path,
        allowed_classes: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """
        Load a dataset and optionally filter by class.
        """

        df = pd.read_csv(csv_path)

        if allowed_classes is not None:
            df = df[df["Class"].isin(allowed_classes)]

        df = df.reset_index(drop=True)

        return df

    def split_features_labels(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Separate features from labels.
        """

        if "Class" not in df.columns:
            raise ValueError("Dataset must contain 'Class' column.")

        X = df.drop(columns=["Class", "Class_Name"], errors="ignore")
        y = df["Class"]

        self.feature_columns = X.columns.tolist()

        return X, y

    def fit_transform(
        self,
        X: pd.DataFrame,
    ):
        """
        Fit scaler and transform features.
        """

        return self.scaler.fit_transform(X)

    def transform(
        self,
        X: pd.DataFrame,
    ):
        """
        Transform using existing scaler.
        """

        return self.scaler.transform(X)

    def save_scaler(
        self,
        path: str | Path,
    ):
        """
        Save fitted scaler.
        """

        joblib.dump(self.scaler, path)

    def load_scaler(
        self,
        path: str | Path,
    ):
        """
        Load saved scaler.
        """

        self.scaler = joblib.load(path)

    def save_feature_columns(
        self,
        path: str | Path,
    ):
        """
        Save feature order.
        """

        pd.Series(self.feature_columns).to_json(path)

    def load_feature_columns(
        self,
        path: str | Path,
    ) -> List[str]:
        """
        Load feature order.
        """

        self.feature_columns = (
            pd.read_json(path, typ="series")
            .tolist()
        )

        return self.feature_columns

    def reorder_columns(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Ensure incoming features follow the
        same order used during training.
        """

        return X[self.feature_columns]


if __name__ == "__main__":

    pre = FeaturePreprocessor()

    df = pre.load_dataset(
        "../../data/train.csv",
        allowed_classes=[1, 2, 5],
    )

    X, y = pre.split_features_labels(df)

    X_scaled = pre.fit_transform(X)

    print("Samples :", len(X))
    print("Features:", X.shape[1])

    pre.save_scaler("../models/scaler.pkl")
    pre.save_feature_columns("../models/feature_columns.json")