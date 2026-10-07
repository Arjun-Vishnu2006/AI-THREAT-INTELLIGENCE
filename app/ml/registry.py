"""Load locally registered model artifacts from the configured, trusted registry directory."""

from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline

from app.config import get_settings
from app.models import ModelMetadata


def load_registered_artifact(model: ModelMetadata):
    root = Path(get_settings().model_registry_dir).resolve()
    artifact = Path(model.artifact_ref or "").resolve()
    if not artifact.is_relative_to(root):
        raise ValueError("Model artifact is outside the configured local registry")
    if not artifact.is_file():
        raise FileNotFoundError("Registered model artifact is missing")
    # Only use artifacts created by the local trainer; joblib is not safe for untrusted files.
    return joblib.load(artifact)


def load_registered_pipeline(model: ModelMetadata) -> Pipeline:
    pipeline = load_registered_artifact(model)
    if not isinstance(pipeline, Pipeline):
        raise TypeError("Registered artifact is not a scikit-learn Pipeline")
    return pipeline
