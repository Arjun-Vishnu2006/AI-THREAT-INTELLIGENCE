"""Train, calibrate, evaluate and register a synthetic exposure category model."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import joblib

from app.config import get_settings
from app.db import Base, SessionLocal, engine
from app.exposure.training import train_and_evaluate_exposure
from app.models import ExperimentMetadata, ModelMetadata

_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def train_and_register(version: str, seed: int = 2026) -> dict:
    if not _VERSION_RE.fullmatch(version) or version in {".", ".."}:
        raise ValueError("Model version must contain only letters, digits, dot, underscore, or hyphen")
    root = Path(get_settings().model_registry_dir).resolve()
    version_dir = root / "exposure-predictor" / version
    artifact_path = version_dir / "model.joblib"
    if version_dir.exists():
        raise FileExistsError(f"Exposure model version already exists and is immutable: {version_dir}")
    model, metadata = train_and_evaluate_exposure(seed=seed)
    version_dir.mkdir(parents=True, exist_ok=False)
    joblib.dump(model, artifact_path)
    metadata_path = version_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        db.add(ModelMetadata(
            name="exposure-predictor", version=version,
            task="exposure_category_prediction", artifact_ref=str(artifact_path),
            parameters={"seed": seed, "calibration_method": metadata["calibration_method"],
                        "categories": metadata["categories"]},
            provenance=metadata,
        ))
        db.add(ExperimentMetadata(
            name="exposure-category-calibration", version=version, status="completed",
            configuration={"seed": seed, "dataset_id": metadata["dataset_id"],
                           "dataset_fingerprint_sha256": metadata["dataset_fingerprint_sha256"],
                           "split_campaigns": metadata["split_campaigns"],
                           "group_leakage_checked": metadata["group_leakage_checked"]},
            results={"held_out_metrics": metadata["held_out_metrics"],
                     "calibration_method": metadata["calibration_method"],
                     "label_semantics": metadata["label_semantics"]},
        ))
        db.commit()
    return {"name": "exposure-predictor", "version": version,
            "artifact_ref": str(artifact_path), "metadata_ref": str(metadata_path),
            "metadata": metadata}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="0.1.0", help="Immutable model version, e.g. 0.1.0")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    print(json.dumps(train_and_register(args.version, args.seed), indent=2))


if __name__ == "__main__":
    main()
