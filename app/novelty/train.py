"""Train, calibrate, evaluate, and register a synthetic multi-level novelty model."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import joblib

from app.config import get_settings
from app.db import Base, SessionLocal, engine
from app.models import ExperimentMetadata, ModelMetadata
from app.novelty.training import train_and_evaluate_novelty

_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def train_and_register(version: str, seed: int = 2026, alpha: float = 0.05) -> dict:
    if not _VERSION_RE.fullmatch(version) or version in {".", ".."}:
        raise ValueError("Model version must contain only letters, digits, dot, underscore, or hyphen")
    root = Path(get_settings().model_registry_dir).resolve()
    version_dir = root / "novelty-fusion" / version
    artifact_path = version_dir / "engine.joblib"
    if version_dir.exists():
        raise FileExistsError(f"Novelty model version already exists and is immutable: {version_dir}")
    novelty_engine, metadata = train_and_evaluate_novelty(seed=seed, alpha=alpha)
    version_dir.mkdir(parents=True, exist_ok=False)
    joblib.dump(novelty_engine, artifact_path)
    metadata_path = version_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        db.add(ModelMetadata(
            name="novelty-fusion", version=version,
            task="multi_level_novelty_detection", artifact_ref=str(artifact_path),
            parameters={"alpha": alpha, "fusion_method": metadata["fusion_method"],
                        "seed": seed, "layers": metadata["versioned_layers"]},
            provenance=metadata,
        ))
        db.add(ExperimentMetadata(
            name="novelty-fusion-ablation",
            version=version,
            status="completed",
            configuration={
                "seed": seed,
                "alpha": alpha,
                "dataset_id": metadata["dataset_id"],
                "dataset_fingerprint_sha256": metadata["dataset_fingerprint_sha256"],
                "layers": metadata["versioned_layers"],
                "split_campaigns": metadata["split_campaigns"],
                "group_leakage_checked": metadata["group_leakage_checked"],
            },
            results={
                "held_out_metrics": metadata["held_out_metrics"],
                "single_layer_metrics": metadata["single_layer_metrics"],
                "ablation_metrics": metadata["ablation_metrics"],
                "synthetic_only": metadata["synthetic_only"],
            },
        ))
        db.commit()
    return {"name": "novelty-fusion", "version": version,
            "artifact_ref": str(artifact_path), "metadata_ref": str(metadata_path),
            "metadata": metadata}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="0.1.0", help="Immutable model version, e.g. 0.1.0")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()
    print(json.dumps(train_and_register(args.version, args.seed, args.alpha), indent=2))


if __name__ == "__main__":
    main()
