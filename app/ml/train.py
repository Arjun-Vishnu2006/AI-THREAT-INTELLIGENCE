"""Train and register an immutable, synthetic-only baseline model version."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import joblib

from app.config import get_settings
from app.db import Base, SessionLocal, engine
from app.models import ModelMetadata
from app.ml.baseline import train_baseline

_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def train_and_register(version: str, seed: int = 2026) -> dict:
    if not _VERSION_RE.fullmatch(version) or version in {".", ".."}:
        raise ValueError("Model version must contain only letters, digits, dot, underscore, or hyphen")
    root = Path(get_settings().model_registry_dir).resolve()
    version_dir = root / "behavior-baseline" / version
    artifact_path = version_dir / "model.joblib"
    if version_dir.exists():
        raise FileExistsError(f"Model version already exists and is immutable: {version_dir}")

    result = train_baseline(seed=seed)
    version_dir.mkdir(parents=True, exist_ok=False)
    joblib.dump(result.pipeline, artifact_path)
    metadata_path = version_dir / "metadata.json"
    metadata_path.write_text(json.dumps(result.metadata, indent=2), encoding="utf-8")

    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        row = ModelMetadata(
            name="behavior-baseline",
            version=version,
            task=result.metadata["task"],
            artifact_ref=str(artifact_path),
            parameters=result.metadata["model_parameters"],
            provenance=result.metadata,
        )
        db.add(row)
        db.commit()
    return {"name": "behavior-baseline", "version": version,
            "artifact_ref": str(artifact_path), "metadata_ref": str(metadata_path),
            "metadata": result.metadata}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="0.1.0", help="Immutable model version, e.g. 0.1.0")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    print(json.dumps(train_and_register(args.version, args.seed), indent=2))


if __name__ == "__main__":
    main()
