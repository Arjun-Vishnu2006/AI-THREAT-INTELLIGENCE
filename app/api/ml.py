from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import ModelMetadata
from app.schemas.records import ModelMetadataRead
from app.ml.baseline import predict_behavior_features
from app.ml.registry import load_registered_pipeline

router = APIRouter(prefix="/models", tags=["model metadata"])


class BehaviorPredictionRequest(BaseModel):
    features: dict[str, str | int | float | bool] = Field(min_length=1, max_length=64)


@router.get("", response_model=list[ModelMetadataRead])
def list_models(db: Session = Depends(get_db)) -> list[ModelMetadata]:
    return list(db.scalars(select(ModelMetadata).order_by(ModelMetadata.name, ModelMetadata.version)))


@router.get("/{name}/{version}", response_model=ModelMetadataRead)
def get_model(name: str, version: str, db: Session = Depends(get_db)) -> ModelMetadata:
    model = db.scalar(select(ModelMetadata).where(
        ModelMetadata.name == name, ModelMetadata.version == version,
    ))
    if model is None:
        raise HTTPException(status_code=404, detail="Model version not found")
    return model


@router.post("/{name}/{version}/predict")
def predict_behavior(name: str, version: str, payload: BehaviorPredictionRequest,
                     db: Session = Depends(get_db)) -> dict[str, str]:
    model = db.scalar(select(ModelMetadata).where(
        ModelMetadata.name == name, ModelMetadata.version == version,
    ))
    if model is None:
        raise HTTPException(status_code=404, detail="Model version not found")
    if model.task != "synthetic_behavior_classification":
        raise HTTPException(status_code=409, detail="Model task is not behavior classification")
    try:
        pipeline = load_registered_pipeline(model)
        return predict_behavior_features(pipeline, payload.features)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
