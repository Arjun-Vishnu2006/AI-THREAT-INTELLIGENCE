from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["health"])


class HealthRead(BaseModel):
    status: str
    service: str


@router.get("/health", response_model=HealthRead)
def health() -> HealthRead:
    return HealthRead(status="ok", service="threat-analysis-api")
