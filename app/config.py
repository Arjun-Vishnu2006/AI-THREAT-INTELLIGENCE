from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Threat Analysis API"
    environment: str = "development"
    api_v1_prefix: str = "/api/v1"
    database_url: str = "sqlite:///./threat_analysis.db"
    threat_intel_path: str = "data/ioc_baseline.json"
    model_registry_dir: str = "./models"
    log_level: str = "INFO"
    cors_origins: list[str] = Field(default_factory=list)

    @field_validator("api_v1_prefix")
    @classmethod
    def normalize_prefix(cls, value: str) -> str:
        return "/" + value.strip("/")


@lru_cache
def get_settings() -> Settings:
    return Settings()
