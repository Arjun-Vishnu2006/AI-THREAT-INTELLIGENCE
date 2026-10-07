"""Configured local threat-intelligence provider shared by API consumers."""

from functools import lru_cache
from pathlib import Path

from app.config import get_settings
from app.services.threat_intel import LocalJSONThreatIntelProvider


@lru_cache(maxsize=1)
def get_threat_intel_provider() -> LocalJSONThreatIntelProvider:
    configured = Path(get_settings().threat_intel_path)
    if not configured.is_absolute():
        configured = Path(__file__).resolve().parents[2] / configured
    return LocalJSONThreatIntelProvider(configured)
