from __future__ import annotations

import hashlib
import ipaddress
import re
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from app.services.contracts import Evidence, IndicatorResult

MAX_ARTIFACT_BYTES = 50 * 1024 * 1024
URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)[^\s<>\"']+")
EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")


def artifact_id(artifact: dict[str, Any]) -> str | None:
    value = artifact.get("id")
    return str(value) if value is not None else None


def indicator(kind: str, value: str, *, normalized: str | None = None,
              attributes: dict[str, Any] | None = None) -> IndicatorResult:
    text = value.strip()
    return IndicatorResult(kind, text, normalized if normalized is not None else text.casefold(), attributes or {})


def domain_of(host: str | None) -> str | None:
    if not host:
        return None
    host = host.rstrip(".").casefold()
    try:
        return host.encode("idna").decode("ascii")
    except UnicodeError:
        return host


def normalize_url(value: str) -> str:
    """Canonicalize URL identity without contacting the submitted destination."""
    try:
        parsed = urlsplit(value if "://" in value else "http://" + value)
        host = domain_of(parsed.hostname) if parsed.hostname else ""
        if not host:
            return value.strip()
        scheme = parsed.scheme.casefold()
        port = parsed.port
        netloc = host
        if ":" in host and not host.startswith("["):
            netloc = f"[{host}]"
        if port and not (scheme == "http" and port == 80) and not (scheme == "https" and port == 443):
            netloc += f":{port}"
        if parsed.username is not None:
            userinfo = parsed.username
            if parsed.password is not None:
                userinfo += ":" + parsed.password
            netloc = f"{userinfo}@{netloc}"
        return urlunsplit((scheme, netloc, parsed.path or "/", parsed.query, parsed.fragment))
    except ValueError:
        return value.strip()


def host_is_ip(host: str | None) -> bool:
    if not host:
        return False
    try:
        ipaddress.ip_address(host.strip("[]"))
        return True
    except ValueError:
        return False


def safe_bytes(value: Any, *, limit: int = MAX_ARTIFACT_BYTES) -> bytes:
    if isinstance(value, bytes):
        data = value
    elif isinstance(value, bytearray):
        data = bytes(value)
    elif isinstance(value, str):
        data = value.encode("utf-8", errors="replace")
    else:
        raise TypeError("artifact content must be bytes, bytearray, or text")
    if len(data) > limit:
        raise ValueError(f"artifact exceeds the {limit}-byte analysis limit")
    return data


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def evidence(kind: str, summary: str, *, refs: tuple[str, ...] = (),
             attributes: dict[str, Any] | None = None) -> Evidence:
    return Evidence(kind=kind, summary=summary, references=refs, attributes=attributes or {})
