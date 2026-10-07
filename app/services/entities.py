"""Canonical indicator identities shared by correlation and graph processing."""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass
from uuid import NAMESPACE_URL, uuid5

from app.services.contracts import IndicatorResult
from app.services.analyzers.common import domain_of, normalize_url

_ALIASES = {
    "host": "domain", "hostname": "domain", "fqdn": "domain",
    "sha256": "file_hash_sha256", "hash": "file_hash_sha256",
    "file_hash": "file_hash_sha256", "file_hash_sha256": "file_hash_sha256",
    "ipv4": "ip", "ipv6": "ip", "email_address": "email",
}


@dataclass(frozen=True)
class NormalizedEntity:
    id: str
    type: str
    value: str
    normalized_value: str
    attributes: dict

    def to_dict(self) -> dict:
        return {"id": self.id, "type": self.type, "value": self.value,
                "normalized_value": self.normalized_value, "attributes": self.attributes}


def normalize_indicator(item: IndicatorResult) -> NormalizedEntity:
    kind = _ALIASES.get(item.type.casefold(), item.type.casefold())
    value = item.value.strip()
    normalized = item.normalized_value.strip() or value
    if kind == "domain":
        normalized = domain_of(normalized) or normalized.casefold()
        value = domain_of(value) or value.casefold()
    elif kind == "ip":
        try:
            normalized = ipaddress.ip_address(normalized.strip("[]")).compressed
            value = ipaddress.ip_address(value.strip("[]")).compressed
        except ValueError:
            normalized = normalized.casefold()
    elif kind == "email":
        if "@" in normalized:
            local, host = normalized.rsplit("@", 1)
            normalized = f"{local.casefold()}@{domain_of(host) or host.casefold()}"
        else:
            normalized = normalized.casefold()
    elif kind == "url":
        normalized = normalize_url(normalized)
        value = normalize_url(value)
    elif kind == "file_hash_sha256":
        normalized = normalized.casefold()
        value = value.casefold()
    elif kind not in {"qr_payload", "artifact", "attachment"}:
        normalized = normalized.casefold()
    identity = f"threat-entity:{kind}:{normalized}"
    return NormalizedEntity(str(uuid5(NAMESPACE_URL, identity)), kind, value, normalized,
                            dict(item.attributes))


def artifact_entity(bundle_kind: str, artifact_id: str) -> NormalizedEntity:
    kind = "artifact"
    normalized = f"{bundle_kind.casefold()}:{artifact_id}"
    identity = f"threat-entity:{kind}:{normalized}"
    return NormalizedEntity(str(uuid5(NAMESPACE_URL, identity)), kind, artifact_id, normalized,
                            {"artifact_kind": bundle_kind.casefold()})
