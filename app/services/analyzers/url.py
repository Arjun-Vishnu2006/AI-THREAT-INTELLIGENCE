from __future__ import annotations

import ipaddress
import re
from urllib.parse import parse_qsl, unquote, urlsplit

from app.services.analyzers.common import (
    artifact_id, domain_of, evidence, host_is_ip, indicator, normalize_url,
)
from app.services.contracts import AnalysisContext
from app.services.feature_types import FeatureBundle

_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
_HEX_ESCAPE_RE = re.compile(r"%[0-9a-fA-F]{2}")


class URLAnalyzer:
    name = "url"
    supported_kinds = frozenset({"url"})

    def analyze(self, artifact: dict, context: AnalysisContext | None = None) -> FeatureBundle:
        raw = str(artifact.get("submitted_value") or artifact.get("url") or "").strip()
        if not raw:
            raise ValueError("URL artifact has no submitted_value")
        candidate = raw if _SCHEME_RE.match(raw) else f"http://{raw}"
        try:
            parsed = urlsplit(candidate)
            hostname = parsed.hostname
            port = parsed.port
        except ValueError as exc:
            raise ValueError(f"URL could not be parsed: {exc}") from exc

        host = domain_of(hostname)
        normalized_url = normalize_url(raw)
        labels = host.split(".") if host else []
        path = parsed.path or "/"
        query_pairs = parse_qsl(parsed.query, keep_blank_values=True)
        decoded_path = unquote(path)
        path_suffix = decoded_path.rsplit("/", 1)[-1].rsplit(".", 1)
        suffix = path_suffix[-1].casefold() if len(path_suffix) == 2 else ""
        is_ip = host_is_ip(host)
        try:
            ip_version = ipaddress.ip_address(host).version if is_ip else None
        except ValueError:
            ip_version = None
        features = {
            "url.length": len(raw),
            "url.scheme": parsed.scheme.casefold(),
            "url.has_explicit_scheme": bool(_SCHEME_RE.match(raw)),
            "url.has_userinfo": parsed.username is not None or parsed.password is not None,
            "url.host": host,
            "url.host_is_ip": is_ip,
            "url.ip_version": ip_version,
            "url.host_length": len(host or ""),
            "url.host_label_count": len(labels),
            "url.longest_host_label": max((len(label) for label in labels), default=0),
            "url.has_punycode_label": any(label.startswith("xn--") for label in labels),
            "url.subdomain_label_count": max(0, len(labels) - 2) if not is_ip else 0,
            "url.port": port,
            "url.path_length": len(path),
            "url.path_segment_count": len([p for p in path.split("/") if p]),
            "url.path_extension": suffix,
            "url.query_length": len(parsed.query),
            "url.query_parameter_count": len(query_pairs),
            "url.query_parameter_names": [key for key, _ in query_pairs],
            "url.fragment_present": bool(parsed.fragment),
            "url.at_symbol_count": raw.count("@"),
            "url.dot_count": raw.count("."),
            "url.hyphen_count": raw.count("-"),
            "url.percent_escape_count": len(_HEX_ESCAPE_RE.findall(raw)),
            "url.contains_non_ascii": any(ord(ch) > 127 for ch in raw),
            "url.redirect_parameter_names": [k for k, _ in query_pairs if k.casefold() in {"url", "redirect", "redirect_uri", "next", "continue", "return", "returnto", "target"}],
            "url.network_requested": False,
        }
        indicators = []
        if host:
            kind = "ip" if is_ip else "domain"
            indicators.append(indicator(kind, host, normalized=host))
        indicators.append(indicator("url", raw, normalized=normalized_url))
        observations = [evidence(
            "url_analysis",
            "URL was structurally analyzed without making a network request.",
            attributes={"normalized_url": normalized_url},
        )]
        if features["url.has_userinfo"]:
            observations.append(evidence("url_structure", "URL contains user information before the host."))
        if features["url.has_punycode_label"]:
            observations.append(evidence("url_structure", "Host contains an IDNA punycode label."))
        if features["url.redirect_parameter_names"]:
            observations.append(evidence("url_structure", "Query contains a parameter commonly used to carry a destination URL.", attributes={"parameter_names": features["url.redirect_parameter_names"]}))
        return FeatureBundle(self.name, "url", artifact_id(artifact), features, tuple(indicators), evidence=tuple(observations), limitations=("No DNS lookup, HTTP request, redirect following, or reputation lookup was performed.",))
