"""Allowlisted, non-identity ML feature extraction from analyzer bundles."""

from typing import Any

from app.services.feature_types import FeatureBundle


def extract_ml_features(bundle: FeatureBundle) -> dict[str, str | int | float]:
    source: dict[str, Any] = bundle.features
    features: dict[str, str | int | float] = {"artifact.kind": bundle.artifact_kind}
    categorical = {
        "url.scheme": "url.scheme",
        "file.detected_type": "file.detected_type",
        "file.suffix": "file.suffix",
    }
    for output_key, source_key in categorical.items():
        value = source.get(source_key)
        if value:
            features[output_key] = str(value)
    numeric = {
        "url.length": "url.length",
        "url.host_label_count": "url.host_label_count",
        "url.subdomain_label_count": "url.subdomain_label_count",
        "url.path_length": "url.path_length",
        "url.query_parameter_count": "url.query_parameter_count",
        "url.percent_escape_count": "url.percent_escape_count",
        "url.has_userinfo": "url.has_userinfo",
        "email.received_hop_count": "email.received_hop_count",
        "email.body_character_count": "email.body_character_count",
        "email.links.count": "email.urls",
        "email.attachments.count": "email.attachments",
        "email.urgency_matches.count": "email.urgency_language_matches",
        "email.reply_domain_differs": "email.reply_to_domain_differs_from_from",
        "file.size_bytes": "file.size_bytes",
        "file.declared_mime_mismatch": "file.declared_mime_mismatch",
        "file.script_like_extension": "file.script_like_extension",
        "file.archive_entry_count": "file.archive_entry_count",
        "qr.decoded_count": "qr.decoded_count",
        "qr.has_url_payload": "qr.has_url_payload",
    }
    for output_key, source_key in numeric.items():
        value = source.get(source_key)
        if isinstance(value, bool):
            features[output_key] = int(value)
        elif isinstance(value, (int, float)):
            features[output_key] = value
        elif isinstance(value, list):
            features[output_key] = len(value)
    return features
