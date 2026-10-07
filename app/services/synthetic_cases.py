"""Small, explicitly synthetic correlation fixtures built from reserved names and bytes."""

from __future__ import annotations

from email.message import EmailMessage
from hashlib import sha256

from app.services.analyzers import EmailAnalyzer, FileAnalyzer, URLAnalyzer
from app.services.contracts import IndicatorResult, RelationshipResult
from app.services.feature_types import FeatureBundle
from app.services.correlation import CrossInputCorrelator
from app.services.threat_graph import NetworkXGraphProcessor


def _run_case(name: str, bundles: list[FeatureBundle], expected: list[str]) -> dict:
    correlation = CrossInputCorrelator().correlate(bundles)
    graph = NetworkXGraphProcessor().process_correlation(correlation)
    return {
        "name": name,
        "synthetic": True,
        "input_artifacts": [bundle.to_dict() for bundle in bundles],
        "expected_relationship_types": expected,
        "result": graph.to_dict(max_paths=20),
    }


def build_synthetic_correlation_cases() -> list[dict]:
    """Return safe `.test` fixture data; no live domains, samples, or network requests."""
    attached_qr_bytes = b"Synthetic QR fixture payload; not an executable file."
    message = EmailMessage()
    message["From"] = "sender@example.test"
    message["To"] = "recipient@example.test"
    message["Subject"] = "Synthetic relationship fixture"
    message.set_content("Please inspect the attached synthetic QR reference.")
    message.add_attachment(attached_qr_bytes, maintype="application", subtype="octet-stream",
                           filename="synthetic-qr-fixture.bin")
    email_bundle = EmailAnalyzer().analyze({"id": "synthetic-email-qr", "submitted_value": message.as_string()})
    destination = "https://landing.example.test/continue?case=synthetic"
    url_bundle = URLAnalyzer().analyze({"id": "synthetic-destination", "submitted_value": destination})
    digest = sha256(attached_qr_bytes).hexdigest()
    file_bundle = FeatureBundle(
        analyzer="synthetic_file_fixture", artifact_kind="file", artifact_id="synthetic-attached-file",
        features={"synthetic_fixture": True, "file.name": "synthetic-qr-fixture.bin",
                  "file.sha256": digest},
        indicators=(IndicatorResult("file_hash_sha256", digest, digest),),
        limitations=("Synthetic file analysis fixture; no attachment bytes are retained.",),
    )
    qr_payload = IndicatorResult("qr_payload", destination, destination)
    decoded_url = next(item for item in url_bundle.indicators if item.type == "url")
    qr_bundle = FeatureBundle(
        analyzer="synthetic_qr_fixture", artifact_kind="qr", artifact_id="synthetic-attached-qr",
        features={"synthetic_fixture": True, "qr.decoded_count": 1,
                  "qr.payloads": [destination], "qr.url_payloads": [destination]},
        indicators=(IndicatorResult("file_hash_sha256", digest, digest), qr_payload, decoded_url),
        relationships=(RelationshipResult(qr_payload, decoded_url, "encodes"),),
        limitations=("Synthetic decoded result; no image decoding was performed.",),
    )
    chain_case = _run_case(
        "email-attached-qr-to-url",
        [email_bundle, file_bundle, qr_bundle, url_bundle],
        ["ATTACHED_TO", "RELATED_TO", "DECODED_TO", "BELONGS_TO"],
    )

    ip_destination = "https://192.0.2.17/synthetic-landing"
    ip_url_bundle = URLAnalyzer().analyze({"id": "synthetic-ip-destination", "submitted_value": ip_destination})
    ip_payload = IndicatorResult("qr_payload", ip_destination, ip_destination)
    ip_decoded_url = next(item for item in ip_url_bundle.indicators if item.type == "url")
    ip_qr_bundle = FeatureBundle(
        analyzer="synthetic_qr_fixture", artifact_kind="qr", artifact_id="synthetic-attached-qr-ip",
        features={"synthetic_fixture": True, "qr.decoded_count": 1,
                  "qr.payloads": [ip_destination], "qr.url_payloads": [ip_destination]},
        indicators=(IndicatorResult("file_hash_sha256", digest, digest), ip_payload, ip_decoded_url),
        relationships=(RelationshipResult(ip_payload, ip_decoded_url, "encodes"),),
        limitations=("Synthetic decoded result; no image decoding was performed.",),
    )
    ip_chain_case = _run_case(
        "email-file-qr-to-ip-url",
        [email_bundle, file_bundle, ip_qr_bundle, ip_url_bundle],
        ["ATTACHED_TO", "RELATED_TO", "DECODED_TO", "BELONGS_TO"],
    )

    body_url = "https://docs.example.test/reference"
    message2 = EmailMessage()
    message2["From"] = "analyst@example.test"
    message2["To"] = "reader@example.test"
    message2["Subject"] = "Synthetic linked document"
    message2.set_content(f"Reference: {body_url}")
    email2 = EmailAnalyzer().analyze({"id": "synthetic-email-link", "submitted_value": message2.as_string()})
    file2 = FileAnalyzer().analyze({
        "id": "synthetic-text-file", "name": "notes.txt", "media_type": "text/plain",
        "content": f"Documentation URL: {body_url}".encode("utf-8"),
    })
    url2 = URLAnalyzer().analyze({"id": "synthetic-doc-url", "submitted_value": body_url})
    shared_case = _run_case(
        "email-and-file-share-url",
        [email2, file2, url2],
        ["LINKS_TO", "HAS_HASH", "BELONGS_TO"],
    )
    return [chain_case, ip_chain_case, shared_case]
