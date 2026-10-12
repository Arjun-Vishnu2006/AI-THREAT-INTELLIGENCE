from __future__ import annotations

import re
from email import policy
from email.parser import BytesParser
from html.parser import HTMLParser
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib

from app.services.analyzers.common import (
    EMAIL_RE,
    URL_RE,
    artifact_id,
    domain_of,
    evidence,
    indicator,
    normalize_url,
    safe_bytes,
    sha256,
)
from app.services.contracts import AnalysisContext
from app.services.feature_types import FeatureBundle


_URGENCY_TERMS = re.compile(
    r"(?i)\b(urgent|immediately|account suspended|verify (?:your )?account|"
    r"password expires|payment failed|action required)\b"
)


@lru_cache(maxsize=1)
def _load_email_model():
    """Load the trained model once; return None if the artifact is unavailable."""
    model_path = (
        Path(__file__).resolve().parents[3]
        / "artifacts"
        / "email_model"
        / "email_spam_model.joblib"
    )

    if not model_path.is_file():
        return None

    return joblib.load(model_path)


class _LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[str] = []
        self.text: list[str] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        if tag.casefold() in {"a", "area", "iframe", "form"}:
            for name, value in attrs:
                if name.casefold() in {"href", "src", "action"} and value:
                    self.links.append(value.strip())

    def handle_data(self, data: str) -> None:
        self.text.append(data)


class EmailAnalyzer:
    name = "email"
    supported_kinds = frozenset({"email"})

    def analyze(
        self,
        artifact: dict[str, Any],
        context: AnalysisContext | None = None,
    ) -> FeatureBundle:
        raw = next(
            (
                artifact[key]
                for key in ("content", "submitted_value", "raw_email")
                if key in artifact and artifact[key] is not None
            ),
            None,
        )

        if raw is None:
            raise ValueError(
                "Email artifact requires content or submitted_value"
            )

        data = safe_bytes(raw)
        message = BytesParser(policy=policy.default).parsebytes(data)

        sender = message.get("From")
        reply_to = message.get("Reply-To")
        return_path = message.get("Return-Path")

        from_addresses = [
            address.addr_spec
            for address in (
                message["From"].addresses if message["From"] else []
            )
        ]
        reply_addresses = [
            address.addr_spec
            for address in (
                message["Reply-To"].addresses if message["Reply-To"] else []
            )
        ]

        from_domains = [
            domain_of(address.rsplit("@", 1)[-1])
            for address in from_addresses
            if "@" in address
        ]
        reply_domains = [
            domain_of(address.rsplit("@", 1)[-1])
            for address in reply_addresses
            if "@" in address
        ]

        body_texts: list[str] = []
        html_links: list[str] = []
        attachment_records: list[dict[str, Any]] = []

        for part in message.walk():
            if part.is_multipart():
                continue

            payload = part.get_payload(decode=True) or b""
            content_type = part.get_content_type().casefold()
            disposition = part.get_content_disposition()
            filename = part.get_filename()

            if disposition == "attachment" or filename:
                attachment_records.append(
                    {
                        "filename": filename,
                        "content_type": content_type,
                        "size_bytes": len(payload),
                        "sha256": sha256(payload),
                    }
                )
                continue

            if content_type in {"text/plain", "text/html"}:
                charset = part.get_content_charset() or "utf-8"

                try:
                    decoded = payload.decode(charset, errors="replace")
                except (LookupError, UnicodeError):
                    decoded = payload.decode("utf-8", errors="replace")

                body_texts.append(decoded)

                if content_type == "text/html":
                    parser = _LinkParser()
                    try:
                        parser.feed(decoded)
                        html_links.extend(parser.links)
                        body_texts.append(" ".join(parser.text))
                    except Exception:
                        pass

        body = "\n".join(body_texts)
        subject = str(message.get("Subject", ""))

        # The trained model uses subject + readable message body.
        # It does not use sender reputation or attachment contents.
        model_text = (subject + "\n" + body).strip()

        body_urls = [
            match.group(0).rstrip(".,);]>")
            for match in URL_RE.finditer(body)
        ]
        links = list(dict.fromkeys(body_urls + html_links))
        email_addresses = list(dict.fromkeys(EMAIL_RE.findall(body)))

        received = message.get_all("Received", []) or []
        auth_results = [
            str(value)
            for value in message.get_all("Authentication-Results", [])
        ]

        urgency_matches = list(
            dict.fromkeys(
                match.group(0).casefold()
                for match in _URGENCY_TERMS.finditer(
                    subject + "\n" + body
                )
            )
        )

        features: dict[str, Any] = {
            "email.message_id": str(message.get("Message-ID", "")) or None,
            "email.subject": subject,
            "email.from_header": str(sender) if sender else None,
            "email.from_addresses": from_addresses,
            "email.from_domains": from_domains,
            "email.reply_to_header": str(reply_to) if reply_to else None,
            "email.reply_to_addresses": reply_addresses,
            "email.reply_to_domains": reply_domains,
            "email.return_path": str(return_path) if return_path else None,
            "email.reply_to_domain_differs_from_from": bool(
                from_domains
                and reply_domains
                and set(from_domains).isdisjoint(reply_domains)
            ),
            "email.received_hop_count": len(received),
            "email.authentication_results": auth_results,
            "email.header_names": list(
                dict.fromkeys(key.casefold() for key in message.keys())
            ),
            "email.body_character_count": len(body),
            "email.body_word_count": len(body.split()),
            "email.urls": links,
            "email.embedded_email_addresses": email_addresses,
            "email.attachments": attachment_records,
            "email.urgency_language_matches": urgency_matches,
        }

        indicators = []

        for address in from_addresses + reply_addresses + email_addresses:
            indicators.append(
                indicator(
                    "email",
                    address,
                    normalized=address.casefold(),
                )
            )

        for url in links:
            indicators.append(
                indicator(
                    "url",
                    url,
                    normalized=normalize_url(url),
                )
            )

        observations = []

        if features["email.reply_to_domain_differs_from_from"]:
            observations.append(
                evidence(
                    "email_header",
                    "Reply-To domain differs from the From domain.",
                    attributes={
                        "from_domains": from_domains,
                        "reply_to_domains": reply_domains,
                    },
                )
            )

        if urgency_matches:
            observations.append(
                evidence(
                    "email_body",
                    "Message contains urgency-related language matches.",
                    attributes={"matches": urgency_matches},
                )
            )

        if attachment_records:
            observations.append(
                evidence(
                    "email_attachment",
                    "Email contains attachment(s); payloads were hashed but not executed.",
                    attributes={"count": len(attachment_records)},
                )
            )

        limitations = [
            "Email authentication results are reported from headers and are not independently verified.",
            "Attached files are not executed; use the file analyzer for static inspection.",
            "The ML model predicts spam versus ham, not verified phishing.",
            "The model uses email subject and readable body text only.",
            "Evaluation used a historical SpamAssassin corpus; performance on modern emails may differ.",
        ]

        # ML prediction is an additional evidence source.
        # Rule-based analysis remains available if model loading or prediction fails.
        try:
            model = _load_email_model()

            if model is None:
                features["email.ml_status"] = "unavailable"
                limitations.append("Email ML model file was not found.")
            elif not model_text:
                features["email.ml_status"] = "skipped"
                limitations.append(
                    "Email ML prediction was skipped because subject and body were empty."
                )
            else:
                prediction = int(model.predict([model_text])[0])
                probabilities = model.predict_proba([model_text])[0]
                classes = list(model.classes_)

                if 1 not in classes:
                    raise ValueError(
                        "Email model does not contain the expected spam class 1."
                    )

                spam_probability = float(
                    probabilities[classes.index(1)]
                )
                prediction_label = "SPAM" if prediction == 1 else "HAM"

                features["email.ml_status"] = "available"
                features["email.ml_prediction"] = prediction_label
                features["email.ml_spam_probability"] = round(
                    spam_probability, 6
                )

                observations.append(
                    evidence(
                        "email_ml_prediction",
                        (
                            "The trained email classifier classified this message as spam."
                            if prediction == 1
                            else "The trained email classifier classified this message as ham."
                        ),
                        attributes={
                            "prediction": prediction_label,
                            "spam_probability": round(spam_probability, 6),
                            "model": "TF-IDF + Logistic Regression",
                            "dataset": "SpamAssassin public corpus",
                            "label_scope": "spam versus ham; not verified phishing",
                        },
                    )
                )

        except Exception as exc:
            features["email.ml_status"] = "error"
            limitations.append(
                "Email ML prediction failed; rule-based email analysis remains available."
            )
            observations.append(
                evidence(
                    "email_ml_error",
                    "The email ML classifier could not produce a prediction.",
                    attributes={"error_type": type(exc).__name__},
                )
            )

        return FeatureBundle(
            self.name,
            "email",
            artifact_id(artifact),
            features,
            tuple(indicators),
            evidence=tuple(observations),
            limitations=tuple(limitations),
        )
