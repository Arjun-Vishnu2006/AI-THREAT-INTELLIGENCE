from __future__ import annotations

import re
from email import policy
from email.parser import BytesParser
from html.parser import HTMLParser
from typing import Any

from app.services.analyzers.common import (
    EMAIL_RE, URL_RE, artifact_id, domain_of, evidence, indicator, normalize_url, safe_bytes, sha256,
)
from app.services.contracts import AnalysisContext
from app.services.feature_types import FeatureBundle

_URGENCY_TERMS = re.compile(r"(?i)\b(urgent|immediately|account suspended|verify (?:your )?account|password expires|payment failed|action required)\b")


class _LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[str] = []
        self.text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.casefold() in {"a", "area", "iframe", "form"}:
            for name, value in attrs:
                if name.casefold() in {"href", "src", "action"} and value:
                    self.links.append(value.strip())

    def handle_data(self, data: str) -> None:
        self.text.append(data)


class EmailAnalyzer:
    name = "email"
    supported_kinds = frozenset({"email"})

    def analyze(self, artifact: dict[str, Any], context: AnalysisContext | None = None) -> FeatureBundle:
        raw = next((artifact[key] for key in ("content", "submitted_value", "raw_email") if key in artifact and artifact[key] is not None), None)
        if raw is None:
            raise ValueError("Email artifact requires content or submitted_value")
        data = safe_bytes(raw)
        message = BytesParser(policy=policy.default).parsebytes(data)
        sender = message.get("From")
        reply_to = message.get("Reply-To")
        return_path = message.get("Return-Path")
        from_addresses = [a.addr_spec for a in (message["From"].addresses if message["From"] else [])]
        reply_addresses = [a.addr_spec for a in (message["Reply-To"].addresses if message["Reply-To"] else [])]
        from_domains = [domain_of(a.rsplit("@", 1)[-1]) for a in from_addresses if "@" in a]
        reply_domains = [domain_of(a.rsplit("@", 1)[-1]) for a in reply_addresses if "@" in a]
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
                attachment_records.append({
                    "filename": filename,
                    "content_type": content_type,
                    "size_bytes": len(payload),
                    "sha256": sha256(payload),
                })
                continue
            if content_type in {"text/plain", "text/html"}:
                charset = part.get_content_charset() or "utf-8"
                decoded = payload.decode(charset, errors="replace")
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
        body_urls = [m.group(0).rstrip(".,);]>") for m in URL_RE.finditer(body)]
        links = list(dict.fromkeys(body_urls + html_links))
        email_addresses = list(dict.fromkeys(EMAIL_RE.findall(body)))
        received = message.get_all("Received", []) or []
        auth_results = [str(v) for v in message.get_all("Authentication-Results", [])]
        subject = str(message.get("Subject", ""))
        urgency_matches = list(dict.fromkeys(m.group(0).casefold() for m in _URGENCY_TERMS.finditer(subject + "\n" + body)))
        features = {
            "email.message_id": str(message.get("Message-ID", "")) or None,
            "email.subject": subject,
            "email.from_header": str(sender) if sender else None,
            "email.from_addresses": from_addresses,
            "email.from_domains": from_domains,
            "email.reply_to_header": str(reply_to) if reply_to else None,
            "email.reply_to_addresses": reply_addresses,
            "email.reply_to_domains": reply_domains,
            "email.return_path": str(return_path) if return_path else None,
            "email.reply_to_domain_differs_from_from": bool(from_domains and reply_domains and set(from_domains).isdisjoint(reply_domains)),
            "email.received_hop_count": len(received),
            "email.authentication_results": auth_results,
            "email.header_names": list(dict.fromkeys(k.casefold() for k in message.keys())),
            "email.body_character_count": len(body),
            "email.body_word_count": len(body.split()),
            "email.urls": links,
            "email.embedded_email_addresses": email_addresses,
            "email.attachments": attachment_records,
            "email.urgency_language_matches": urgency_matches,
        }
        indicators = []
        for addr in from_addresses + reply_addresses + email_addresses:
            indicators.append(indicator("email", addr, normalized=addr.casefold()))
        for url in links:
            indicators.append(indicator("url", url, normalized=normalize_url(url)))
        observations = []
        if features["email.reply_to_domain_differs_from_from"]:
            observations.append(evidence("email_header", "Reply-To domain differs from the From domain.", attributes={"from_domains": from_domains, "reply_to_domains": reply_domains}))
        if urgency_matches:
            observations.append(evidence("email_body", "Message contains urgency-related language matches.", attributes={"matches": urgency_matches}))
        if attachments := attachment_records:
            observations.append(evidence("email_attachment", "Email contains attachment(s); payloads were hashed but not executed.", attributes={"count": len(attachments)}))
        return FeatureBundle(self.name, "email", artifact_id(artifact), features, tuple(indicators), evidence=tuple(observations), limitations=("Email authentication results are reported from headers and are not independently verified.", "Attached files are not executed; use the file analyzer for static inspection."))
