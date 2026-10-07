from __future__ import annotations

from io import BytesIO
from typing import Any

from app.services.analyzers.common import artifact_id, domain_of, evidence, indicator, safe_bytes
from app.services.contracts import AnalysisContext, RelationshipResult
from app.services.feature_types import FeatureBundle
from app.services.analyzers.url import URLAnalyzer


class QRAnalyzer:
    name = "qr"
    supported_kinds = frozenset({"qr"})

    def analyze(self, artifact: dict[str, Any], context: AnalysisContext | None = None) -> FeatureBundle:
        raw = next((artifact[key] for key in ("content", "image_bytes") if key in artifact and artifact[key] is not None), None)
        if raw is None:
            raise ValueError("QR artifact requires image content bytes")
        data = safe_bytes(raw)
        try:
            from PIL import Image  # type: ignore[import-not-found]
            import cv2  # type: ignore[import-not-found]
            import numpy as np  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("QR decoding requires optional dependencies: pip install .[qr]") from exc
        try:
            with Image.open(BytesIO(data)) as source_image:
                width, height = source_image.size
                if width <= 0 or height <= 0 or width * height > 40_000_000:
                    raise ValueError("Image dimensions exceed the QR analysis limit")
                source_image.verify()
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("Input is not a supported image") from exc
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Input is not a decodable image")
        detector = cv2.QRCodeDetector()
        decoded: list[str] = []
        if hasattr(detector, "detectAndDecodeMulti"):
            success, values, _points, _straight = detector.detectAndDecodeMulti(image)
            if success:
                decoded = [v for v in values if v]
        if not decoded:
            value, _points, _straight = detector.detectAndDecode(image)
            if value:
                decoded = [value]
        urls = [value for value in decoded if value.casefold().startswith(("http://", "https://"))]
        indicators = [indicator("qr_payload", value, normalized=value) for value in decoded]
        relationships = []
        url_analyses = []
        for value in urls:
            url_output = URLAnalyzer().analyze({"submitted_value": value}, context)
            url_analyses.append(url_output)
            qr_indicator = indicator("qr_payload", value, normalized=value)
            url_indicator = next(i for i in url_output.indicators if i.type == "url")
            relationships.append(RelationshipResult(qr_indicator, url_indicator, "encodes"))
            indicators.extend(url_output.indicators)
        features = {
            "qr.decoded_count": len(decoded),
            "qr.payloads": decoded,
            "qr.url_payloads": urls,
            "qr.has_url_payload": bool(urls),
            "qr.decoder": f"opencv-{cv2.__version__}",
            "qr.url_analyses": [output.features for output in url_analyses],
        }
        observations = tuple(
            [evidence("qr_payload", "QR code encodes a URL.", attributes={"url": url}) for url in urls]
            + [item for output in url_analyses for item in output.evidence]
        )
        limitations = () if decoded else ("No QR payload could be decoded from this image.",)
        return FeatureBundle(self.name, "qr", artifact_id(artifact), features, tuple(indicators), tuple(relationships), observations, limitations)
