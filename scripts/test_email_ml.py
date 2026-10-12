import json
from app.services.analyzers.email import EmailAnalyzer

sample_email = """From: security@example.com
Reply-To: different-domain@example.net
Subject: URGENT: Verify your account immediately
MIME-Version: 1.0
Content-Type: text/plain; charset="utf-8"

Your account requires immediate verification.
Please review your account details.
"""

result = EmailAnalyzer().analyze({
    "id": "email-ml-test-001",
    "content": sample_email,
})

data = result.to_dict()

print(json.dumps({
    "analyzer": data["analyzer"],
    "features": {
        key: value
        for key, value in data["features"].items()
        if key.startswith("email.ml_")
        or key == "email.reply_to_domain_differs_from_from"
    },
    "evidence": data["evidence"],
    "limitations": data["limitations"],
}, indent=2))
