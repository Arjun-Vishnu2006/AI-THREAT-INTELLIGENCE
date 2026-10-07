"""Versioned, score-free risk triage policy over registered model/evidence outputs."""

from typing import Any

POLICY_VERSION = "evidence-risk-policy-v1"


def assess_evidence_risk(exposure_result: dict[str, Any]) -> dict[str, Any]:
    prediction = exposure_result.get("prediction") or {}
    category = str(prediction.get("exposure_category", "UNKNOWN")).upper()
    knowledge = str(exposure_result.get("knowledge_status", "NOT_ASSESSED")).upper()
    anomaly = str(exposure_result.get("anomaly_status", "NOT_ASSESSED")).upper()
    novelty = str(exposure_result.get("novelty_status", "NOT_ASSESSED")).upper()
    matches = exposure_result.get("known_matches") or []
    malicious_matches = [
        match for match in matches
        if "malicious" in str(match.get("classification", "")).casefold()
    ]
    factors: list[dict[str, Any]] = []
    if malicious_matches:
        factors.append({
            "kind": "known_malicious_indicator_match",
            "count": len(malicious_matches),
            "indicator_types": sorted({str(item.get("indicator_type")) for item in malicious_matches}),
            "record_ids": sorted({str(item.get("record_id")) for item in malicious_matches}),
            "source": "exact_match_threat_intelligence",
        })
    if category in {"LOW", "GUARDED", "HIGH"}:
        factors.append({
            "kind": "exposure_category_prediction",
            "category": category,
            "probability": prediction.get("probabilities", {}).get(category),
            "model": exposure_result.get("model"),
        })
    if anomaly in {"ANOMALOUS", "NOT_ANOMALOUS"}:
        factors.append({"kind": "individual_anomaly_status", "value": anomaly})
    if novelty in {"NOVEL", "NOT_NOVEL"}:
        factors.append({"kind": "multi_level_novelty_status", "value": novelty})
    if knowledge in {"KNOWN", "UNKNOWN"}:
        factors.append({"kind": "knowledge_status", "value": knowledge})

    if malicious_matches or (category == "HIGH" and (anomaly == "ANOMALOUS" or novelty == "NOVEL")):
        level = "HIGH"
        rationale = "A known malicious exact match or a high exposure prediction supported by anomaly/novelty evidence."
    elif category in {"GUARDED", "HIGH"} or anomaly == "ANOMALOUS" or novelty == "NOVEL" or knowledge == "KNOWN":
        level = "GUARDED"
        rationale = "At least one model or evidence signal warrants analyst review under the configured policy."
    else:
        level = "LOW"
        rationale = "No configured escalation condition was met; this does not establish safety."
    return {
        "risk_level": level,
        "policy_version": POLICY_VERSION,
        "alert_recommended": level in {"GUARDED", "HIGH"},
        "factors": factors,
        "rationale": rationale,
        "numeric_risk_score": None,
        "interpretation": "Policy triage based on registered model outputs and evidence; not a probability of harm.",
    }
