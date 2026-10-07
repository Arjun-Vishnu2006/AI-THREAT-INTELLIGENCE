"""Run the Phase 10 frozen synthetic novelty, exposure, and baseline evaluation."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy
import sklearn

from app.exposure.training import train_and_evaluate_exposure
from app.ml.baseline import train_baseline
from app.novelty.training import train_and_evaluate_novelty


def evaluate(seed: int = 2026, alpha: float = 0.05) -> dict[str, Any]:
    started = perf_counter()
    novelty_started = perf_counter()
    _, novelty = train_and_evaluate_novelty(seed=seed, alpha=alpha)
    novelty_seconds = perf_counter() - novelty_started
    exposure_started = perf_counter()
    _, exposure = train_and_evaluate_exposure(seed=seed)
    exposure_seconds = perf_counter() - exposure_started
    baseline_started = perf_counter()
    baseline = train_baseline(seed=seed, max_iter=2000)
    baseline_seconds = perf_counter() - baseline_started
    stages = novelty["stage_comparison"]
    baseline_stage = stages["baseline_individual_anomaly"]
    full_stage = stages["full_multi_level_fusion"]
    candidate_stages = [stages[name] for name in (
        "relationship_anomaly", "graph_structure_novelty", "attack_chain_novelty",
        "temporal_novelty", "full_multi_level_fusion",
    )]
    exposure_metrics = exposure["held_out_metrics"]["calibrated"]
    return {
        "protocol": "docs/phase10-research-protocol.md",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "synthetic_only": True,
        "effectiveness_claimed": False,
        "configuration": {"seed": seed, "novelty_alpha": alpha},
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": numpy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "runtime_seconds": {
            "novelty_train_calibrate_evaluate": novelty_seconds,
            "exposure_train_calibrate_evaluate": exposure_seconds,
            "phase4_behavior_baseline_train_evaluate": baseline_seconds,
            "total": perf_counter() - started,
        },
        "hypothesis_assessment": {
            "H1_recall_gain_over_individual_baseline": {
                "supported_on_synthetic_split": any(item["recall"] > baseline_stage["recall"] for item in candidate_stages),
                "baseline_recall": baseline_stage["recall"],
                "best_candidate_recall": max(item["recall"] for item in candidate_stages),
                "interpretation": "Predeclared recall-increase criterion only; no real-world inference.",
            },
            "H2_fusion_pr_auc_and_fpr_tradeoff": {
                "supported_on_synthetic_split": (
                    full_stage["pr_auc"] > baseline_stage["pr_auc"]
                    and full_stage["false_positive_rate"] <= baseline_stage["false_positive_rate"] + 0.05
                ),
                "baseline_pr_auc": baseline_stage["pr_auc"],
                "fusion_pr_auc": full_stage["pr_auc"],
                "baseline_false_positive_rate": baseline_stage["false_positive_rate"],
                "fusion_false_positive_rate": full_stage["false_positive_rate"],
                "interpretation": "Both predeclared conditions must hold; no real-world inference.",
            },
            "H3_exposure_metrics_reportable": {
                "supported_on_synthetic_split": all(
                    numpy.isfinite(exposure_metrics[name])
                    for name in ("multiclass_brier_score", "top_label_expected_calibration_error")
                ),
                "interpretation": "Finite held-out metrics do not establish transfer calibration.",
            },
            "H4_runtime_measured": {
                "supported_on_this_run": (
                    novelty["inference_latency"]["sample_count"] > 0
                    and exposure["held_out_metrics"]["single_example_inference_latency"]["sample_count"] > 0
                ),
                "interpretation": "Local sequential measurement only; not a production capacity claim.",
            },
        },
        "novelty": novelty,
        "exposure": exposure,
        "phase4_behavior_baseline": baseline.metadata,
    }


def _semantic_signature(report: dict[str, Any]) -> str:
    novelty = {key: value for key, value in report["novelty"].items() if key != "inference_latency"}
    exposure = dict(report["exposure"])
    exposure["held_out_metrics"] = {
        key: value for key, value in exposure["held_out_metrics"].items()
        if key != "single_example_inference_latency"
    }
    comparable = {
        "configuration": report["configuration"],
        "novelty": novelty,
        "exposure": exposure,
        "phase4_behavior_baseline": report["phase4_behavior_baseline"],
        "hypothesis_assessment": report["hypothesis_assessment"],
    }
    return json.dumps(comparable, sort_keys=True, separators=(",", ":"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--output", type=Path, default=Path("reports/phase10-seed-2026.json"))
    parser.add_argument("--verify-reproducibility", action="store_true",
                        help="Repeat the run and compare deterministic metrics/fingerprints (excludes time measurements)")
    args = parser.parse_args()
    report = evaluate(seed=args.seed, alpha=args.alpha)
    if args.verify_reproducibility:
        repeated = evaluate(seed=args.seed, alpha=args.alpha)
        matches = _semantic_signature(report) == _semantic_signature(repeated)
        report["reproducibility_check"] = {
            "repeated_run_count": 2,
            "deterministic_metrics_and_fingerprints_match": matches,
            "timing_fields_excluded": True,
        }
        if not matches:
            raise RuntimeError("Repeated evaluation changed metrics or fingerprints; report not written")
    else:
        report["reproducibility_check"] = {"repeated_run_count": 1, "cross_run_check": "not requested"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "report": str(args.output),
        "synthetic_only": report["synthetic_only"],
        "effectiveness_claimed": report["effectiveness_claimed"],
        "hypothesis_assessment": report["hypothesis_assessment"],
        "reproducibility_check": report["reproducibility_check"],
        "novelty_stages": list(report["novelty"]["stage_comparison"]),
        "exposure_metrics": report["exposure"]["held_out_metrics"]["calibrated"],
        "runtime_seconds": report["runtime_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
