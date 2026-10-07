# Phase 10 research protocol (pre-evaluation)

**Status:** Protocol defined before the Phase 10 evaluation run. This protocol is for synthetic pipeline validation; it is not a clinical, operational, or real-world security efficacy study.

**Amendment before final run:** A preliminary run emitted a LogisticRegression iteration-limit warning from the Phase 4 behavior baseline reference. The evaluation-only reference maximum iteration limit was set to 2000; this does not change Phase 4's model-training default or any registered immutable model. The same seed and split procedure are retained. The final report is regenerated after this change and is the run to interpret. This is recorded to keep the experiment trail explicit.

## Research questions and hypotheses

- **RQ1 — Incremental novelty evidence:** How do URL/email/file/QR individual anomaly, relationship, graph-structure, attack-chain, and temporal signals perform on held-out known-pattern and deliberately unseen synthetic cases?
  - **H1:** At the fixed, predeclared `alpha=0.05` decision threshold, at least one relationship/graph/path/temporal layer or their full fusion will increase unseen-pattern recall over the individual-anomaly baseline. Report false-positive rate alongside recall; do not select a threshold on the held-out set.
- **RQ2 — Fusion and error tradeoff:** Does full multi-level fusion change precision, recall, F1, false-positive rate (FPR), false-negative rate (FNR), ROC-AUC, or PR-AUC relative to each single layer?
  - **H2:** Full fusion will increase PR-AUC over the individual-only baseline without an FPR increase greater than five percentage points. If the observed results do not meet both criteria, H2 is not supported.
- **RQ3 — Exposure category prediction and calibration:** How does the calibrated exposure predictor perform on campaign-disjoint synthetic LOW/GUARDED/HIGH scenarios?
  - **H3:** The held-out multiclass Brier score and expected calibration error are finite and reportable for all categories. This is a reporting/validity hypothesis, not a claim that calibration transfers to real threats.
- **RQ4 — Runtime cost:** What is the observed cost of novelty scoring, exposure prediction, and the combined train/calibrate/evaluate workflow on this local synthetic corpus?
  - **H4:** Per-example inference timing can be measured reproducibly on the evaluation host and will be reported with sample count, median, p95, and mean; no production capacity target is asserted.

## Frozen comparison design

Use seed `2026`, novelty `alpha=0.05`, the synthetic novelty corpus version `synthetic-multilevel-novelty-v1`, and its campaign-disjoint fit/calibration/held-out partitions. The binary positive class is the corpus label `unseen`; the negative class is a held-out instance of a reference motif. Compare, in order:

1. Individual anomaly baseline.
2. Relationship anomaly only.
3. Graph-structure novelty only.
4. Attack-chain/path novelty only.
5. Temporal novelty only, reporting how many held-out examples have a temporal reference.
6. Full multi-level fusion, using the pipeline's Bonferroni-corrected minimum empirical conformal p-value.

Also report leave-one-layer-out ablations for full fusion. Single-layer stages with unavailable evidence are `NOT_ASSESSED` (negative decision for fixed-threshold metrics); curves use a neutral score for unavailable observations and include coverage. The primary point estimates are precision, recall, F1, FPR and FNR at fixed alpha; ROC-AUC and PR-AUC use continuous nonconformity (or inverse combined p-value for fusion). Include confusion counts, score coverage, and up to 25 false-positive/false-negative synthetic sample IDs with their layer evidence for error analysis. No threshold is tuned on held-out data.

Exposure prediction is evaluated separately with its existing campaign-grouped fit/calibration/test partition. Report accuracy, macro-F1, per-category precision/recall/F1/FPR/FNR, multiclass and per-category Brier scores, log loss, reliability bins, ECE, ROC-AUC/PR-AUC where defined, confusion counts, and held-out inference latency.

## Reproducibility and interpretation constraints

- Preserve split campaign manifests and corpus fingerprints in every report; fail if groups overlap.
- Use only the generated synthetic corpora, frozen seed, alpha, and feature pipeline for this run. Never tune on held-out examples.
- Record Python and relevant library versions, UTC generation time, sample counts, and timing method. Inference timings are sequential single-process per-example measurements without a production load profile.
- Save a machine-readable report under `reports/`; the report is an experiment artifact, not a registered production model.
- Synthetic motifs are constructed scenarios, not representative prevalence estimates. No result establishes real-world detection effectiveness, calibrated real-world risk, or production performance.
- An anomaly/novelty false positive means a synthetic reference sample crossed the frozen decision threshold; false negatives mean a deliberately unseen sample did not. Review their IDs and layer evidence before interpreting metric changes.
