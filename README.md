# Threat Analysis API Foundation

This repository contains the service foundation, deterministic individual analyzers, cross-input graph processing, novelty layers, and a calibrated synthetic exposure-category predictor. The novelty autoencoder remains experimental; synthetic research models do not establish real-world maliciousness or impact.

## Purpose and processing flow

This is a research prototype for analyzing submitted URLs, email messages, QR images, and files (including APKs where static parsing is available). Its processing path is:

`input → analyzer → FeatureBundle → correlation/graph → NoveltyEngine → exposure category → evidence-based risk policy → API/dashboard`

The novelty engine has six layers: individual-feature anomaly, relationship anomaly, graph structure, attack-chain/path novelty, temporal novelty, and an optional Android autoencoder. The first five can assess applicable non-Android input. The autoencoder runs only when a valid, strictly checked 470-column Android feature vector is present. The project does not execute submitted files or make network requests for submitted URLs.

The application is a local research prototype, not a production security service. Its responses are evidence summaries and synthetic-scenario model outputs; they do not establish maliciousness, real-world risk, or detection effectiveness.

## Run locally

Requires Python 3.11 or newer.

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
py -m pip install -e .
Copy-Item .env.example .env
uvicorn app.main:app --reload
```

The equivalent explicit command is `python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload`.

The investigation dashboard is available at `/`; the API docs are available at `/docs`. The default SQLite database is `threat_analysis.db` in the working directory. Development mode creates missing tables on startup. For a persistent deployment, set `ENVIRONMENT=production`, provide a PostgreSQL `DATABASE_URL`, and manage schema evolution through database migrations.

## API surface

- `GET /api/v1/health`
- `POST /api/v1/analyze/url`
- `POST /api/v1/analyze/email`
- `POST /api/v1/analyze/qr`
- `POST /api/v1/analyze/file`
- `POST /api/v1/correlate`
- `GET /api/v1/correlate/synthetic-cases`
- `POST /api/v1/novelty/analyze`
- `POST /api/v1/exposure/predict`
- `POST /api/v1/risk/assess`
- `GET /api/v1/alerts`
- `PATCH /api/v1/alerts/{alert_id}`
- `GET /api/v1/alerts/{alert_id}/investigation`
- `GET /api/v1/intel/dataset`
- `POST /api/v1/intel/check`
- `GET /api/v1/models`
- `GET /api/v1/models/{name}/{version}`
- `POST /api/v1/models/{name}/{version}/predict`
- `POST /api/v1/sessions`
- `GET /api/v1/sessions`
- `GET /api/v1/sessions/{session_id}`
- `POST /api/v1/sessions/{session_id}/artifacts`
- `GET /api/v1/sessions/{session_id}/artifacts`
- `GET /api/v1/sessions/{session_id}/evidence`
- `GET /api/v1/sessions/{session_id}/history`

Artifact `storage_ref` is only a reference field; upload transport and binary storage are future work. Inputs and outputs use explicit Pydantic schemas. Analysis findings are persisted as observations with optional confidence supplied by a future producer; no default risk score is calculated.

Submitted email/file/QR content is processed in memory and is not written to artifact storage. To retain security evidence, pass a session ID as the `session_id` query parameter to an individual analyzer request, or include `session_id` in a correlation request. Normalized indicator observations, analyzer evidence and correlated relationships are stored in `security_evidence`; raw request content is excluded. Retrieve those records from `GET /api/v1/sessions/{session_id}/evidence`.

## Phase 2 analyzers

The deterministic analyzer implementations live under `app.services.analyzers` and emit the common `FeatureBundle` representation (`features`, `indicators`, `relationships`, `evidence`, and `limitations`). They do not calculate a threat score and never fetch a submitted URL. The API accepts `submitted_value` for URLs and RFC email content; QR and file requests accept standard `content_base64`, with optional `name` and `media_type`. Payloads are capped at 50 MiB. Email attachments are hashed and described but are not run. Files are inspected in memory with bounded size/archive limits and are never executed or extracted. QR decoding is optional; install it with `python -m pip install -e .[qr]` (PowerShell may require quoting: `python -m pip install -e '.[qr]'`).

Example use from Python:

```python
from datetime import datetime, timezone
from app.services.analyzers import URLAnalyzer
from app.services.contracts import AnalysisContext

context = AnalysisContext(session_id="example", observed_at=datetime.now(timezone.utc))
result = URLAnalyzer().analyze({"submitted_value": "https://example.org/path?a=1"}, context)
print(result.to_dict())
```

## Phase 3 correlation and graph

`app.services.correlation.CrossInputCorrelator` canonicalizes indicators, merges identity-equal entities, and builds typed edges such as `CONTAINS`, `ATTACHED_TO`, `DECODED_TO`, `LINKS_TO`, `BELONGS_TO`, `RESOLVES_TO`, `HAS_HASH`, and `RELATED_TO`. Matching SHA-256 evidence can connect a file inspection to QR analysis of the same bytes. The API accepts a list of the same URL/email/QR/file inputs at `POST /api/v1/correlate`; it returns entities, edges, graph structure features, and bounded directed paths between artifacts. `GET /api/v1/correlate/synthetic-cases` returns safe `.test` fixtures, including multi-hop email → file → QR → URL → domain/IP paths, and their computed graph output. Install the new required graph dependency after updating the checkout with `python -m pip install -e .`.

Relationship paths describe observed/shared evidence only; they are not verdicts or validated attacks. The system does not perform DNS lookups. `RESOLVES_TO` is emitted only when the caller supplies resolution evidence through the feature bundle. Synthetic cases use reserved `.test` domains or RFC 5737 documentation IP space and never contact external infrastructure.

## Phase 4 known-threat baseline and ML baseline

The local `data/ioc_baseline.json` dataset contains synthetic-only exact-match indicators. `GET /api/v1/intel/dataset` reports the active dataset; `POST /api/v1/intel/check` accepts typed indicators and returns separate `knowledge_status` (`KNOWN`/`UNKNOWN`) and `behavior_status` (`NOT_ASSESSED`) fields. `KNOWN` means an exact match to the configured provider. The included data is for development only and is not real threat intelligence. Replace or extend it only with authorized local IOC sources; lookups do not make network requests.

## Dataset provenance and split use

The Android autoencoder experiments refer to `data/train.csv`, `data/validation.csv`, and `data/test.csv`, with a 470-column feature schema. The repository does not document the dataset publisher, original collection, license, or redistribution terms. The class mapping recorded in the experiment metadata is `1=Adware`, `2=Banking`, `3=SMS`, `4=Riskware`, `5=Benign`. In 0.2.0, Adware, Banking, and Benign are training-known classes; SMS and Riskware are held out as novel classes. In 0.3.0, only the designated Benign training subset fits the model, validation selects the threshold, and test is reserved for final metrics. Validation and test are not used to fit model parameters.

The provenance of those three CSVs is unknown, so they are excluded by `.gitignore` and should not be publicly redistributed until their source and redistribution rights are verified. Do not characterize them as synthetic or real-world data without confirming the source. By contrast, the Phase 5/6 multi-layer novelty and exposure evaluations and the included IOC baseline are explicitly synthetic. Their results test pipeline behavior only.

The local database currently registers `novelty-fusion/0.1.0` and `exposure-predictor/0.1.0`; both are synthetic research artifacts. Their model files are retained as application assets, but a fresh database does not inherit those registration rows. The 0.1.0 novelty artifact is historical: its saved autoencoder calibration values repeat a threshold and are not valid AE calibration scores. The other novelty layers remain available, but the artifact must not be presented as validating the autoencoder. Android dataset experiments 0.2.0 and 0.3.0 are unregistered and experimental; the repository excludes their outputs from public commits until data provenance is verified. Their metrics do not establish runtime APK compatibility or effectiveness. Keep the 0.2.0/0.3.0 artifacts out of public commits until dataset rights are known; do not deploy either AE experiment. The canonical schema/runtime assets under `app/novelty/models/` are source dependencies and should remain included.

The baseline classifier is a reproducible synthetic behavior-classification experiment, not a maliciousness or risk model. It uses allowlisted analyzer features, a scikit-learn pipeline, and label-stratified splits grouped by synthetic campaign so a campaign cannot occur in more than one of train, validation, or test. Metrics are recorded separately for validation and held-out test data. Train an immutable version and save its artifact plus metadata to the configured registry with:

```powershell
python -m pip install -e .
python -m app.ml.train --version 0.1.0 --seed 2026
```

The command records model metadata and provenance in the database and writes a local model artifact under `models/`. Query registered versions through `/api/v1/models`. Prediction returns a behavior class with knowledge, anomaly, and novelty statuses marked `NOT_ASSESSED`; it does not produce threat verdicts or risk scores. Never load model artifacts from untrusted sources.

## Phase 5 deep novelty research baseline

The novelty pipeline keeps knowledge matches (`KNOWN`/`UNKNOWN`), individual anomaly (`ANOMALOUS`/`NOT_ANOMALOUS`), and multi-level novelty (`NOVEL`/`NOT_NOVEL`) as separate fields. Its layers assess allowlisted per-artifact features, normalized relationship frequencies, graph signatures, observed artifact-to-artifact paths, and (when a session has prior persisted observations) first-seen entities and relationships. The Android autoencoder is an additional layer only when a valid named 470-feature Android vector is present. The fusion uses held-out empirical conformal tail probabilities with the finite-sample +1 correction and applies a Bonferroni correction across layers available for that request. A missing temporal reference is reported and the temporal layer is omitted for that request. This endpoint reports `knowledge_status: NOT_ASSESSED`; use `/api/v1/intel/check` for the independent exact-match IOC assessment.

The bundled evaluation is deliberately synthetic and deterministic. It uses disjoint campaign IDs for fit, calibration, and held-out partitions, checks for overlap, evaluates held-out reference and deliberately unseen multi-input cases, reports per-layer metrics, and runs an ablation with each layer removed. These results demonstrate reproducibility and exercise the code; they are not evidence of real-world detection performance. Conformal calibration also relies on exchangeability between calibration examples and future inputs, which synthetic data alone cannot establish. Explanations include the features, relationships, graph signatures, and paths that contributed to each layer; they do not claim malicious intent.

The currently registered `novelty-fusion/0.1.0` is a historical synthetic artifact. The current `NoveltyEngine.fit()` requires at least 20 calibration snapshots with valid 470-feature Android vectors before it can calibrate the autoencoder. The synthetic corpus contains no such vectors, so a fresh synthetic-only run of the command below fails clearly at autoencoder calibration. Do not treat the existing artifact's autoencoder calibration scores as valid: its 120 saved values are copies of one threshold. The source code now computes real calibration scores and rejects missing/malformed vectors; the old immutable artifact is intentionally unchanged.

The Android APK extractor currently has deterministic extraction coverage for 435 of the 470 schema columns. The remaining 35 are unavailable, so the strict runtime path does not construct an autoencoder vector from ordinary APK analysis. **Real APK end-to-end validation has not been performed because no real APK fixture is present in the repository.** Parser-fixture tests validate the adapter only; they are not real-APK validation.

The dataset-only autoencoder experiments are under `models/android-novelty-autoencoder/0.2.0/` and `0.3.0/`. Both are unregistered. The 0.2.0 held-out ROC-AUC was 0.461. The controlled Benign-only 0.3.0 held-out ROC-AUC was 0.567 with a 0.959 false-positive rate; its F1 of 0.919 is close to an all-novel prediction baseline because 84.5% of test labels are novel. Neither experiment is suitable for deployment or validates runtime APK compatibility. Keep the autoencoder experimental while the non-autoencoder layers are evaluated independently.

The command documents the historical training workflow; with the current strict autoencoder calibration behavior, do not register a new synthetic-only artifact:

```powershell
python -m app.novelty.train --version 0.1.0 --seed 2026
```

This writes `models/novelty-fusion/<version>/engine.joblib` and `metadata.json`, registers model provenance, and stores the evaluation/ablation record as an experiment. The version cannot be overwritten; choose a new version for another run. Restart the API after training, then submit the same artifact shapes accepted by `/api/v1/correlate` to `POST /api/v1/novelty/analyze`. Optional `session_id` enables temporal comparison against prior security evidence; using a session also persists the current evidence and assessment. Use `?model_version=0.1.0` to pin the registered version. Do not load model artifacts from untrusted sources.

## Phase 6 exposure category prediction

The exposure model uses allowlisted individual analyzer features, normalized relationship signatures, graph and path features, session temporal evidence, exact-match IOC status, and per-layer novelty outputs. The `LOW`, `GUARDED`, and `HIGH` categories describe synthetic evidence scenarios. They are not estimates of real-world impact, maliciousness, or a risk score. Known IOC, anomaly, and novelty assessments remain separate response fields.

Training uses campaign-disjoint fit, probability-calibration, and held-out test partitions. A multinomial logistic calibration map is fit only on the calibration groups. Evaluation metadata records accuracy, macro-F1, multiclass Brier score, log loss, top-label expected calibration error, reliability bins, and per-category Brier scores. The synthetic evaluation is for reproducibility and pipeline development; it is not a deployment-quality estimate. Feature explanations show model contributions and the extracted graph/novelty evidence that informed the category.

Train a new immutable model version after Phase 5 has been trained:

```powershell
python -m app.exposure.train --version 0.1.0 --seed 2026
```

This registers the model and evaluation as model/experiment metadata and saves files under `models/exposure-predictor/<version>/`. Restart the API and submit the same artifact request body used by `/api/v1/correlate` to `POST /api/v1/exposure/predict`. Optional `session_id` enables temporal features and persists the structured evidence and prediction. `model_version` and `novelty_model_version` query parameters can pin either model version. Model files must come from the local trusted registry.

## Phase 7 evidence chain, risk policy, and alerts

`POST /api/v1/risk/assess` runs the registered exposure and novelty models plus the local exact-match IOC provider for artifacts in an existing analysis session. It stores analyzer evidence, IOC results, normalized cross-input relationships, graph and path evidence, novelty evidence, exposure output, a risk-policy finding, and directed `EvidenceLink` provenance edges. The finding points to the evidence records and related artifact IDs. Submitted artifact bytes are not persisted.

The versioned `evidence-risk-policy-v1` combines actual model statuses and IOC classifications into `LOW`, `GUARDED`, or `HIGH` triage. A malicious exact-match IOC, or a `HIGH` exposure category accompanied by an anomalous or novel assessment, yields `HIGH`; other escalation signals yield `GUARDED`. `GUARDED` and `HIGH` create open alerts; `LOW` creates a finding without an alert. This policy emits no numeric risk score and does not interpret an exposure probability as probability of harm. The bundled IOC dataset is synthetic only.

Use `GET /api/v1/alerts` to browse alerts, `PATCH /api/v1/alerts/{alert_id}` to acknowledge, resolve, reopen, or mark false positive, and `GET /api/v1/alerts/{alert_id}/investigation` to follow the alert through finding, evidence links, evidence, related artifacts, graph and paths. Each alert status change is appended to alert event history. `GET /api/v1/sessions/{session_id}/history` returns the session evidence, findings, alerts, links, and alert events. The development startup creates the new tables; persistent deployments should create them through the project's migration process.

## Phase 8 investigation dashboard

Open `http://127.0.0.1:8000/` while the API is running. The dashboard reads health, saved sessions, alert queue, session history, graph evidence, novelty layers, paths, and alert investigations from the API. Its intake form creates a session and artifact references, then submits the selected URL, email, file, and/or QR image to `/api/v1/risk/assess`. Raw email and binary input are sent for in-memory analysis and are not saved as artifact contents; file/QR references store only metadata and a browser-computed SHA-256 when available. The existing evidence pipeline persists structured outputs.

The risk assessment entry point requires the registered Phase 5 novelty and Phase 6 exposure models. If they are not present, the API returns an actionable error and the dashboard keeps the session reference available for review. The dashboard does not add a separate score or infer verdicts beyond the saved model/evidence output. Use the session ID field to reload any existing analysis and select an alert to inspect its evidence chain and update its lifecycle status.

## Phase 9 browser and camera integrations

The optional Chrome/Edge Manifest V3 extension is in `browser_extension/`. Setup steps and its permission list are in [browser_extension/README.md](browser_extension/README.md). It supports explicit current-page checks, a link context-menu check, and optional background top-level page/download-source IOC lookups. Results come from `/api/v1/intel/check` and are exact matches against the configured local dataset; the bundled records are synthetic only. Lookups omit `session_id`, so the extension does not add visited URLs to persistent analysis history.

Page monitoring requires separately granting optional `webNavigation` permission and HTTP/HTTPS access to all sites. Download monitoring separately requests the browser's `downloads` permission and sees download source URL/metadata; it does not read file contents. Notifications are issued only for exact local matches and depend on browser/OS notification settings. The extension does not collect browsing history or send data to a remote service. Browser navigation callbacks occur after navigation begins and cannot reliably block a page before it loads. This is not an OS-wide traffic filter, antivirus, or production safe-browsing service. Only enable broad site access when its local lookup behavior is acceptable.

The dashboard QR input offers the device file/camera picker, and **Open camera** starts a browser camera stream only after a user click. It requests video only, takes a still image for the normal QR analysis path, and stops the stream immediately after capture or close. Camera APIs require a secure browser context (localhost is supported for local development) and user permission. Mobile browser extension support varies by platform; use the responsive dashboard and camera/file picker where extensions are unavailable. No OS-level interception or background service is installed. Email remains website-only; no email provider integration or mailbox permission is present.

## Phase 10 evaluation and hardening

The preregistered synthetic research questions, hypotheses, fixed comparison sequence, and interpretation rules are in [docs/phase10-research-protocol.md](docs/phase10-research-protocol.md). Run the frozen seed-2026 evaluation with:

```powershell
python -m app.research.evaluate
```

This writes `reports/phase10-seed-2026.json` without registering or overwriting model versions. For a repeated deterministic-metric/fingerprint check (timings excluded), run `python -m app.research.evaluate --verify-reproducibility`. The report compares individual, relationship, graph, attack-chain, and temporal novelty layers against full fusion; includes per-layer leave-one-out ablations, precision/recall/F1/FPR/FNR, ROC-AUC/PR-AUC, bounded error examples with layer evidence, exposure category calibration and class metrics, campaign split manifests/fingerprints, and local inference/training time measurements. It explicitly marks the preregistered hypotheses as supported or not supported on this synthetic split. Use `--seed`, `--alpha`, or `--output` to select a separate run/output; record those choices in research results. Performance numbers describe one local, sequential synthetic run and do not constitute production load results.

The Phase 10 UI hardening adds a restrictive content security policy, same-origin script/connect rules, no-referrer and frame protections, and a camera-only Permissions Policy to the dashboard/static resources. The dashboard uses system fonts and makes no third-party font requests. Existing model loading restricts artifacts to the configured local registry; submitted files remain non-executed with analyzer size limits. Keep this development API on loopback; it has no user authentication and should not be exposed to an untrusted network. No synthetic benchmark result is evidence of real-world effectiveness.

## Tests and local checks

Install the project and its test extra, then run these commands from the repository root:

```powershell
python -m pip install -e ".[test]"
python -m pytest -q
python -m compileall -q app tests
python -m pip check
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

The CSV dataset experiments cannot be reproduced from a public checkout unless the user obtains the datasets from an authorized source and places the three split files under `data/`. A fresh database also has no model-registration rows; the registered artifacts documented above are not a replacement for an authorized, reproducible registration/setup workflow. No license file is currently included. Confirm source ownership and third-party/data rights before adding a license or redistributing data; if the project authors control the source and want a permissive source-code license, MIT is a simple option. The data may require separate terms.
