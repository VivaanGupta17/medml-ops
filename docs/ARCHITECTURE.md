# MedML-Ops Architecture

> **Pipeline Architecture Documentation**
> FDA-Compliant Machine Learning Operations for Medical Device AI

---

## Overview

MedML-Ops is a modular, end-to-end MLOps pipeline designed around the FDA's GMLP principles and the Software as a Medical Device (SaMD) lifecycle. Every component maps to a regulatory requirement; nothing is added for its own sake.

The pipeline has five distinct phases that mirror the SaMD lifecycle:
1. **Data → Validate** (GMLP P3, P4)
2. **Validate → Train** (GMLP P5)
3. **Train → Evaluate** (GMLP P3, P7, P8)
4. **Evaluate → Deploy** (GMLP P6, P9)
5. **Deploy → Monitor → Update** (GMLP P10 + PCCP)

---

## High-Level Architecture

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                              MedML-Ops Pipeline                                 │
│                                                                                  │
│  ┌─────────────┐                                                                │
│  │  Raw Data   │ ──── SHA-256 Hash ─────────────────────────────────────┐      │
│  │  (CSV/DICOM)│                                                         │      │
│  └──────┬──────┘                                                         │      │
│         │                                                                 │      │
│         ▼                                                                 │      │
│  ┌──────────────────┐   schema errors    ┌───────────────┐               │      │
│  │  Schema          │ ──────────────────▶│  Validation   │               │      │
│  │  Validator       │                    │  Report       │               │      │
│  │  (GE-style)      │   drift detected   │  (JSON/HTML)  │               │      │
│  │  + DICOM Audit   │ ──────────────────▶└───────────────┘               │      │
│  └──────┬───────────┘                                                     │      │
│         │                                                                 │      │
│         ▼                                                                 ▼      │
│  ┌──────────────────┐                                          ┌─────────────┐  │
│  │  Automated       │◀──── pipeline_config.yaml               │  MLflow     │  │
│  │  Training        │                                          │  Tracking   │  │
│  │  (Optuna HPO     │──── hyperparams, metrics, artifacts ───▶│  Server     │  │
│  │   Stratified CV) │                                          │  (GMLP P5) │  │
│  └──────┬───────────┘                                          └─────┬───────┘  │
│         │                                                            │          │
│         ▼                                                            │          │
│  ┌──────────────────┐                                                │          │
│  │  Model           │──── AUROC, CI, calibration, 510(k) ──────────▶│          │
│  │  Evaluator       │    predicate comparison, op points             │          │
│  │  (Clinical       │                                                │          │
│  │   Grade)         │                                                │          │
│  └──────┬───────────┘                                                │          │
│         │                                                            │          │
│         ▼                                                            │          │
│  ┌──────────────────┐   ┌──────────────────┐                        │          │
│  │  Demographic     │   │  Regression      │                        │          │
│  │  Bias Detector   │   │  Tester          │                        │          │
│  │  (Eq.Odds, Dem.  │   │  (Golden test    │                        │          │
│  │   Parity, Cal.)  │   │   set + PCCP)    │                        │          │
│  └──────┬───────────┘   └──────────────────┘                        │          │
│         │                                                            │          │
│         ▼                                                            │          │
│  ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐ │          │
│  │  GMLP            │   │  PCCP            │   │  Model Card      │ │          │
│  │  Checklist       │   │  Manager         │   │  Generator       │ │          │
│  │  (10 Principles) │   │  (Change Ctrl)   │   │  (FDA-ready)     │ │          │
│  └──────┬───────────┘   └────────┬─────────┘   └──────────────────┘ │          │
│         │                        │                                    │          │
│         ▼                        ▼                                    │          │
│  ┌────────────────────────────────────────────────────────────────┐   │          │
│  │                   MLflow Model Registry                        │◀──┘          │
│  │  model_name:version + stage (Staging → Production)            │             │
│  └───────────────────────────────────────┬────────────────────────┘             │
│                                          │                                       │
│                                          ▼                                       │
│  ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐             │
│  │  FastAPI         │   │  Drift Monitor   │   │  Prediction      │             │
│  │  Model Server    │──▶│  (PSI, KS,       │   │  Audit Log       │             │
│  │  + Audit Logger  │   │   Evidently AI)  │   │  (JSONL)         │             │
│  └──────────────────┘   └──────────────────┘   └──────────────────┘             │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## Module Architecture

### `src/data_validation/`

#### `schema_validator.py`
**Responsibility:** Enforce data contracts, detect drift.

| Class/Function | Purpose |
|----------------|---------|
| `MedicalSchemaSpec` | Declarative schema: required columns, ranges, categoricals, missing tolerance |
| `MedicalDataValidator` | Runs full validation suite |
| `ValidationReport` | Aggregated pass/fail with per-check details |
| `_check_data_drift()` | KS test + chi-square + PSI for train→production shift |
| `validate_dicom_metadata()` | Checks required DICOM tags (PatientID, Modality, Manufacturer) |

**Design decision:** PSI is computed at prediction time (not just training), so the same code serves both training data validation and production monitoring.

#### `bias_detector.py`
**Responsibility:** Demographic fairness analysis.

| Class/Function | Purpose |
|----------------|---------|
| `DemographicBiasDetector` | Orchestrates subgroup + fairness analysis |
| `SubgroupMetrics` | Per-group metrics (AUROC, sensitivity, specificity, PPV, F1) |
| `FairnessMetrics` | Pairwise fairness (equalized odds, demographic parity, calibration gap) |
| `BiasReport` | Full report with HTML export |
| `_intersectional_analysis()` | Computes metrics for intersected groups (e.g., elderly+female) |

**Flag thresholds:**
- **LOW:** < 5% gap — acceptable
- **MODERATE:** 5–10% — document and monitor  
- **HIGH:** > 10% — required remediation before submission

---

### `src/training/`

#### `experiment_tracker.py`
**Responsibility:** GMLP-compliant MLflow wrapper.

Key methods:
- `log_dataset_metadata()` — computes SHA-256, logs split info
- `verify_train_test_separation()` — programmatic leak check
- `set_seeds()` — reproducibility enforcement across all frameworks
- `_log_environment()` — captures Python version, git hash, key library versions
- `log_gmlp_documentation()` — saves compliance JSON as run artifact

**GMLP documentation schema:**
```json
{
  "gmlp_version": "2021-10",
  "intended_use": "...",
  "dataset_provenance": {
    "sha256": "...",
    "version": "v2.1",
    "source": "RSNA 2024",
    "n_samples": 10000
  },
  "reproducibility": {
    "random_seed": 42,
    "git_commit": "abc123...",
    "python_version": "3.10.14"
  },
  "team_sign_off": {
    "ml_engineer": "...",
    "clinical_expert": "...",
    "regulatory_reviewer": "..."
  }
}
```

#### `automated_training.py`
**Responsibility:** Reproducible end-to-end training with HPO.

```
TrainingConfig (YAML-driven)
        ↓
AutomatedTrainingPipeline.run()
        ├── _prepare_data()          # Feature extraction
        ├── _train_test_split()      # Group-aware split (patient-level)
        ├── _run_hpo()               # Optuna TPE, 50 trials
        ├── _run_cross_validation()  # Stratified/Group/StratifiedGroup KFold
        ├── final_model.fit()        # Train on full dev set
        └── _evaluate_model()        # Test set metrics
```

**Leakage prevention:** When `group_column` (patient ID) is provided, `GroupShuffleSplit` and `StratifiedGroupKFold` ensure all samples from a patient go to the same partition.

---

### `src/evaluation/`

#### `model_evaluator.py`
**Responsibility:** Clinical-grade evaluation beyond standard ML metrics.

Operating point selection methods:
- **Youden's J** — maximize sensitivity + specificity - 1 (balanced detection)
- **Iso-F1** — maximize F1 score
- **Sensitivity targets** — specificity at fixed 85%/90%/95% sensitivity (screening context)
- **Fixed threshold** — standard 0.5 cutoff

Calibration analysis:
- **Brier score** — overall probabilistic accuracy
- **ECE** (Expected Calibration Error) — mean gap between confidence and accuracy across bins
- **MCE** (Maximum Calibration Error) — worst-case calibration bin

Statistical tests:
- **Mann-Whitney U** — model vs. chance (AUROC > 0.5)
- **Superiority test** — model vs. predicate device AUROC
- **Non-inferiority test** — 510(k) substantial equivalence margin (Δ = 0.02)

#### `regression_testing.py`
**Responsibility:** PCCP-aligned change validation.

Test types:
1. **Overall regression** — AUROC, F1, sensitivity, specificity vs. baseline
2. **Performance floor** — absolute minimum requirements (never go below)
3. **Critical case test** — zero new false negatives on flagged critical cases
4. **Slice regression** — per demographic subgroup
5. **McNemar's test** — statistical significance of error pattern changes

---

### `src/deployment/`

#### `model_server.py`
**Responsibility:** Production inference with full audit trail.

```
POST /predict
    ├── Input validation (Pydantic schema)
    ├── feature → numpy array
    ├── model.predict_proba()
    ├── PredictionAuditLogger.log()     # SHA-256 input hash, no raw PHI
    └── PredictionResponse
        ├── probability
        ├── predicted_class
        ├── high_confidence flag
        └── compliance_note

GET /health → HealthResponse (model_loaded, uptime, predictions_served)
GET /health/ready → Kubernetes readiness probe
GET /health/live → Kubernetes liveness probe
```

**PHI protection:** The audit logger hashes input features and patient IDs with SHA-256 before writing to log. Raw feature values are never persisted.

---

### `src/monitoring/`

#### `drift_monitor.py`
**Responsibility:** Production data and prediction drift detection.

| Test | Features | Alert Threshold |
|------|----------|----------------|
| Kolmogorov-Smirnov | Numeric features | p < 0.05 |
| Chi-square | Categorical features | p < 0.05 |
| PSI | All features | > 0.10 (warn), > 0.25 (alert) |
| Prediction distribution | y_prob | KS test + PSI |
| Label prevalence | y_true (if available) | Chi-square |

Alert levels:
- **GREEN:** < 20% features drifted, no prediction drift
- **YELLOW:** > 20% features drifted
- **RED:** > 3 high-PSI features OR prediction drift detected

#### `model_card_generator.py`
**Responsibility:** Automated FDA-submission-ready model card generation.

Sections:
1. Model details (name, version, framework, git commit)
2. Intended use + out-of-scope applications
3. Training data description (provenance, demographics, preprocessing)
4. Quantitative analysis (overall + subgroup performance, calibration, CIs)
5. Primary operating point
6. Fairness metrics
7. Known limitations and ethical considerations
8. Regulatory and provenance information

---

### `src/compliance/`

#### `gmlp_checklist.py`
**Responsibility:** Automated GMLP audit with evidence collection.

Audit logic for each principle:
- **Automated checks:** Reads MLflow run params (git commit, seed, dataset hash, etc.)
- **Manual attestations:** `checker.attest_principle(1, "clinical_review", "Dr. Smith approved")` 
- **Pipeline flags:** Binary inputs (`has_bias_report`, `has_model_card`, etc.)

Output: 10-principle compliance matrix with `score`, `evidence`, `gaps`, and `remediation`

#### `pccp_manager.py`
**Responsibility:** PCCP specification and change validation.

```
PCCPSpecification
    ├── allowed_changes: List[AllowedChange]
    │   ├── change_type
    │   ├── conditions (manual verification)
    │   ├── required_testing (automated)
    │   └── max_allowed_performance_degradation
    └── performance_floor: Dict[metric, minimum_value]

PCCPChangeRequest → PCCPManager.validate_change() → PCCPValidationReport
    ├── within_pccp: bool
    ├── impact_level: MINOR/MODERATE/MAJOR/CRITICAL
    ├── status: APPROVED/REJECTED/PENDING_TESTING/UNDER_REVIEW
    └── regulatory_pathway: text description
```

---

## Data Flow

```
Raw Data
    │
    ▼ SHA-256 hash computed
schema_validator.py
    │
    ▼ Train/val/test split (patient-level)
experiment_tracker.py  ──── MLflow run starts
    │
    ▼ Seeds set, environment logged
automated_training.py
    │ HPO (Optuna)
    │ Cross-validation (StratifiedGroupKFold)
    │ Final training
    ▼
model_evaluator.py  ──────── Evaluation report (AUROC, calibration, CIs)
    │
    ▼
bias_detector.py  ─────────── Bias report (subgroup + fairness)
    │
    ▼
regression_testing.py  ─────── Regression report (vs. golden test set)
    │
    ▼
gmlp_checklist.py  ──────────── GMLP audit (10-principle compliance)
    │
    ▼
pccp_manager.py  ─────────────── Change validation (if update)
    │
    ▼
model_card_generator.py  ──────── Model card (HTML + JSON)
    │
    ▼
MLflow Model Registry  ─────────── model_name:version (Staging)
    │
    ▼ Approved → Production
model_server.py (FastAPI)
    │
    ├── Every prediction → PredictionAuditLogger
    │
    └── Scheduled batch → drift_monitor.py → Alert
```

---

## Technology Choices

| Technology | Why Chosen |
|-----------|-----------|
| **MLflow** | Industry-standard experiment tracking; model registry; supports GMLP data provenance; self-hostable (no vendor lock-in) |
| **FastAPI** | Automatic OpenAPI docs; Pydantic validation; async support; production-proven |
| **Optuna** | State-of-the-art TPE sampler; supports pruning; reproducible via seed |
| **Evidently AI** | Purpose-built ML monitoring; rich drift dashboards; GMLP-aligned |
| **Great Expectations** | Declarative data contracts; supports schema-as-code |
| **SciPy** | Authoritative statistical tests (KS, chi-square, Mann-Whitney, McNemar) |
| **Docker** | Reproducible environments; pinned base images; GMLP Principle 2 |

---

## Deployment Topologies

### Local Development
```bash
make train  # Local mlruns/ storage
make serve  # http://localhost:8000
make monitor  # http://localhost:8001
```

### Docker Compose (Small Team)
```bash
docker-compose up  # Postgres + MinIO + MLflow + FastAPI + Monitor
```

### Kubernetes (Production)
- Training: Kubernetes Job with `Dockerfile.train`
- Serving: Deployment with `Dockerfile.serve`, HPA on CPU/latency
- MLflow: StatefulSet with Postgres backend
- Monitoring: CronJob calling drift monitor API

---

*See [FDA_COMPLIANCE.md](FDA_COMPLIANCE.md) for regulatory context for each architectural decision.*
