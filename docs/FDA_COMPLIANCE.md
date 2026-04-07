# FDA Regulatory Compliance Documentation

> **MedML-Ops: FDA-Compliant Machine Learning Operations Pipeline for Medical AI**

This document explains how the MedML-Ops pipeline implements each of the FDA's 10 Good Machine Learning Practice (GMLP) guiding principles, the PCCP framework for post-market model updates, 510(k) submission preparation guidance, alignment with IEC 62304, and considerations under 21 CFR Part 820.

---

## Table of Contents

1. [FDA GMLP Guiding Principles — Implementation](#1-fda-gmlp-guiding-principles--implementation)
2. [PCCP — Predetermined Change Control Plan](#2-pccp--predetermined-change-control-plan)
3. [510(k) Submission Preparation](#3-510k-submission-preparation)
4. [IEC 62304 Software Lifecycle Alignment](#4-iec-62304-software-lifecycle-alignment)
5. [21 CFR Part 820 Quality System Considerations](#5-21-cfr-part-820-quality-system-considerations)
6. [Regulatory References](#6-regulatory-references)

---

## 1. FDA GMLP Guiding Principles — Implementation

The FDA, Health Canada, and UK MHRA jointly published 10 GMLP guiding principles in October 2021. The following table maps each principle to the corresponding MedML-Ops implementation.

### Principle 1 — Multi-Disciplinary Expertise Is Leveraged

> *Teams include relevant clinical, software engineering, regulatory, and cybersecurity expertise throughout the product lifecycle.*

**Implementation:**
- `GMLPDocumentation.set_team_sign_off()` records ML engineer, clinical expert, and regulatory reviewer sign-offs in every MLflow run
- `GMLPComplianceChecker._check_p1_multidisciplinary()` audits for documented team involvement
- Pipeline config supports `team_sign_off` metadata fields
- The PCCP specification requires clinical review before deployment of any moderate/major change

**Evidence artifacts:** `mlruns/<run_id>/compliance/gmlp_documentation.json` → `team_sign_off`

---

### Principle 2 — Good Software Engineering and Security Practices Are Implemented

> *Established software development practices are used, including version control, testing, and security review.*

**Implementation:**
- Git version control enforced with pre-commit hooks (`black`, `isort`, `mypy`)
- `GMLPExperimentTracker._log_environment()` captures git commit hash, Python version, and all key package versions at training time
- Requirements pinned in `requirements.txt` with version bounds
- Docker images use pinned base image tags (`python:3.10.14-slim-bookworm`) for supply chain traceability
- CI/CD pipeline (configured via `.pre-commit-config.yaml`) with automated linting and type checking

**Evidence artifacts:** `env.git_commit`, `env.python_version`, `env.pkg_*` in MLflow params

---

### Principle 3 — Clinical Study Design Generates Appropriate Evidence

> *The study design is clinically motivated and generates statistically valid evidence for the intended use.*

**Implementation:**
- `MedicalModelEvaluator` computes bootstrap confidence intervals (1000 iterations, 95% CI) for all primary metrics
- `AutomatedTrainingPipeline` supports stratified and group-aware cross-validation
- `model_evaluator.py` includes statistical significance testing (Mann-Whitney U vs. chance, superiority/non-inferiority tests vs. predicate)
- Clinical operating point selection via Youden's J, iso-F1, and per-sensitivity-target thresholds
- Statistical comparison against predicate device included in `510(k)` context

**Evidence artifacts:** `reports/evaluation_report.json` → `confidence_intervals`, `statistical_tests`

---

### Principle 4 — Data Management Supports Transparency and Generalizability

> *Data collection, labeling, and management are documented with provenance and quality information.*

**Implementation:**
- `GMLPExperimentTracker.log_dataset_metadata()` computes and logs SHA-256 hash of every dataset split
- `MedicalDataValidator` enforces schema, detects missing values, outliers, and categorical violations
- DICOM metadata validation via `validate_dicom_metadata()` checks required tags (PatientID, StudyDate, Modality, Manufacturer, PixelSpacing)
- Dataset version string logged alongside data source and collection period
- Demographic completeness check ensures bias analysis prerequisites are met

**Evidence artifacts:** `dataset.train.sha256`, `dataset.train.source`, `dataset.train.version` in MLflow params

---

### Principle 5 — Re-Training Practices Are Transparent

> *When models are re-trained, the process is documented to ensure safe and effective performance is maintained.*

**Implementation:**
- `GMLPExperimentTracker.set_seeds()` logs and enforces random seeds across NumPy, Python, PyTorch, and TensorFlow
- Full environment captured: Python version, git commit, all key library versions
- `ModelRegressionTester` provides regression testing against a locked golden test set
- `PCCPManager.validate_change()` documents every model change with acceptance criteria
- `PCCP_change_log.json` provides full audit trail of all model modifications

**Evidence artifacts:** `random_seed`, `env.*` in MLflow params; `compliance/pccp_change_log.json`

---

### Principle 6 — Model Design Is Tailored to Available Data and Addresses Bias

> *Model design and training explicitly consider data limitations and potential biases.*

**Implementation:**
- `DemographicBiasDetector` stratifies performance across age group, biological sex, and self-reported ethnicity
- Fairness metrics: equalized odds (TPR gap, FPR gap), demographic parity gap, calibration gap
- Intersectional analysis (e.g., elderly Black women as a subgroup)
- Automated HIGH/MODERATE/LOW bias flags with FDA-aligned 10% threshold
- `bias_report.html` provides visualization-ready subgroup performance tables for 510(k) inclusion
- Class imbalance handling via `class_weight="balanced"` in default configurations

**Evidence artifacts:** `reports/bias_report.json`, `reports/bias_report.html`

---

### Principle 7 — Focus Is Placed on the Performance of the Human-AI Team

> *Testing evaluates the complete human-AI system, not just the algorithm in isolation.*

**Implementation:**
- `model_server.py` implements `high_confidence` flag on every prediction to support appropriate human escalation
- Prediction logging with `PredictionAuditLogger` enables retrospective human-AI performance review
- Confidence threshold configurable per deployment context
- `PCCP_specification` includes `human_oversight_required` field
- Model card documents "out-of-scope uses" including autonomous use without clinician review

**Evidence artifacts:** `logs/predictions.jsonl`, `compliance/model_card.json`

---

### Principle 8 — Testing Demonstrates Performance in Clinically Relevant Conditions

> *Evaluation covers realistic conditions, including edge cases and multi-site performance.*

**Implementation:**
- `GMLPExperimentTracker.verify_train_test_separation()` programmatically verifies zero patient overlap
- `ModelRegressionTester` includes critical case testing (zero new false negatives on flagged cases)
- `MedicalModelEvaluator` evaluates at multiple sensitivity operating points (85%, 90%, 95%)
- Multi-site evaluation supported via `site_expansion` PCCP change type
- Drift detection (`DriftMonitor`) ensures evaluation conditions reflect deployment environment

**Evidence artifacts:** `train_test_overlap_count=0` in MLflow params; `reports/regression_test_report.json`

---

### Principle 9 — Users Are Provided Clear, Essential Information

> *Users have access to model description, intended use, inputs, outputs, and performance information.*

**Implementation:**
- `ModelCardGenerator` produces Google-style model cards extended with FDA regulatory metadata
- HTML model card includes: intended use, out-of-scope applications, training data description, subgroup performance tables, known limitations
- FastAPI `/docs` endpoint provides OpenAPI specification for all inference API inputs/outputs
- `compliance_note` included in every API response reminding users of clinical oversight requirement

**Evidence artifacts:** `docs/model_card.html`, `compliance/model_card.json`

---

### Principle 10 — Deployed Models Are Monitored for Performance

> *Post-deployment monitoring detects performance changes and there is a process for updating the model.*

**Implementation:**
- `DriftMonitor` monitors data drift (PSI, KS test), prediction drift, and label drift
- Evidently AI integration for rich HTML drift dashboards
- Three-level alert system: GREEN / YELLOW / RED with automated recommendations
- `PCCPManager` provides structured process for model updates without requiring new submissions
- FastAPI `/alerts` endpoint provides programmatic access to recent drift alerts
- Production prediction log (`predictions.jsonl`) enables retrospective performance analysis

**Evidence artifacts:** `logs/drift_reports.jsonl`, `compliance/pccp_specification.json`

---

## 2. PCCP — Predetermined Change Control Plan

### Overview

The FDA's PCCP framework (Draft Guidance, 2023) allows AI/ML device manufacturers to pre-define:
1. **Modification Protocol:** What types of changes are allowed
2. **Performance Validation Protocol:** How each change type is validated
3. **Impact Assessment:** When regulatory notification is required

This enables safe post-market model updates without requiring a new 510(k) for every change.

### Pre-Approved Change Types in MedML-Ops

| Change Type | Impact | Human Review | Regulatory Notification |
|-------------|--------|-------------|------------------------|
| Retraining — same architecture | Minor | No | No |
| Hyperparameter adjustment | Minor | No | No |
| Decision threshold adjustment | Moderate | Yes | No |
| Retraining — new data/sites | Moderate | Yes | Yes (PAS) |
| Preprocessing change | Moderate | Yes | No |
| Architecture change | Major/Critical | Yes | New 510(k) likely |

### PCCP Workflow

```
1. Developer proposes change → PCCPChangeRequest
       ↓
2. PCCPManager.validate_change() checks:
   - Is change type in PCCP specification?
   - Performance degradation within bounds?
   - Required testing completed?
       ↓
3. Impact level assigned: MINOR / MODERATE / MAJOR / CRITICAL
       ↓
4a. MINOR → Auto-approved → Deploy
4b. MODERATE → Human review → Approve/reject → Deploy
4c. MAJOR → Regulatory team review → Potentially new submission
4d. CRITICAL → New 510(k) or PMA supplement required
       ↓
5. Change logged in pccp_change_log.json for audit trail
```

### Usage

```python
from src.compliance.pccp_manager import PCCPSpecification, PCCPManager, PCCPChangeRequest, ChangeType

# Create default PCCP specification
spec = PCCPSpecification.create_default(
    device_name="ChestXR-AI v1",
    intended_use="Detection of pulmonary nodules in chest radiographs",
    baseline_auroc=0.92,
)
spec.save_json("compliance/pccp_specification.json")

# Validate a proposed change
manager = PCCPManager(spec)
change = PCCPChangeRequest(
    change_type=ChangeType.RETRAINING_SAME_ARCHITECTURE,
    description="Re-train with 2,000 new cases from Partner Site B",
    requester="Dr. ML Engineer",
    baseline_metrics={"auroc": 0.92, "sensitivity": 0.88},
    new_metrics={"auroc": 0.93, "sensitivity": 0.89},
)
report = manager.validate_change(change)
print(report.summary())
```

---

## 3. 510(k) Submission Preparation

### What This Pipeline Provides

For a 510(k) premarket notification, FDA requires:

| Required Element | MedML-Ops Artifact |
|-----------------|-------------------|
| Device description | `compliance/model_card.json` → `intended_use` |
| Indications for use | `compliance/model_card.json` → `primary_intended_uses` |
| Performance testing | `reports/evaluation_report.json` |
| Confidence intervals | `evaluation_report.json` → `confidence_intervals` |
| Subgroup analysis | `reports/bias_report.json` and `bias_report.html` |
| Training/test separation | MLflow `train_test_overlap_count=0` |
| Predicate comparison | `evaluation_report.json` → `predicate_comparison` |
| GMLP compliance | `compliance/gmlp_audit.json` |
| Algorithm description | `docs/ARCHITECTURE.md` |
| Software documentation | `docs/FDA_COMPLIANCE.md` |

### Generating Submission Package

```bash
# Run full pipeline
make pipeline CONFIG=configs/my_submission.yaml

# Generate compliance report
make report

# Check GMLP readiness
python -c "
import json
gmlp = json.load(open('compliance/gmlp_audit.json'))
print(f'GMLP Score: {gmlp[\"overall_score\"]*100:.1f}%')
print(f'Submission Ready: {gmlp[\"submission_ready\"]}')
"
```

### Substantial Equivalence Argument

The `MedicalModelEvaluator` supports statistical comparison to a predicate device:

```python
evaluator = MedicalModelEvaluator()
report = evaluator.evaluate(
    y_true, y_prob,
    predicate_metrics={
        "auroc": 0.88,      # Predicate device AUROC from 510(k) database
        "sensitivity": 0.85,
        "specificity": 0.82,
    }
)
# report.predicate_comparison contains:
# - delta vs predicate for each metric
# - substantially_equivalent (bool, |delta| ≤ 0.02)
# - Superiority and non-inferiority p-values
```

---

## 4. IEC 62304 Software Lifecycle Alignment

IEC 62304 defines requirements for medical device software development. Key lifecycle phases and their MedML-Ops alignment:

| IEC 62304 Activity | MedML-Ops Implementation |
|-------------------|-------------------------|
| **5.1 — Software development planning** | `configs/pipeline_config.yaml` with version-controlled configuration |
| **5.2 — Software requirements analysis** | `IntendedUse`, `PCCPSpecification.intended_use` |
| **5.3 — Software architectural design** | `docs/ARCHITECTURE.md` |
| **5.5 — Software unit implementation** | Modular src/ package with type hints |
| **5.7 — Software integration testing** | `tests/` with pytest; `make test-pipeline` |
| **5.8 — Software system testing** | `ModelRegressionTester` on golden test set |
| **6 — Software maintenance** | `PCCPManager`, `DriftMonitor` for post-market changes |
| **7 — Software risk management** | Bias flags, GMLP audit gaps documented as risks |
| **8 — Software configuration management** | Git versioning, MLflow run IDs, SHA-256 hashes |
| **9 — Software problem resolution** | Regression testing catches regressions; audit log for incidents |

### Software Safety Classification

Under IEC 62304, AI/ML medical device software is typically classified as:
- **Class C** if a failure could cause death or serious injury (e.g., autonomous diagnosis)
- **Class B** if used as clinical decision support with mandatory clinician review

MedML-Ops is designed for **Class B** (CDS with human oversight) deployments. Class C deployments require additional design controls.

---

## 5. 21 CFR Part 820 Quality System Considerations

The FDA's Quality System Regulation (21 CFR Part 820) applies to Class II and III medical devices. Key requirements:

| 21 CFR 820 Section | MedML-Ops Alignment |
|-------------------|---------------------|
| **820.30 — Design controls** | Pipeline config as design spec; GMLP checklist as design review |
| **820.40 — Document controls** | Git versioning for all code and configs; MLflow for experiment records |
| **820.50 — Purchasing controls** | `requirements.txt` with pinned dependencies (supply chain) |
| **820.70 — Production processes** | Reproducible training with locked seeds and logged environment |
| **820.75 — Process validation** | `GMLPComplianceChecker` validates training process completeness |
| **820.80 — Receiving/in-process/final acceptance** | `ModelRegressionTester` as acceptance test |
| **820.100 — Corrective and preventive action (CAPA)** | Bias flags, drift alerts, regression failures trigger CAPA |
| **820.180 — General records** | MLflow artifacts, prediction logs, compliance JSON files |
| **820.198 — Complaint files** | `PredictionAuditLogger` maintains prediction audit trail |

---

## 6. Regulatory References

| Document | Link |
|----------|------|
| FDA GMLP Guiding Principles (2021) | https://www.fda.gov/medical-devices/software-medical-device-samd/good-machine-learning-practice-medical-device-development-guiding-principles |
| FDA AI/ML Action Plan (2021) | https://www.fda.gov/media/145022/download |
| FDA PCCP Draft Guidance (2023) | https://www.fda.gov/regulatory-information/search-fda-guidance-documents/marketing-submission-recommendations-predetermined-change-control-plan |
| FDA Software Guidance (2023) | https://www.fda.gov/regulatory-information/search-fda-guidance-documents/software-functions-intended-use-computer-software-assurance-instead-software-validation |
| IEC 62304:2006+AMD1:2015 | https://www.iso.org/standard/38421.html |
| 21 CFR Part 820 | https://www.ecfr.gov/current/title-21/chapter-I/subchapter-H/part-820 |
| FDA 510(k) Database | https://www.fda.gov/medical-devices/premarket-notification-510k/510k-premarket-notification |
| Model Cards (Mitchell et al. 2019) | https://arxiv.org/abs/1810.03993 |
| Equality of Opportunity (Hardt et al. 2016) | https://arxiv.org/abs/1610.02413 |

---

*This document was generated as part of the MedML-Ops pipeline and is intended for educational purposes. It does not constitute legal or regulatory advice. Consult qualified regulatory counsel before FDA submissions.*
