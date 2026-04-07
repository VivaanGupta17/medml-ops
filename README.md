# MedML-Ops: FDA-Compliant Machine Learning Operations Pipeline for Medical AI

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MLflow](https://img.shields.io/badge/MLflow-2.x-orange.svg)](https://mlflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![FDA GMLP](https://img.shields.io/badge/FDA-GMLP%20Compliant-red.svg)](https://www.fda.gov/medical-devices/software-medical-device-samd/good-machine-learning-practice-medical-device-development-guiding-principles)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen.svg)](https://pre-commit.com/)

> **End-to-end MLOps pipeline engineered for medical device AI — incorporating FDA Good Machine Learning Practice (GMLP) principles, PCCP-ready model versioning, automated demographic bias monitoring, and 510(k)-aligned evaluation workflows.**

---

## Why This Exists

Most ML pipelines stop at model accuracy. Medical AI requires something fundamentally different: auditability, bias transparency, regulatory alignment, and the ability to safely update models post-market under a Predetermined Change Control Plan (PCCP).

This pipeline is built around the FDA's 10 GMLP guiding principles and structured to support the full Software as a Medical Device (SaMD) lifecycle — from data ingestion through post-market surveillance.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         MedML-Ops Pipeline                              │
│                                                                         │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌───────────────────┐ │
│  │   Data   │    │  Schema  │    │ Training │    │   Experiment      │ │
│  │Ingestion │───▶│Validation│───▶│ Pipeline │───▶│   Tracking        │ │
│  │          │    │+ Drift   │    │(AutoML)  │    │   (MLflow)        │ │
│  └──────────┘    └──────────┘    └──────────┘    └───────────────────┘ │
│                       │                │                   │           │
│                       ▼                ▼                   ▼           │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌───────────────────┐ │
│  │   GMLP   │    │  Bias    │    │  Model   │    │   Regression      │ │
│  │Compliance│◀───│ Detector │◀───│Evaluator │◀───│   Testing         │ │
│  │ Checklist│    │(Fairness)│    │(Clinical)│    │   (PCCP-aligned)  │ │
│  └──────────┘    └──────────┘    └──────────┘    └───────────────────┘ │
│        │              │                │                               │
│        ▼              ▼                ▼                               │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌───────────────────┐ │
│  │  PCCP    │    │  Model   │    │  Model   │    │   FastAPI         │ │
│  │ Manager  │───▶│ Registry │───▶│  Server  │───▶│   Deployment      │ │
│  │          │    │(MLflow)  │    │(Versioned│    │   (Dockerized)    │ │
│  └──────────┘    └──────────┘    └──────────┘    └───────────────────┘ │
│                                       │                               │
│                                       ▼                               │
│                  ┌────────────────────────────────────┐              │
│                  │         Production Monitoring       │              │
│                  │  Data Drift │ Prediction Drift      │              │
│                  │  PSI / KS  │ Evidently AI           │              │
│                  │  Alerting  │ Model Cards (Auto-gen) │              │
│                  └────────────────────────────────────┘              │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Key Differentiators

### 1. FDA GMLP Compliance Built-In
Every component maps to one or more of the FDA's 10 Good Machine Learning Practice principles. The `gmlp_checklist.py` module automatically audits your pipeline state and generates compliance documentation suitable for 510(k) submissions.

### 2. Demographic Bias Detection (FDA-Required)
`bias_detector.py` stratifies model performance across age groups, biological sex, and self-reported ethnicity — computing equalized odds, demographic parity gap, and calibration curves per subgroup. This directly addresses FDA guidance on eliminating bias from clinical AI.

### 3. PCCP-Ready Change Control
`pccp_manager.py` implements the FDA's Predetermined Change Control Plan framework, allowing you to pre-define and validate model updates without requiring a new 510(k) for each version change.

### 4. Clinical Operating Point Selection
`model_evaluator.py` goes beyond AUROC to support Youden's J threshold selection, iso-sensitivity curves, and comparison against predicate device performance benchmarks for 510(k) substantial equivalence arguments.

### 5. Automated Model Cards
`model_card_generator.py` produces Google-style model cards extended with FDA submission metadata — intended use, training population demographics, known limitations, and subgroup performance tables.

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Experiment Tracking | [MLflow](https://mlflow.org/) 2.x |
| Data Validation | [Great Expectations](https://greatexpectations.io/) + custom DICOM validation |
| Model Serving | [FastAPI](https://fastapi.tiangolo.com/) |
| Drift Monitoring | [Evidently AI](https://www.evidentlyai.com/) |
| Containerization | Docker + Docker Compose |
| AutoML / HPO | Optuna |
| Statistical Testing | SciPy, statsmodels |
| Calibration | scikit-learn calibration, netcal |
| Configuration | YAML + Pydantic |
| Code Quality | Black, isort, mypy, pre-commit |

---

## Repository Structure

```
medml-ops/
├── src/
│   ├── data_validation/
│   │   ├── schema_validator.py       # Great Expectations-style schema + DICOM validation
│   │   └── bias_detector.py          # Demographic fairness metrics & subgroup analysis
│   ├── training/
│   │   ├── experiment_tracker.py     # MLflow GMLP-compliant experiment tracking
│   │   └── automated_training.py     # Optuna HPO + reproducible training pipeline
│   ├── evaluation/
│   │   ├── model_evaluator.py        # Clinical operating point + calibration + 510(k) metrics
│   │   └── regression_testing.py     # PCCP-aligned regression & slice-based testing
│   ├── deployment/
│   │   ├── model_server.py           # FastAPI inference server with prediction logging
│   │   └── docker/
│   │       ├── Dockerfile.train      # Training container
│   │       ├── Dockerfile.serve      # Serving container
│   │       └── docker-compose.yml    # Full pipeline orchestration
│   ├── monitoring/
│   │   ├── drift_monitor.py          # PSI, KS, chi-square drift detection + Evidently AI
│   │   └── model_card_generator.py   # FDA-ready automated model cards
│   └── compliance/
│       ├── gmlp_checklist.py         # FDA GMLP 10-principle automated audit
│       └── pccp_manager.py           # Predetermined Change Control Plan manager
├── configs/
│   └── pipeline_config.yaml          # Full pipeline configuration
├── scripts/
│   ├── run_pipeline.py               # End-to-end pipeline execution
│   └── generate_report.py            # Compliance report generation
├── docs/
│   ├── FDA_COMPLIANCE.md             # GMLP, PCCP, 510(k), IEC 62304, 21 CFR 820
│   └── ARCHITECTURE.md               # Pipeline architecture deep-dive
├── tests/                            # Unit and integration tests
├── notebooks/                        # Exploratory analysis notebooks
├── Makefile                          # Common pipeline commands
├── requirements.txt
├── setup.py
└── .gitignore
```

---

## Quick Start

### Prerequisites
- Python 3.10+
- Docker & Docker Compose
- MLflow tracking server (or use local file store)

### Installation

```bash
git clone https://github.com/yourusername/medml-ops.git
cd medml-ops

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### Run the Full Pipeline

```bash
# Configure your pipeline
cp configs/pipeline_config.yaml configs/my_experiment.yaml
# Edit configs/my_experiment.yaml

# Run end-to-end
make pipeline CONFIG=configs/my_experiment.yaml

# Or step-by-step:
make validate    # Data validation
make train       # Model training
make evaluate    # Evaluation + bias audit
make serve       # Start inference server
make monitor     # Start drift monitoring
make report      # Generate compliance report
```

### Docker Deployment

```bash
# Build and start the full pipeline
cd src/deployment/docker
docker-compose up --build

# Services:
# - MLflow UI:       http://localhost:5000
# - FastAPI server:  http://localhost:8000
# - API Docs:        http://localhost:8000/docs
# - Monitoring:      http://localhost:8001
```

---

## Pipeline Walkthrough

### Step 1: Data Validation

```python
from src.data_validation.schema_validator import MedicalDataValidator

validator = MedicalDataValidator(config_path="configs/pipeline_config.yaml")
report = validator.validate_dataset(
    data_path="data/training_cohort.csv",
    reference_path="data/reference_schema.json"
)
print(report.summary())
```

### Step 2: Bias Detection

```python
from src.data_validation.bias_detector import DemographicBiasDetector

detector = DemographicBiasDetector(
    sensitive_attributes=["age_group", "sex", "ethnicity"]
)
bias_report = detector.analyze(
    y_true=labels,
    y_pred=predictions,
    demographics=demographic_df
)
bias_report.save_html("reports/bias_audit_v1.html")
```

### Step 3: Experiment Tracking

```python
from src.training.experiment_tracker import GMLPExperimentTracker

with GMLPExperimentTracker(experiment_name="chest_xray_classifier") as tracker:
    tracker.log_dataset_metadata(dataset_path, version="v2.1")
    tracker.log_hyperparams(config)
    # ... training loop ...
    tracker.log_metrics({"auroc": 0.94, "f1": 0.87})
    tracker.log_gmlp_documentation()
```

### Step 4: Compliance Audit

```python
from src.compliance.gmlp_checklist import GMLPComplianceChecker

checker = GMLPComplianceChecker(run_id=tracker.run_id)
report = checker.run_full_audit()
report.export_pdf("compliance/gmlp_audit_v1.pdf")
```

---

## FDA Regulatory Alignment

| FDA Principle | Implementation |
|--------------|----------------|
| Multi-disciplinary expertise | Documented stakeholder sign-offs in PCCP |
| Good software engineering | Pre-commit hooks, type hints, unit tests |
| Clinical study design | Stratified splits, demographic balance checks |
| Data management | Schema validation, DICOM metadata audit, drift detection |
| Re-training practices | PCCP manager, automated regression testing |
| Model design tradeoffs | Clinical operating point selection, sensitivity/specificity curves |
| Human-AI collaboration | Confidence thresholding, uncertainty quantification |
| Test dataset independence | Train/test separation verification in GMLP checklist |
| Evaluation reflects deployment | Domain shift detection, production drift monitoring |
| Monitoring post-deployment | Evidently AI drift, PSI/KS alerts, automated model cards |

See [docs/FDA_COMPLIANCE.md](docs/FDA_COMPLIANCE.md) for the full regulatory mapping.

---

## Relevant Standards & Guidance

- **FDA GMLP** — [Good Machine Learning Practice for Medical Device Development](https://www.fda.gov/medical-devices/software-medical-device-samd/good-machine-learning-practice-medical-device-development-guiding-principles)
- **FDA AI/ML Action Plan** — [Artificial Intelligence/Machine Learning-Based Software as a Medical Device Action Plan](https://www.fda.gov/media/145022/download)
- **IEC 62304** — Medical device software lifecycle processes
- **21 CFR Part 820** — FDA Quality System Regulation
- **ISO 13485** — Medical devices quality management systems
- **FDA 510(k)** — Premarket notification framework

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). All contributions must maintain GMLP compliance standards documented in [docs/FDA_COMPLIANCE.md](docs/FDA_COMPLIANCE.md).

---

## License

MIT License — see [LICENSE](LICENSE).

---

*Built for the intersection of software engineering rigor and FDA regulatory science. Designed to demonstrate what production medical AI engineering actually looks like.*
