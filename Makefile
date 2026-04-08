# =============================================================================
# MedML-Ops Makefile
# =============================================================================
# Common commands for the FDA-compliant MLOps pipeline.
#
# Usage:
#   make pipeline              # Run full pipeline
#   make train                 # Training only
#   make evaluate              # Evaluation + bias analysis
#   make serve                 # Start inference server
#   make monitor               # Start drift monitor
#   make report                # Generate compliance report
#   make docker-up             # Start full Docker stack
#
# Override config: make pipeline CONFIG=configs/my_experiment.yaml
# =============================================================================

# Defaults (override from CLI)
CONFIG ?= configs/pipeline_config.yaml
STEPS  ?= validate,train,evaluate,bias,regression,gmlp,model_card,report
PORT   ?= 8000
MODEL  ?= models/current
MODEL_REGISTRY ?= $(MODEL_REGISTRY_PATH)

PYTHON := python
PIP    := pip

# Formatting / linting
BLACK  := black
ISORT  := isort
MYPY   := mypy

.DEFAULT_GOAL := help

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------
.PHONY: help
help:  ## Show this help message
	@echo ""
	@echo "MedML-Ops — FDA-Compliant MLOps Pipeline"
	@echo "========================================="
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
.PHONY: install
install:  ## Install all Python dependencies
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .
	@echo "✓ Dependencies installed"

.PHONY: install-dev
install-dev:  ## Install dev dependencies (linting, testing)
	$(MAKE) install
	$(PIP) install black isort mypy pytest pytest-cov pre-commit
	pre-commit install
	@echo "✓ Dev dependencies installed"

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
.PHONY: pipeline
pipeline:  ## Run the full FDA-compliant pipeline
	$(PYTHON) scripts/run_pipeline.py --config $(CONFIG) --steps $(STEPS)

.PHONY: validate
validate:  ## Run data validation only
	$(PYTHON) scripts/run_pipeline.py --config $(CONFIG) --steps validate

.PHONY: train
train:  ## Run model training (with HPO)
	$(PYTHON) scripts/run_pipeline.py --config $(CONFIG) --steps train

.PHONY: train-fast
train-fast:  ## Run training without HPO (faster iteration)
	$(PYTHON) scripts/run_pipeline.py --config $(CONFIG) --steps train --skip-hpo

.PHONY: evaluate
evaluate:  ## Run model evaluation + bias analysis
	$(PYTHON) scripts/run_pipeline.py --config $(CONFIG) --steps evaluate,bias

.PHONY: compliance
compliance:  ## Run GMLP audit + model card generation
	$(PYTHON) scripts/run_pipeline.py --config $(CONFIG) --steps gmlp,model_card

.PHONY: report
report:  ## Generate compliance HTML report from existing outputs
	$(PYTHON) scripts/generate_report.py \
		--reports-dir reports/ \
		--output reports/compliance_summary.html
	@echo "✓ Report: reports/compliance_summary.html"

# ---------------------------------------------------------------------------
# Serving
# ---------------------------------------------------------------------------
.PHONY: serve
serve:  ## Start the FastAPI inference server
	uvicorn src.deployment.model_server:app \
		--host 0.0.0.0 --port $(PORT) --reload
	@echo "✓ Serving at http://localhost:$(PORT)"
	@echo "  Docs: http://localhost:$(PORT)/docs"

.PHONY: serve-prod
serve-prod:  ## Start production inference server (no reload, multiple workers)
	MODEL_PATH=$(MODEL) uvicorn src.deployment.model_server:app \
		--host 0.0.0.0 --port $(PORT) --workers 4

# ---------------------------------------------------------------------------
# Monitoring
# ---------------------------------------------------------------------------
.PHONY: monitor
monitor:  ## Start the drift monitoring server
	uvicorn src.monitoring.drift_monitor:app \
		--host 0.0.0.0 --port 8001 --reload

.PHONY: mlflow-ui
mlflow-ui:  ## Start the MLflow tracking server UI
	mlflow ui --host 0.0.0.0 --port 5000 --backend-store-uri mlruns

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------
.PHONY: docker-up
docker-up:  ## Start full Docker stack (MLflow + serving + monitoring)
	docker-compose -f src/deployment/docker/docker-compose.yml up --build

.PHONY: docker-down
docker-down:  ## Stop all Docker services
	docker-compose -f src/deployment/docker/docker-compose.yml down

.PHONY: docker-train
docker-train:  ## Run training in Docker container
	docker-compose -f src/deployment/docker/docker-compose.yml \
		--profile training run --rm training \
		python scripts/run_pipeline.py --config $(CONFIG) --steps train

.PHONY: docker-build
docker-build:  ## Build all Docker images
	docker build -f src/deployment/docker/Dockerfile.train \
		-t medml-ops/training:latest .
	docker build -f src/deployment/docker/Dockerfile.serve \
		-t medml-ops/serving:latest .

# ---------------------------------------------------------------------------
# Code Quality
# ---------------------------------------------------------------------------
.PHONY: format
format:  ## Format code with black + isort
	$(BLACK) src/ scripts/ tests/
	$(ISORT) src/ scripts/ tests/
	@echo "✓ Code formatted"

.PHONY: lint
lint:  ## Run linting checks
	$(BLACK) --check src/ scripts/
	$(ISORT) --check-only src/ scripts/
	@echo "✓ Linting passed"

.PHONY: typecheck
typecheck:  ## Run mypy type checking
	$(MYPY) src/ --ignore-missing-imports
	@echo "✓ Type checking passed"

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------
.PHONY: test
test:  ## Run unit tests
	pytest tests/ -v --tb=short

.PHONY: test-cov
test-cov:  ## Run tests with coverage report
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing
	@echo "✓ Coverage report: htmlcov/index.html"

.PHONY: test-pipeline
test-pipeline:  ## Run end-to-end pipeline test with demo data
	$(PYTHON) scripts/run_pipeline.py --config $(CONFIG) --steps all --dry-run
	$(PYTHON) scripts/run_pipeline.py --config $(CONFIG) --steps validate,train,evaluate,bias \
		--skip-hpo
	@echo "✓ Pipeline integration test passed"

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
.PHONY: clean
clean:  ## Remove generated artifacts (keeps mlruns and models)
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache/ htmlcov/ .mypy_cache/
	@echo "✓ Cleaned up cached files"

.PHONY: clean-reports
clean-reports:  ## Remove generated reports
	rm -rf reports/*.json reports/*.html
	@echo "✓ Reports cleaned"

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------
.PHONY: version
version:  ## Show installed package versions
	$(PYTHON) -c "import mlflow; print(f'MLflow: {mlflow.__version__}')"
	$(PYTHON) -c "import sklearn; print(f'scikit-learn: {sklearn.__version__}')"
	$(PYTHON) -c "import fastapi; print(f'FastAPI: {fastapi.__version__}')"
	$(PYTHON) -c "import pandas; print(f'pandas: {pandas.__version__}')"

lint:
	ruff check src/
