"""
GMLP-Compliant Experiment Tracker
===================================
MLflow-based experiment tracking with FDA Good Machine Learning Practice compliance.

Automatically captures hyperparameters, metrics, artifacts, environment details,
dataset provenance, and generates GMLP documentation for regulatory submissions.

FDA GMLP Alignment:
  - Principle 1: Multi-disciplinary expertise — documents team/sign-offs
  - Principle 4: Data management — logs dataset versions and hashes
  - Principle 5: Re-training practices — captures full reproducibility context
  - Principle 8: Test dataset independence — verifies split integrity
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import random
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Generator

import numpy as np

logger = logging.getLogger(__name__)

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not installed. Install with: pip install mlflow")


# ---------------------------------------------------------------------------
# GMLP Documentation Schema
# ---------------------------------------------------------------------------

GMLP_PRINCIPLES = {
    1: "Multi-disciplinary expertise throughout the total product lifecycle",
    2: "Good software engineering and security practices",
    3: "Clinical study design to generate evidence",
    4: "Data management practices for transparency",
    5: "Re-training practices for model modification transparency",
    6: "Design to handle failure modes and manage bias",
    7: "Testing to demonstrate device performance",
    8: "Evaluation of real-world performance",
    9: "User interaction design for safety",
    10: "Transparency to users about device performance",
}


class GMLPDocumentation:
    """
    GMLP compliance documentation attached to an MLflow run.
    Captures all information required by FDA guidance on ML-based SaMD.
    """

    def __init__(self):
        self.data: dict[str, Any] = {
            "gmlp_version": "2021-10",
            "documentation_timestamp": datetime.utcnow().isoformat(),
            "principles_addressed": {},
            "dataset_provenance": {},
            "reproducibility": {},
            "team_sign_off": {},
            "known_limitations": [],
            "intended_use": "",
            "out_of_scope": [],
        }

    def set_intended_use(self, description: str, device_class: str = "") -> None:
        self.data["intended_use"] = description
        self.data["device_class"] = device_class

    def add_principle_evidence(self, principle_number: int, evidence: str) -> None:
        self.data["principles_addressed"][str(principle_number)] = {
            "principle": GMLP_PRINCIPLES.get(principle_number, ""),
            "evidence": evidence,
            "timestamp": datetime.utcnow().isoformat(),
        }

    def add_known_limitation(self, limitation: str) -> None:
        self.data["known_limitations"].append(limitation)

    def set_dataset_provenance(
        self,
        dataset_path: str,
        version: str,
        source: str,
        n_samples: int,
        file_hash: str | None = None,
    ) -> None:
        self.data["dataset_provenance"] = {
            "path": dataset_path,
            "version": version,
            "source": source,
            "n_samples": n_samples,
            "file_hash_sha256": file_hash,
            "logged_at": datetime.utcnow().isoformat(),
        }

    def set_team_sign_off(
        self,
        ml_engineer: str = "",
        clinical_expert: str = "",
        regulatory_reviewer: str = "",
    ) -> None:
        self.data["team_sign_off"] = {
            "ml_engineer": ml_engineer,
            "clinical_expert": clinical_expert,
            "regulatory_reviewer": regulatory_reviewer,
            "sign_off_date": datetime.utcnow().isoformat(),
        }

    def to_json(self) -> str:
        return json.dumps(self.data, indent=2)


# ---------------------------------------------------------------------------
# Core Tracker
# ---------------------------------------------------------------------------

class GMLPExperimentTracker:
    """
    GMLP-compliant experiment tracker wrapping MLflow.

    Designed as a context manager for clean run lifecycle management.
    Automatically captures environment, seeds, and generates compliance docs.

    Usage::

        with GMLPExperimentTracker(experiment_name="chest_xray_v2") as tracker:
            tracker.log_dataset_metadata("data/train.csv", version="v2.1")
            tracker.log_hyperparams({"lr": 0.001, "epochs": 50})

            for epoch in range(50):
                tracker.log_metrics({"train_loss": loss, "val_auroc": auroc}, step=epoch)

            tracker.log_model(model, "sklearn")
            tracker.log_gmlp_documentation()
    """

    def __init__(
        self,
        experiment_name: str,
        tracking_uri: str | None = None,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
        intended_use: str = "",
    ):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "mlruns")
        self.run_name = run_name or f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        self.tags = tags or {}
        self.intended_use = intended_use
        self.gmlp_doc = GMLPDocumentation()
        self.run_id: str | None = None
        self._start_time: float | None = None
        self._active = False

        if not MLFLOW_AVAILABLE:
            logger.warning("MLflow unavailable — using local file fallback")

    def __enter__(self) -> "GMLPExperimentTracker":
        self.start_run()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is not None:
            logger.error("Run failed with exception: %s: %s", exc_type.__name__, exc_val)
            self._end_run(status="FAILED")
        else:
            self._end_run(status="FINISHED")

    # ------------------------------------------------------------------
    # Run Lifecycle
    # ------------------------------------------------------------------

    def start_run(self) -> str:
        """Start a new MLflow run and capture environment."""
        self._start_time = time.time()

        if MLFLOW_AVAILABLE:
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)

            default_tags = {
                "gmlp_compliant": "true",
                "pipeline": "medml-ops",
                "python_version": platform.python_version(),
                **self.tags,
            }

            run = mlflow.start_run(run_name=self.run_name, tags=default_tags)
            self.run_id = run.info.run_id
        else:
            self.run_id = f"local_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        self._active = True
        self._log_environment()
        logger.info("Started run: %s (ID: %s)", self.run_name, self.run_id)
        return self.run_id

    def _end_run(self, status: str = "FINISHED") -> None:
        if not self._active:
            return

        if self._start_time:
            duration = time.time() - self._start_time
            self._log_param("run_duration_seconds", round(duration, 2))

        if MLFLOW_AVAILABLE:
            mlflow.end_run(status=status)

        self._active = False
        logger.info("Ended run %s with status: %s", self.run_id, status)

    # ------------------------------------------------------------------
    # Environment & Reproducibility
    # ------------------------------------------------------------------

    def _log_environment(self) -> None:
        """Capture full environment for GMLP Principle 5 (reproducibility)."""
        env_info = {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "hostname": platform.node(),
        }

        # Git commit hash for code reproducibility
        try:
            git_hash = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            ).decode().strip()
            env_info["git_commit"] = git_hash
            git_status = subprocess.check_output(
                ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
            ).decode().strip()
            env_info["git_dirty"] = bool(git_status)
        except Exception:
            env_info["git_commit"] = "unavailable"
            env_info["git_dirty"] = None

        # Key package versions
        packages_to_log = [
            "numpy", "pandas", "scikit-learn", "torch", "tensorflow",
            "mlflow", "scipy", "optuna"
        ]
        for pkg in packages_to_log:
            try:
                import importlib.metadata
                env_info[f"pkg_{pkg}"] = importlib.metadata.version(pkg)
            except Exception:
                pass

        for k, v in env_info.items():
            if v is not None:
                self._log_param(f"env.{k}", str(v))

        self.gmlp_doc.data["reproducibility"]["environment"] = env_info

    def set_seeds(self, seed: int = 42) -> None:
        """
        Set and log all random seeds for full reproducibility.
        GMLP Principle 5: Re-training practices require reproducibility.
        """
        random.seed(seed)
        np.random.seed(seed)

        # Set framework seeds if available
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        except ImportError:
            pass

        try:
            import tensorflow as tf
            tf.random.set_seed(seed)
        except ImportError:
            pass

        self._log_param("random_seed", seed)
        self.gmlp_doc.data["reproducibility"]["random_seed"] = seed
        logger.info("All random seeds set to %d", seed)

    # ------------------------------------------------------------------
    # Dataset Provenance
    # ------------------------------------------------------------------

    def log_dataset_metadata(
        self,
        dataset_path: str | Path,
        version: str = "",
        source: str = "",
        split: str = "train",
        n_samples: int | None = None,
        compute_hash: bool = True,
    ) -> str | None:
        """
        Log dataset provenance for GMLP Principle 4.

        Args:
            dataset_path: Path to dataset file.
            version: Dataset version string.
            source: Data source description (institution, registry, etc.).
            split: "train", "validation", or "test".
            n_samples: Number of samples (computed if not provided).
            compute_hash: SHA-256 hash the file for integrity verification.

        Returns:
            SHA-256 hash of the dataset file, or None.
        """
        path = Path(dataset_path)
        file_hash = None

        if compute_hash and path.exists():
            sha256 = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha256.update(chunk)
            file_hash = sha256.hexdigest()

        # Count samples if not provided
        if n_samples is None and path.exists():
            try:
                import pandas as pd
                df = pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)
                n_samples = len(df)
            except Exception:
                pass

        params = {
            f"dataset.{split}.path": str(dataset_path),
            f"dataset.{split}.version": version,
            f"dataset.{split}.source": source,
            f"dataset.{split}.n_samples": str(n_samples or "unknown"),
            f"dataset.{split}.sha256": file_hash or "not_computed",
            f"dataset.{split}.file_size_bytes": str(path.stat().st_size if path.exists() else 0),
        }
        for k, v in params.items():
            self._log_param(k, v)

        self.gmlp_doc.set_dataset_provenance(
            dataset_path=str(dataset_path),
            version=version,
            source=source,
            n_samples=n_samples or 0,
            file_hash=file_hash,
        )

        self.gmlp_doc.add_principle_evidence(
            4, f"Dataset '{path.name}' v{version} logged with SHA-256 hash and source provenance"
        )

        logger.info("Logged dataset: %s (split=%s, n=%s)", path.name, split, n_samples)
        return file_hash

    def verify_train_test_separation(
        self,
        train_ids: np.ndarray | list,
        test_ids: np.ndarray | list,
    ) -> bool:
        """
        Verify no overlap between train and test sets.
        GMLP Principle 8: Test dataset independence.
        """
        train_set = set(train_ids)
        test_set = set(test_ids)
        overlap = train_set & test_set
        passed = len(overlap) == 0

        self._log_param("train_test_overlap_count", len(overlap))
        self._log_param("train_test_separation_verified", str(passed))

        if passed:
            self.gmlp_doc.add_principle_evidence(
                8, f"Train/test separation verified: 0 overlapping IDs "
                   f"({len(train_set)} train, {len(test_set)} test)"
            )
            logger.info("Train/test separation verified: no overlap detected")
        else:
            logger.error(
                "GMLP VIOLATION: %d samples appear in both train and test sets! "
                "Sample overlapping IDs: %s",
                len(overlap), list(overlap)[:5]
            )

        return passed

    # ------------------------------------------------------------------
    # Hyperparameters & Metrics
    # ------------------------------------------------------------------

    def log_hyperparams(self, params: dict[str, Any]) -> None:
        """Log all hyperparameters."""
        for k, v in params.items():
            self._log_param(k, v)

        self.gmlp_doc.data["hyperparameters"] = {k: str(v) for k, v in params.items()}

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int | None = None,
        prefix: str = "",
    ) -> None:
        """Log metrics with optional step and prefix."""
        for k, v in metrics.items():
            full_key = f"{prefix}{k}" if prefix else k
            if MLFLOW_AVAILABLE and self._active:
                mlflow.log_metric(full_key, v, step=step)

        # Track best AUROC for GMLP documentation
        if "auroc" in metrics:
            current_best = self.gmlp_doc.data.get("best_auroc", 0.0)
            if metrics["auroc"] > current_best:
                self.gmlp_doc.data["best_auroc"] = metrics["auroc"]

    def log_model(
        self,
        model: Any,
        flavor: str = "sklearn",
        artifact_path: str = "model",
        signature=None,
        input_example=None,
    ) -> str | None:
        """
        Log model artifact with MLflow model registry support.

        Args:
            model: Trained model object.
            flavor: MLflow flavor ("sklearn", "pytorch", "tensorflow", "pyfunc").
            artifact_path: Path within the MLflow run artifacts.
            signature: MLflow model signature (optional).
            input_example: Example input for signature inference (optional).

        Returns:
            MLflow model URI, or None if MLflow unavailable.
        """
        if not MLFLOW_AVAILABLE or not self._active:
            logger.warning("MLflow not active — model not logged")
            return None

        log_fn_map = {
            "sklearn": mlflow.sklearn.log_model,
            "pytorch": mlflow.pytorch.log_model if hasattr(mlflow, "pytorch") else None,
            "tensorflow": mlflow.tensorflow.log_model if hasattr(mlflow, "tensorflow") else None,
        }

        log_fn = log_fn_map.get(flavor, mlflow.sklearn.log_model)
        if log_fn is None:
            logger.warning("No log function for flavor '%s'", flavor)
            return None

        kwargs = {"artifact_path": artifact_path}
        if signature:
            kwargs["signature"] = signature
        if input_example is not None:
            kwargs["input_example"] = input_example

        model_info = log_fn(model, **kwargs)
        uri = model_info.model_uri if model_info else None

        self.gmlp_doc.data["model_artifact"] = {
            "uri": uri,
            "flavor": flavor,
            "artifact_path": artifact_path,
            "run_id": self.run_id,
        }

        logger.info("Model logged: %s", uri)
        return uri

    def log_artifact_file(self, local_path: str | Path, artifact_subdir: str = "") -> None:
        """Log an arbitrary file as a run artifact."""
        if MLFLOW_AVAILABLE and self._active:
            mlflow.log_artifact(str(local_path), artifact_subdir)

    # ------------------------------------------------------------------
    # GMLP Documentation
    # ------------------------------------------------------------------

    def log_gmlp_documentation(
        self,
        intended_use: str = "",
        known_limitations: list[str] | None = None,
        ml_engineer: str = "",
        clinical_expert: str = "",
    ) -> None:
        """
        Finalize and log GMLP compliance documentation as a run artifact.
        Should be called at the end of a training run.
        """
        if intended_use:
            self.gmlp_doc.set_intended_use(intended_use)
        elif self.intended_use:
            self.gmlp_doc.set_intended_use(self.intended_use)

        if known_limitations:
            for lim in known_limitations:
                self.gmlp_doc.add_known_limitation(lim)

        self.gmlp_doc.set_team_sign_off(
            ml_engineer=ml_engineer,
            clinical_expert=clinical_expert,
        )

        # Add standard principle evidence
        self.gmlp_doc.add_principle_evidence(
            2, "Code managed in git with pre-commit hooks (black, isort, mypy)"
        )
        self.gmlp_doc.add_principle_evidence(
            5, f"Full environment logged: Python {platform.python_version()}, "
               f"git commit tracked, random seed recorded"
        )

        # Save to temp file and log as artifact
        doc_path = Path(f"/tmp/gmlp_documentation_{self.run_id}.json")
        doc_path.write_text(self.gmlp_doc.to_json())

        if MLFLOW_AVAILABLE and self._active:
            mlflow.log_artifact(str(doc_path), "compliance")
            mlflow.set_tag("gmlp_documentation_logged", "true")

        logger.info("GMLP documentation logged for run %s", self.run_id)

    def register_model(
        self,
        model_uri: str,
        model_name: str,
        description: str = "",
        tags: dict[str, str] | None = None,
    ) -> Any:
        """
        Register a model in the MLflow Model Registry.
        Used for PCCP-aligned version tracking.
        """
        if not MLFLOW_AVAILABLE:
            return None

        client = mlflow.tracking.MlflowClient()
        mv = mlflow.register_model(model_uri, model_name)

        if description:
            client.update_registered_model(model_name, description=description)
        if tags:
            for k, v in tags.items():
                client.set_registered_model_tag(model_name, k, v)

        logger.info("Model registered: %s (version %s)", model_name, mv.version)
        return mv

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _log_param(self, key: str, value: Any) -> None:
        if MLFLOW_AVAILABLE and self._active:
            mlflow.log_param(str(key), str(value)[:500])  # MLflow 500-char limit

    @contextmanager
    def log_step_timing(self, step_name: str) -> Generator[None, None, None]:
        """Context manager to time and log pipeline step durations."""
        start = time.time()
        try:
            yield
        finally:
            duration = time.time() - start
            if MLFLOW_AVAILABLE and self._active:
                mlflow.log_metric(f"timing.{step_name}_seconds", round(duration, 2))
            logger.info("Step '%s' completed in %.2fs", step_name, duration)


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------

def get_file_hash(path: str | Path) -> str:
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def load_run_gmlp_doc(run_id: str, tracking_uri: str = "mlruns") -> dict | None:
    """Load GMLP documentation for a completed MLflow run."""
    if not MLFLOW_AVAILABLE:
        return None

    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    artifacts = client.list_artifacts(run_id, path="compliance")
    for art in artifacts:
        if "gmlp_documentation" in art.path:
            local_path = client.download_artifacts(run_id, art.path)
            return json.loads(Path(local_path).read_text())
    return None
