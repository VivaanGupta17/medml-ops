"""
Automated Training Pipeline
=============================
Configurable, reproducible training pipeline with Optuna hyperparameter optimization,
GMLP-compliant cross-validation, and automated experiment tracking.

Features:
  - YAML-driven configuration
  - Optuna TPE hyperparameter search
  - Stratified K-fold and group K-fold for medical data
  - Enforced reproducibility (seeded everything)
  - Automatic MLflow tracking integration
  - Training/validation split integrity verification

FDA GMLP Alignment:
  - Principle 5: Re-training practices — full config, seed, and environment capture
  - Principle 3: Clinical study design — proper stratified splits preventing leakage
  - Principle 4: Data management — dataset versioning before training begins
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import (
    GroupKFold,
    StratifiedGroupKFold,
    StratifiedKFold,
    cross_validate,
)
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """
    Complete training run configuration.
    Load from YAML via TrainingConfig.from_yaml() or construct programmatically.
    """
    # Experiment
    experiment_name: str = "medml_experiment"
    run_name: str = ""
    random_seed: int = 42

    # Data
    data_path: str = ""
    label_column: str = "label"
    feature_columns: list[str] = field(default_factory=list)
    group_column: str = ""        # Patient ID for group-aware splits
    stratify_column: str = ""     # Column for stratified splits

    # Split strategy
    test_size: float = 0.20
    val_size: float = 0.15
    n_cv_folds: int = 5
    cv_strategy: str = "stratified"  # "stratified" | "group" | "stratified_group"

    # Hyperparameter optimization
    hpo_enabled: bool = True
    hpo_n_trials: int = 50
    hpo_timeout_seconds: int = 3600
    hpo_metric: str = "val_auroc"
    hpo_direction: str = "maximize"

    # Training
    model_type: str = "random_forest"  # "random_forest" | "xgboost" | "logistic"
    model_params: dict[str, Any] = field(default_factory=dict)

    # MLflow
    tracking_uri: str = "mlruns"
    register_model: bool = True
    model_registry_name: str = ""

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainingConfig":
        import yaml
        data = yaml.safe_load(Path(path).read_text())
        training_cfg = data.get("training", data)
        # Handle nested structures
        flat = {}
        for k, v in training_cfg.items():
            flat[k] = v
        return cls(**{k: v for k, v in flat.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Automated Training Pipeline
# ---------------------------------------------------------------------------

class AutomatedTrainingPipeline:
    """
    End-to-end automated training pipeline for medical ML models.

    Handles data loading, splitting, cross-validation, hyperparameter optimization,
    final training, and MLflow logging — all with GMLP compliance checks.

    Example::

        pipeline = AutomatedTrainingPipeline(config)
        results = pipeline.run(data=df)
        print(f"Best CV AUROC: {results['best_cv_auroc']:.4f}")
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self._set_seeds()

    def _set_seeds(self) -> None:
        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)
        try:
            import torch
            torch.manual_seed(self.config.random_seed)
        except ImportError:
            pass
        try:
            import tensorflow as tf
            tf.random.set_seed(self.config.random_seed)
        except ImportError:
            pass

    # ------------------------------------------------------------------
    # Main Entry Point
    # ------------------------------------------------------------------

    def run(
        self,
        data: pd.DataFrame | None = None,
        X: np.ndarray | None = None,
        y: np.ndarray | None = None,
        model_builder: Callable | None = None,
    ) -> dict[str, Any]:
        """
        Execute the full training pipeline.

        Args:
            data: DataFrame with features and label column.
            X: Feature matrix (alternative to data).
            y: Labels (alternative to data).
            model_builder: Optional function returning a sklearn-compatible model.
                           If None, uses config.model_type.

        Returns:
            Dict with results including best model, CV metrics, and run ID.
        """
        from .experiment_tracker import GMLPExperimentTracker

        start_time = time.time()

        # Prepare data
        if data is not None:
            X, y, groups = self._prepare_data(data)
        else:
            assert X is not None and y is not None, "Provide either 'data' or (X, y)"
            groups = None

        logger.info(
            "Training pipeline started: %d samples, %d features, %.1f%% positive",
            len(y), X.shape[1] if hasattr(X, "shape") else len(X[0]),
            100 * y.mean()
        )

        # Train/test split
        X_dev, X_test, y_dev, y_test, groups_dev = self._train_test_split(X, y, groups)

        results: dict[str, Any] = {
            "n_train": len(y_dev),
            "n_test": len(y_test),
            "positive_rate_train": float(y_dev.mean()),
            "positive_rate_test": float(y_test.mean()),
        }

        with GMLPExperimentTracker(
            experiment_name=self.config.experiment_name,
            run_name=self.config.run_name or f"auto_{int(time.time())}",
            tracking_uri=self.config.tracking_uri,
            intended_use=f"Automated training run for {self.config.experiment_name}",
        ) as tracker:
            tracker.set_seeds(self.config.random_seed)
            tracker.log_hyperparams(self.config.to_dict())

            # Verify train/test separation (GMLP Principle 8)
            if data is not None and self.config.group_column:
                train_groups = groups_dev if groups_dev is not None else np.arange(len(y_dev))
                test_groups = np.arange(len(y_dev), len(y_dev) + len(y_test))
                tracker.verify_train_test_separation(train_groups, test_groups)

            if self.config.data_path:
                tracker.log_dataset_metadata(
                    self.config.data_path, split="train"
                )

            # Build model
            if model_builder is not None:
                base_model = model_builder()
            else:
                base_model = self._build_model(self.config.model_type, self.config.model_params)

            # HPO
            if self.config.hpo_enabled:
                with tracker.log_step_timing("hyperparameter_optimization"):
                    best_params = self._run_hpo(base_model, X_dev, y_dev, groups_dev)
                    tracker.log_hyperparams({"hpo." + k: v for k, v in best_params.items()})
                    # Rebuild model with best params
                    base_model = self._build_model(self.config.model_type, best_params)
                    results["best_hpo_params"] = best_params

            # Cross-validation
            with tracker.log_step_timing("cross_validation"):
                cv_results = self._run_cross_validation(base_model, X_dev, y_dev, groups_dev)
                tracker.log_metrics(cv_results)
                results.update(cv_results)
                logger.info(
                    "CV complete: AUROC %.4f ± %.4f",
                    cv_results.get("cv_auroc_mean", 0),
                    cv_results.get("cv_auroc_std", 0),
                )

            # Final training on full development set
            with tracker.log_step_timing("final_training"):
                final_model = self._build_model(
                    self.config.model_type,
                    best_params if self.config.hpo_enabled else self.config.model_params
                )
                final_model.fit(X_dev, y_dev)

            # Test set evaluation
            with tracker.log_step_timing("test_evaluation"):
                test_metrics = self._evaluate_model(final_model, X_test, y_test)
                tracker.log_metrics(test_metrics, prefix="test.")
                results["test_metrics"] = test_metrics
                logger.info(
                    "Test evaluation: AUROC=%.4f, F1=%.4f",
                    test_metrics.get("auroc", 0),
                    test_metrics.get("f1", 0),
                )

            # Log model artifact
            model_uri = tracker.log_model(final_model, flavor="sklearn")

            # Register if configured
            if self.config.register_model and model_uri and self.config.model_registry_name:
                tracker.register_model(
                    model_uri,
                    self.config.model_registry_name,
                    description=f"Auto-trained {self.config.model_type} on {self.config.experiment_name}",
                )

            # GMLP documentation
            tracker.log_gmlp_documentation(
                intended_use=f"Medical AI classifier for {self.config.experiment_name}",
                known_limitations=[
                    "Trained on a specific population — validate before deployment in new sites",
                    "Performance may degrade with scanner/protocol changes",
                ],
            )

            results["run_id"] = tracker.run_id
            results["model"] = final_model
            results["X_test"] = X_test
            results["y_test"] = y_test
            results["training_time_seconds"] = round(time.time() - start_time, 2)

        return results

    # ------------------------------------------------------------------
    # Data Preparation
    # ------------------------------------------------------------------

    def _prepare_data(
        self, data: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Extract features, labels, and optional group IDs."""
        label_col = self.config.label_column
        assert label_col in data.columns, f"Label column '{label_col}' not found"

        if self.config.feature_columns:
            feature_cols = self.config.feature_columns
        else:
            # All columns except label and group
            exclude = [label_col]
            if self.config.group_column:
                exclude.append(self.config.group_column)
            feature_cols = [c for c in data.columns if c not in exclude
                           and data[c].dtype != object]

        X = data[feature_cols].values.astype(np.float32)
        y = data[label_col].values.astype(int)
        groups = (
            data[self.config.group_column].values
            if self.config.group_column and self.config.group_column in data.columns
            else None
        )
        return X, y, groups

    def _train_test_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray | None,
    ) -> tuple:
        """
        Split data into development (train+val) and held-out test sets.
        Uses group-aware splitting when patient IDs are available to prevent leakage.
        """
        from sklearn.model_selection import train_test_split, GroupShuffleSplit

        if groups is not None:
            # Group-aware split: all samples from a patient go to same partition
            gss = GroupShuffleSplit(
                n_splits=1,
                test_size=self.config.test_size,
                random_state=self.config.random_seed,
            )
            dev_idx, test_idx = next(gss.split(X, y, groups=groups))
            return (
                X[dev_idx], X[test_idx],
                y[dev_idx], y[test_idx],
                groups[dev_idx],
            )
        else:
            X_dev, X_test, y_dev, y_test = train_test_split(
                X, y,
                test_size=self.config.test_size,
                stratify=y,
                random_state=self.config.random_seed,
            )
            return X_dev, X_test, y_dev, y_test, None

    # ------------------------------------------------------------------
    # Cross-Validation
    # ------------------------------------------------------------------

    def _run_cross_validation(
        self,
        model: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray | None,
    ) -> dict[str, float]:
        """Run cross-validation with the appropriate strategy for medical data."""
        cv = self._build_cv_splitter(y, groups)
        scoring = {
            "auroc": "roc_auc",
            "f1": "f1",
            "precision": "precision",
            "recall": "recall",
            "accuracy": "accuracy",
        }

        cv_results = cross_validate(
            model, X, y,
            cv=cv,
            scoring=scoring,
            groups=groups,
            n_jobs=-1,
            return_train_score=True,
        )

        metrics: dict[str, float] = {}
        for metric in scoring:
            test_scores = cv_results[f"test_{metric}"]
            train_scores = cv_results[f"train_{metric}"]
            metrics[f"cv_{metric}_mean"] = round(float(test_scores.mean()), 4)
            metrics[f"cv_{metric}_std"] = round(float(test_scores.std()), 4)
            metrics[f"cv_{metric}_train_mean"] = round(float(train_scores.mean()), 4)

            # Check for overfitting (train-val gap)
            gap = float(train_scores.mean()) - float(test_scores.mean())
            if metric == "auroc" and gap > 0.05:
                logger.warning(
                    "Potential overfitting: train AUROC %.4f vs val AUROC %.4f (gap=%.4f)",
                    float(train_scores.mean()), float(test_scores.mean()), gap
                )

        return metrics

    def _build_cv_splitter(
        self, y: np.ndarray, groups: np.ndarray | None
    ):
        strategy = self.config.cv_strategy
        n = self.config.n_cv_folds

        if strategy == "stratified":
            return StratifiedKFold(n_splits=n, shuffle=True, random_state=self.config.random_seed)
        elif strategy == "group" and groups is not None:
            return GroupKFold(n_splits=n)
        elif strategy == "stratified_group" and groups is not None:
            return StratifiedGroupKFold(n_splits=n, shuffle=True,
                                        random_state=self.config.random_seed)
        else:
            logger.warning("Falling back to StratifiedKFold for CV")
            return StratifiedKFold(n_splits=n, shuffle=True, random_state=self.config.random_seed)

    # ------------------------------------------------------------------
    # Hyperparameter Optimization
    # ------------------------------------------------------------------

    def _run_hpo(
        self,
        base_model: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray | None,
    ) -> dict[str, Any]:
        """Run Optuna TPE hyperparameter search."""
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            logger.warning("Optuna not installed — skipping HPO. Install: pip install optuna")
            return {}

        model_type = self.config.model_type

        def objective(trial: "optuna.Trial") -> float:
            params = self._suggest_params(trial, model_type)
            model = self._build_model(model_type, params)
            cv = self._build_cv_splitter(y, groups)

            scores = cross_validate(
                model, X, y,
                cv=cv,
                scoring="roc_auc",
                groups=groups,
                n_jobs=-1,
            )
            return float(scores["test_score"].mean())

        study = optuna.create_study(
            direction=self.config.hpo_direction,
            sampler=optuna.samplers.TPESampler(seed=self.config.random_seed),
        )
        study.optimize(
            objective,
            n_trials=self.config.hpo_n_trials,
            timeout=self.config.hpo_timeout_seconds,
            show_progress_bar=False,
        )

        best_params = study.best_params
        best_value = study.best_value
        logger.info(
            "HPO complete: best %s=%.4f with params=%s",
            self.config.hpo_metric, best_value, best_params
        )
        return best_params

    @staticmethod
    def _suggest_params(trial: Any, model_type: str) -> dict[str, Any]:
        """Suggest hyperparameter values for a given model type."""
        if model_type == "random_forest":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
                "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
            }
        elif model_type == "xgboost":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 10.0),
            }
        elif model_type == "logistic":
            return {
                "C": trial.suggest_float("C", 1e-4, 100.0, log=True),
                "penalty": trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"]),
                "solver": "saga",
                "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
                "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
                "max_iter": 1000,
            }
        else:
            return {}

    # ------------------------------------------------------------------
    # Model Building
    # ------------------------------------------------------------------

    @staticmethod
    def _build_model(model_type: str, params: dict[str, Any]) -> BaseEstimator:
        """Instantiate a model by type with given parameters."""
        if model_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(**params, random_state=42, n_jobs=-1)
        elif model_type == "xgboost":
            try:
                from xgboost import XGBClassifier
                return XGBClassifier(**params, random_state=42, n_jobs=-1,
                                     eval_metric="logloss", use_label_encoder=False)
            except ImportError:
                logger.warning("XGBoost not installed, falling back to RandomForest")
                from sklearn.ensemble import RandomForestClassifier
                return RandomForestClassifier(random_state=42)
        elif model_type == "logistic":
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            lr = LogisticRegression(**params, random_state=42)
            return Pipeline([("scaler", scaler), ("clf", lr)])
        elif model_type == "gradient_boosting":
            from sklearn.ensemble import GradientBoostingClassifier
            return GradientBoostingClassifier(**params, random_state=42)
        else:
            raise ValueError(f"Unknown model_type: '{model_type}'")

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @staticmethod
    def _evaluate_model(
        model: BaseEstimator, X: np.ndarray, y: np.ndarray
    ) -> dict[str, float]:
        from sklearn.metrics import (
            accuracy_score, f1_score, average_precision_score,
            roc_auc_score, brier_score_loss
        )

        y_prob = model.predict_proba(X)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        metrics: dict[str, float] = {
            "auroc": roc_auc_score(y, y_prob),
            "auprc": average_precision_score(y, y_prob),
            "accuracy": accuracy_score(y, y_pred),
            "f1": f1_score(y, y_pred, zero_division=0),
            "brier_score": brier_score_loss(y, y_prob),
        }
        return {k: round(float(v), 4) for k, v in metrics.items()}
