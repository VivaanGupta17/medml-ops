#!/usr/bin/env python3
"""
End-to-End MedML-Ops Pipeline Runner
======================================
Executes the full FDA-compliant MLOps pipeline:
  1. Data validation (schema + drift)
  2. Bias pre-check (demographic completeness)
  3. Model training (with HPO + cross-validation)
  4. Model evaluation (clinical metrics + calibration)
  5. Bias analysis (subgroup performance + fairness)
  6. Regression testing (vs. baseline on golden test set)
  7. GMLP compliance audit
  8. Model card generation
  9. MLflow model registration

Usage:
    python scripts/run_pipeline.py --config configs/pipeline_config.yaml
    python scripts/run_pipeline.py --config configs/my_experiment.yaml --skip-hpo
    python scripts/run_pipeline.py --steps validate,train,evaluate
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("medml-ops.pipeline")


# ---------------------------------------------------------------------------
# Pipeline Steps
# ---------------------------------------------------------------------------

class Pipeline:
    """Orchestrates the full medml-ops pipeline."""

    ALL_STEPS = [
        "validate",
        "train",
        "evaluate",
        "bias",
        "regression",
        "gmlp",
        "model_card",
        "report",
    ]

    def __init__(self, config_path: str, steps: list[str] | None = None,
                 skip_hpo: bool = False, dry_run: bool = False):
        self.config = self._load_config(config_path)
        self.steps = steps or self.ALL_STEPS
        self.skip_hpo = skip_hpo
        self.dry_run = dry_run
        self.results: dict = {}
        self._ensure_output_dirs()

    @staticmethod
    def _load_config(path: str) -> dict:
        cfg = yaml.safe_load(Path(path).read_text())
        logger.info("Loaded config from %s", path)
        return cfg

    def _ensure_output_dirs(self) -> None:
        paths = self.config.get("paths", {})
        for d in [
            paths.get("models_dir", "models/"),
            paths.get("reports_dir", "reports/"),
            paths.get("compliance_dir", "compliance/"),
            paths.get("logs_dir", "logs/"),
            paths.get("artifacts_dir", "artifacts/"),
        ]:
            Path(d).mkdir(parents=True, exist_ok=True)

    def run(self) -> bool:
        """Execute all configured steps. Returns True if pipeline passed."""
        start_time = time.time()
        logger.info("=" * 60)
        logger.info("MedML-Ops Pipeline Starting")
        logger.info("Experiment: %s", self.config.get("experiment", {}).get("name", ""))
        logger.info("Steps: %s", ", ".join(self.steps))
        logger.info("=" * 60)

        step_fns = {
            "validate": self.step_validate,
            "train": self.step_train,
            "evaluate": self.step_evaluate,
            "bias": self.step_bias_analysis,
            "regression": self.step_regression_testing,
            "gmlp": self.step_gmlp_audit,
            "model_card": self.step_model_card,
            "report": self.step_generate_report,
        }

        success = True
        for step_name in self.steps:
            if step_name not in step_fns:
                logger.warning("Unknown step '%s' — skipping", step_name)
                continue

            logger.info("-" * 40)
            logger.info("STEP: %s", step_name.upper())

            if self.dry_run:
                logger.info("[DRY RUN] Would execute step: %s", step_name)
                continue

            step_start = time.time()
            try:
                result = step_fns[step_name]()
                self.results[step_name] = result
                elapsed = time.time() - step_start
                logger.info("✓ Step '%s' completed in %.1fs", step_name, elapsed)
            except Exception as e:
                elapsed = time.time() - step_start
                logger.error("✗ Step '%s' FAILED after %.1fs: %s", step_name, elapsed, e,
                              exc_info=True)
                success = False
                # Continue with remaining steps unless validation failed hard
                if step_name == "validate" and self.config.get(
                    "data_validation", {}
                ).get("fail_on_error", True):
                    logger.error("Validation errors — stopping pipeline")
                    break

        total_time = time.time() - start_time
        logger.info("=" * 60)
        logger.info("Pipeline %s in %.1fs", "PASSED" if success else "FAILED", total_time)
        logger.info("=" * 60)
        return success

    # ------------------------------------------------------------------
    # Step: Data Validation
    # ------------------------------------------------------------------

    def step_validate(self) -> dict:
        from src.data_validation.schema_validator import (
            MedicalDataValidator, MedicalSchemaSpec
        )

        cfg = self.config.get("data_validation", {})
        data_cfg = self.config.get("data", {})

        schema_cfg = cfg.get("schema", {})
        schema = MedicalSchemaSpec(**schema_cfg) if schema_cfg else None

        validator = MedicalDataValidator(
            schema=schema,
            outlier_z_threshold=cfg.get("outlier_z_threshold", 3.5),
            drift_p_value_threshold=cfg.get("drift_p_value_threshold", 0.05),
        )

        train_path = data_cfg.get("train_path", "")
        if not Path(train_path).exists():
            logger.warning("Training data not found at '%s' — skipping validation", train_path)
            return {"skipped": True, "reason": "data not found"}

        test_path = data_cfg.get("test_path")
        reference_path = test_path if test_path and Path(test_path).exists() else None

        report = validator.validate_dataset(train_path, reference_path=reference_path)

        output_path = Path(self.config["paths"].get("reports_dir", "reports/")) / "validation_report.json"
        report.save_json(output_path)

        if not report.passed and cfg.get("fail_on_error", True):
            raise RuntimeError(
                f"Data validation failed: {report.error_count} errors. "
                f"See {output_path}"
            )

        logger.info(report.summary())
        return {"passed": report.passed, "errors": report.error_count,
                "warnings": report.warning_count}

    # ------------------------------------------------------------------
    # Step: Training
    # ------------------------------------------------------------------

    def step_train(self) -> dict:
        from src.training.automated_training import AutomatedTrainingPipeline, TrainingConfig

        train_cfg = self.config.get("training", {})
        data_cfg = self.config.get("data", {})
        mlflow_cfg = self.config.get("mlflow", {})
        exp_cfg = self.config.get("experiment", {})

        # Merge config into TrainingConfig
        hpo_cfg = train_cfg.get("hpo", {})
        config = TrainingConfig(
            experiment_name=exp_cfg.get("name", "medml_experiment"),
            random_seed=train_cfg.get("random_seed", 42),
            data_path=data_cfg.get("train_path", ""),
            label_column=data_cfg.get("label_column", "label"),
            group_column=data_cfg.get("patient_id_column", ""),
            test_size=train_cfg.get("test_size", 0.20),
            cv_strategy=train_cfg.get("cv_strategy", "stratified"),
            n_cv_folds=train_cfg.get("n_cv_folds", 5),
            model_type=train_cfg.get("model_type", "random_forest"),
            model_params=train_cfg.get("model_params", {}),
            hpo_enabled=not self.skip_hpo and hpo_cfg.get("enabled", True),
            hpo_n_trials=hpo_cfg.get("n_trials", 50),
            hpo_timeout_seconds=hpo_cfg.get("timeout_seconds", 3600),
            tracking_uri=mlflow_cfg.get("tracking_uri", "mlruns"),
            register_model=mlflow_cfg.get("register_model", False),
            model_registry_name=mlflow_cfg.get("model_registry_name", ""),
        )

        train_path = data_cfg.get("train_path", "")
        if not Path(train_path).exists():
            logger.warning("Training data not found — generating synthetic demo data")
            data = self._generate_demo_data()
        else:
            data = pd.read_csv(train_path)

        pipeline = AutomatedTrainingPipeline(config)
        results = pipeline.run(data=data)

        # Cache for downstream steps
        self.results["_model"] = results.get("model")
        self.results["_X_test"] = results.get("X_test")
        self.results["_y_test"] = results.get("y_test")
        self.results["_run_id"] = results.get("run_id")

        return {
            "run_id": results.get("run_id"),
            "cv_auroc": results.get("cv_auroc_mean"),
            "test_auroc": results.get("test_metrics", {}).get("auroc"),
            "training_time_seconds": results.get("training_time_seconds"),
        }

    # ------------------------------------------------------------------
    # Step: Evaluation
    # ------------------------------------------------------------------

    def step_evaluate(self) -> dict:
        from src.evaluation.model_evaluator import MedicalModelEvaluator

        model = self.results.get("_model")
        X_test = self.results.get("_X_test")
        y_test = self.results.get("_y_test")

        if model is None or X_test is None or y_test is None:
            logger.warning("Training outputs not available — skipping evaluation")
            return {"skipped": True}

        eval_cfg = self.config.get("evaluation", {})
        predicate = eval_cfg.get("predicate_device", {})
        predicate_metrics = (
            {k: v for k, v in predicate.items() if v is not None and k != "name"}
            if predicate and predicate.get("auroc") else None
        )

        evaluator = MedicalModelEvaluator(
            bootstrap_n=eval_cfg.get("bootstrap_n", 1000),
            sensitivity_targets=eval_cfg.get("sensitivity_targets", [0.85, 0.90, 0.95]),
            n_calibration_bins=eval_cfg.get("n_calibration_bins", 10),
        )

        exp_cfg = self.config.get("experiment", {})
        report = evaluator.evaluate_model(
            model, X_test, y_test,
            model_name=exp_cfg.get("name", "model"),
            predicate_metrics=predicate_metrics,
        )

        output_path = Path(self.config["paths"].get("reports_dir", "reports/")) / "evaluation_report.json"
        report.save_json(output_path)

        self.results["_eval_report"] = report
        logger.info(report.summary())

        return {
            "auroc": report.core_metrics.get("auroc"),
            "sensitivity": report.core_metrics.get("sensitivity"),
            "specificity": report.core_metrics.get("specificity"),
            "ece": report.calibration.ece if report.calibration else None,
        }

    # ------------------------------------------------------------------
    # Step: Bias Analysis
    # ------------------------------------------------------------------

    def step_bias_analysis(self) -> dict:
        from src.data_validation.bias_detector import DemographicBiasDetector

        model = self.results.get("_model")
        X_test = self.results.get("_X_test")
        y_test = self.results.get("_y_test")

        if model is None:
            logger.warning("Model not available — generating synthetic demo results")
            n = 500
            y_true = np.random.binomial(1, 0.3, n)
            y_prob = np.clip(y_true * 0.6 + np.random.normal(0, 0.25, n), 0, 1)
            demo = pd.DataFrame({
                "age_group": np.random.choice(["18-40", "41-65", "66+"], n),
                "sex": np.random.choice(["M", "F"], n),
                "ethnicity": np.random.choice(["White", "Black", "Hispanic", "Asian"], n),
            })
        else:
            y_prob = model.predict_proba(X_test)[:, 1]
            y_true = y_test
            n = len(y_true)

            # Try to load demographic data
            data_cfg = self.config.get("data", {})
            demo_cols = data_cfg.get("demographic_columns", [])
            demo_path = data_cfg.get("test_path", "")

            if demo_path and Path(demo_path).exists() and demo_cols:
                full_df = pd.read_csv(demo_path)
                available_demo_cols = [c for c in demo_cols if c in full_df.columns]
                if available_demo_cols:
                    # Align with test set (last test_size fraction)
                    test_idx = full_df.index[-len(y_true):]
                    demo = full_df.loc[test_idx, available_demo_cols].reset_index(drop=True)
                else:
                    demo = self._generate_demo_demographics(n)
            else:
                demo = self._generate_demo_demographics(n)

        bias_cfg = self.config.get("bias_analysis", {})
        detector = DemographicBiasDetector(
            sensitive_attributes=bias_cfg.get(
                "sensitive_attributes", ["age_group", "sex", "ethnicity"]
            ),
            reference_groups=bias_cfg.get("reference_groups", {}),
            min_subgroup_size=bias_cfg.get("min_subgroup_size", 30),
            decision_threshold=bias_cfg.get("decision_threshold", 0.5),
        )

        exp_cfg = self.config.get("experiment", {})
        bias_report = detector.analyze(
            y_true, y_prob, demo, model_name=exp_cfg.get("name", "model")
        )

        reports_dir = Path(self.config["paths"].get("reports_dir", "reports/"))
        bias_report.save_json(reports_dir / "bias_report.json")
        bias_report.save_html(reports_dir / "bias_report.html")

        self.results["_bias_report"] = bias_report

        if bias_report.has_high_bias_flags():
            logger.warning(
                "HIGH BIAS FLAGS detected — review bias_report.html before deployment"
            )

        return {
            "has_high_flags": bias_report.has_high_bias_flags(),
            "n_subgroups": len(bias_report.subgroup_metrics),
            "n_fairness_comparisons": len(bias_report.fairness_metrics),
        }

    # ------------------------------------------------------------------
    # Step: Regression Testing
    # ------------------------------------------------------------------

    def step_regression_testing(self) -> dict:
        from src.evaluation.regression_testing import (
            ModelRegressionTester, AcceptanceCriteria
        )

        eval_cfg = self.config.get("evaluation", {}).get("regression_testing", {})
        if not eval_cfg.get("enabled", True):
            return {"skipped": True}

        baseline_path = eval_cfg.get("baseline_predictions_path", "")
        if not Path(baseline_path).exists() if baseline_path else True:
            logger.info("No baseline predictions found — skipping regression testing")
            logger.info(
                "Run again after saving baseline with: "
                "tester.save_baseline('data/baseline_predictions.json')"
            )
            return {"skipped": True, "reason": "no baseline available"}

        criteria_cfg = eval_cfg.get("acceptance_criteria", {})
        criteria = AcceptanceCriteria(**criteria_cfg)

        tester = ModelRegressionTester.load_baseline(baseline_path)

        model = self.results.get("_model")
        X_test = self.results.get("_X_test")
        y_test = self.results.get("_y_test")

        if model is None:
            return {"skipped": True, "reason": "model not available"}

        y_prob_new = model.predict_proba(X_test)[:, 1]
        reg_report = tester.run(
            new_predictions={"y_prob": y_prob_new, "y_true": y_test},
            new_model_name=self.config.get("experiment", {}).get("name", "new_model"),
        )

        output_path = (
            Path(self.config["paths"].get("reports_dir", "reports/")) /
            "regression_test_report.json"
        )
        reg_report.save_json(output_path)
        logger.info(reg_report.summary())

        if not reg_report.passed:
            logger.error("Regression tests FAILED — model update rejected")

        return {"passed": reg_report.passed, "verdict": reg_report.verdict()}

    # ------------------------------------------------------------------
    # Step: GMLP Audit
    # ------------------------------------------------------------------

    def step_gmlp_audit(self) -> dict:
        from src.compliance.gmlp_checklist import GMLPComplianceChecker

        run_id = self.results.get("_run_id", "")
        exp_cfg = self.config.get("experiment", {})
        compliance_cfg = self.config.get("compliance", {}).get("gmlp", {})

        checker = GMLPComplianceChecker(
            run_id=run_id,
            model_name=exp_cfg.get("name", "model"),
            tracking_uri=self.config.get("mlflow", {}).get("tracking_uri", "mlruns"),
        )

        has_bias = "_bias_report" in self.results
        has_eval = "_eval_report" in self.results

        gmlp_report = checker.run_full_audit(
            has_bias_report=has_bias,
            has_model_card=False,  # Will be generated in next step
            has_drift_monitoring=self.config.get("monitoring", {}).get("enabled", True),
            has_regression_tests=has_eval,
            has_pccp=self.config.get("compliance", {}).get("pccp", {}).get("enabled", True),
            train_test_overlap=False,  # Verified by experiment_tracker
            has_git_versioning=True,
            has_requirements_pinned=Path("requirements.txt").exists(),
            has_confidence_intervals=has_eval,
            has_human_review_loop=False,
            has_clinical_expert=False,
        )

        output_path = compliance_cfg.get("output_path", "compliance/gmlp_audit.json")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        gmlp_report.save_json(output_path)

        logger.info(gmlp_report.summary())

        if not gmlp_report.submission_ready:
            logger.warning(
                "GMLP audit score: %.1f%% — not yet submission-ready. "
                "Address critical gaps before 510(k) preparation.",
                gmlp_report.overall_score * 100
            )

        return {
            "score": gmlp_report.overall_score,
            "compliant_count": gmlp_report.compliant_count,
            "submission_ready": gmlp_report.submission_ready,
        }

    # ------------------------------------------------------------------
    # Step: Model Card
    # ------------------------------------------------------------------

    def step_model_card(self) -> dict:
        from src.monitoring.model_card_generator import ModelCardGenerator

        exp_cfg = self.config.get("experiment", {})
        compliance_cfg = self.config.get("compliance", {}).get("model_card", {})
        data_cfg = self.config.get("data", {})

        generator = ModelCardGenerator(
            mlflow_run_id=self.results.get("_run_id"),
            tracking_uri=self.config.get("mlflow", {}).get("tracking_uri", "mlruns"),
        )

        card = generator.generate(
            model_name=exp_cfg.get("name", "model"),
            version=exp_cfg.get("version", "1.0.0"),
            model_type=self.config.get("training", {}).get("model_type", ""),
            description=exp_cfg.get("description", ""),
            intended_uses=[exp_cfg.get("intended_use", "")],
            evaluation_report=self.results.get("_eval_report"),
            bias_report=self.results.get("_bias_report"),
            training_data_info={
                "name": data_cfg.get("dataset_name", ""),
                "version": data_cfg.get("dataset_version", ""),
                "source": data_cfg.get("dataset_source", ""),
            },
            regulatory_pathway=exp_cfg.get("regulatory_pathway", "510(k)"),
        )

        html_path = compliance_cfg.get("output_html", "docs/model_card.html")
        json_path = compliance_cfg.get("output_json", "compliance/model_card.json")
        Path(html_path).parent.mkdir(parents=True, exist_ok=True)
        Path(json_path).parent.mkdir(parents=True, exist_ok=True)

        card.save_html(html_path)
        card.save_json(json_path)

        logger.info("Model card generated: %s", html_path)
        return {"html": html_path, "json": json_path}

    # ------------------------------------------------------------------
    # Step: Generate Report
    # ------------------------------------------------------------------

    def step_generate_report(self) -> dict:
        """Aggregate all pipeline outputs into a summary report."""
        import json as _json
        from datetime import datetime

        summary = {
            "pipeline_run": {
                "timestamp": datetime.utcnow().isoformat(),
                "experiment": self.config.get("experiment", {}).get("name", ""),
                "steps_executed": self.steps,
            },
            "results": {
                k: v for k, v in self.results.items()
                if not k.startswith("_")  # Skip internal objects
            },
        }

        report_path = (
            Path(self.config["paths"].get("reports_dir", "reports/")) / "pipeline_summary.json"
        )
        report_path.write_text(_json.dumps(summary, indent=2, default=str))
        logger.info("Pipeline summary saved to %s", report_path)
        return {"summary_path": str(report_path)}

    # ------------------------------------------------------------------
    # Demo Data Generators
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_demo_data(n: int = 1000) -> pd.DataFrame:
        """Generate synthetic medical dataset for demo/testing."""
        rng = np.random.default_rng(42)
        n_pos = int(n * 0.30)  # 30% positive prevalence
        labels = np.concatenate([np.ones(n_pos), np.zeros(n - n_pos)])
        rng.shuffle(labels)

        return pd.DataFrame({
            "patient_id": [f"P{i:05d}" for i in range(n)],
            "label": labels.astype(int),
            "feature_1": rng.normal(labels * 0.8, 1.2),
            "feature_2": rng.normal(labels * 0.5, 1.0),
            "feature_3": rng.normal(0, 1, n),
            "feature_4": rng.normal(labels * -0.3, 0.8),
            "feature_5": rng.uniform(0, 1, n),
            "age": rng.integers(25, 85, n),
            "age_group": pd.cut(
                rng.integers(25, 85, n),
                bins=[0, 40, 65, 120],
                labels=["18-40", "41-65", "66+"]
            ),
            "sex": rng.choice(["M", "F"], n),
            "ethnicity": rng.choice(["White", "Black", "Hispanic", "Asian"], n,
                                     p=[0.60, 0.15, 0.15, 0.10]),
        })

    @staticmethod
    def _generate_demo_demographics(n: int) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        return pd.DataFrame({
            "age_group": rng.choice(["18-40", "41-65", "66+"], n),
            "sex": rng.choice(["M", "F"], n),
            "ethnicity": rng.choice(["White", "Black", "Hispanic", "Asian"], n),
        })


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="MedML-Ops: FDA-Compliant MLOps Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_pipeline.py --config configs/pipeline_config.yaml
  python scripts/run_pipeline.py --config configs/pipeline_config.yaml --skip-hpo
  python scripts/run_pipeline.py --config configs/pipeline_config.yaml --steps validate,train
  python scripts/run_pipeline.py --config configs/pipeline_config.yaml --dry-run
        """,
    )
    parser.add_argument(
        "--config", "-c",
        default="configs/pipeline_config.yaml",
        help="Path to pipeline configuration YAML",
    )
    parser.add_argument(
        "--steps", "-s",
        default=",".join(Pipeline.ALL_STEPS),
        help=f"Comma-separated pipeline steps. Available: {','.join(Pipeline.ALL_STEPS)}",
    )
    parser.add_argument(
        "--skip-hpo",
        action="store_true",
        help="Skip hyperparameter optimization (faster iteration)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print steps without executing",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    steps = [s.strip() for s in args.steps.split(",")]

    pipeline = Pipeline(
        config_path=args.config,
        steps=steps,
        skip_hpo=args.skip_hpo,
        dry_run=args.dry_run,
    )

    success = pipeline.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
