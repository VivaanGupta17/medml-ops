"""
Model Regression Testing
=========================
Automated regression test suite ensuring model updates don't degrade performance
on a golden test set. PCCP-aligned change validation with slice-based testing.

Key capabilities:
  - Compare new model vs. baseline on a locked golden test set
  - Performance degradation alerts with configurable thresholds
  - Slice-based regression (per demographic subgroup)
  - Critical failure detection (never-miss cases)
  - PCCP-aligned change categorization

FDA GMLP Alignment:
  - Principle 5: Re-training practices — validates changes before deployment
  - Principle 7: Testing demonstrates performance maintained after updates
  - PCCP Framework: Pre-defined acceptance criteria for model changes
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class RegressionTestResult:
    """Result of a single regression test."""
    test_name: str
    metric: str
    baseline_value: float
    new_value: float
    delta: float
    threshold: float
    passed: bool
    severity: str       # "critical" | "error" | "warning" | "info"
    message: str
    slice_name: str = "overall"

    def to_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in self.__dict__.items()}


@dataclass
class RegressionTestReport:
    """Complete regression test report for a model update."""
    baseline_model_name: str
    new_model_name: str
    golden_test_set: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    test_results: list[RegressionTestResult] = field(default_factory=list)
    pccp_change_category: str = ""   # "modification_1", "retraining", "architecture_change"
    overall_verdict: str = ""        # "APPROVED" | "REJECTED" | "NEEDS_REVIEW"
    summary_stats: dict[str, Any] = field(default_factory=dict)
    gmlp_alignment: str = "Principles 5, 7 + PCCP Framework"

    @property
    def passed(self) -> bool:
        return not any(
            not r.passed and r.severity in ("critical", "error")
            for r in self.test_results
        )

    @property
    def critical_failures(self) -> list[RegressionTestResult]:
        return [r for r in self.test_results if not r.passed and r.severity == "critical"]

    @property
    def errors(self) -> list[RegressionTestResult]:
        return [r for r in self.test_results if not r.passed and r.severity == "error"]

    @property
    def warnings(self) -> list[RegressionTestResult]:
        return [r for r in self.test_results if not r.passed and r.severity == "warning"]

    def verdict(self) -> str:
        if self.critical_failures:
            return "REJECTED — Critical regression detected"
        elif self.errors:
            return "NEEDS_REVIEW — Performance degradation exceeds thresholds"
        elif self.warnings:
            return "APPROVED_WITH_WARNINGS — Minor degradation, document and monitor"
        else:
            return "APPROVED — All regression tests passed"

    def summary(self) -> str:
        v = self.verdict()
        lines = [
            f"=== Regression Test Report ===",
            f"Baseline   : {self.baseline_model_name}",
            f"Candidate  : {self.new_model_name}",
            f"Test Set   : {self.golden_test_set}",
            f"Timestamp  : {self.timestamp}",
            f"Verdict    : {v}",
            f"",
            f"Results    : {len(self.test_results)} tests",
            f"  Critical : {len(self.critical_failures)}",
            f"  Errors   : {len(self.errors)}",
            f"  Warnings : {len(self.warnings)}",
            "",
        ]
        for r in sorted(self.test_results, key=lambda x: (x.passed, x.severity)):
            icon = "✓" if r.passed else ("✗" if r.severity in ("error", "critical") else "⚠")
            lines.append(
                f"  {icon} [{r.severity.upper():8s}] [{r.slice_name:15s}] "
                f"{r.metric}: {r.baseline_value:.4f} → {r.new_value:.4f} "
                f"(Δ={r.delta:+.4f}, threshold={r.threshold:+.4f})"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "baseline_model": self.baseline_model_name,
            "new_model": self.new_model_name,
            "golden_test_set": self.golden_test_set,
            "timestamp": self.timestamp,
            "verdict": self.verdict(),
            "pccp_change_category": self.pccp_change_category,
            "passed": self.passed,
            "critical_failures": len(self.critical_failures),
            "errors": len(self.errors),
            "warnings": len(self.warnings),
            "gmlp_alignment": self.gmlp_alignment,
            "test_results": [r.to_dict() for r in self.test_results],
            "summary_stats": self.summary_stats,
        }

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))
        logger.info("Regression test report saved to %s", path)


# ---------------------------------------------------------------------------
# Acceptance Criteria
# ---------------------------------------------------------------------------

@dataclass
class AcceptanceCriteria:
    """
    Configurable thresholds for regression test pass/fail decisions.

    PCCP pre-defined thresholds: Changes within these bounds are pre-approved
    and do not require a new 510(k) submission.
    """
    # Maximum allowed absolute degradation
    auroc_degradation_error: float = 0.02     # >2% AUROC drop → error
    auroc_degradation_critical: float = 0.05  # >5% AUROC drop → critical
    f1_degradation_error: float = 0.03
    sensitivity_degradation_error: float = 0.03
    specificity_degradation_error: float = 0.03
    auprc_degradation_error: float = 0.03

    # Slice-based thresholds (demographic subgroups)
    slice_auroc_degradation_error: float = 0.05
    slice_sensitivity_degradation_error: float = 0.05

    # Absolute minimum performance floor
    auroc_minimum: float = 0.75
    sensitivity_minimum: float = 0.80
    specificity_minimum: float = 0.70

    # Critical case regression: must not flag new false negatives on critical cases
    critical_fn_new_max: int = 0    # Zero new false negatives on critical cases

    @classmethod
    def from_dict(cls, d: dict) -> "AcceptanceCriteria":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Core Regression Tester
# ---------------------------------------------------------------------------

class ModelRegressionTester:
    """
    PCCP-aligned regression test suite for model updates.

    Compares a new model candidate against a locked baseline on a golden test set.
    Pre-defined acceptance criteria determine whether changes are auto-approved
    (within PCCP) or require human review / new regulatory submission.

    Example::

        tester = ModelRegressionTester(
            baseline_predictions={"y_prob": baseline_proba, "y_true": y_test},
            golden_test_set_name="chest_xray_golden_v1",
            criteria=AcceptanceCriteria(auroc_degradation_error=0.02),
        )
        report = tester.run(
            new_predictions={"y_prob": new_proba, "y_true": y_test},
            new_model_name="chest_xray_v3",
            demographics=demo_df,
        )
        print(report.summary())
    """

    def __init__(
        self,
        baseline_predictions: dict[str, np.ndarray],
        golden_test_set_name: str,
        baseline_model_name: str = "baseline",
        criteria: AcceptanceCriteria | None = None,
        sensitive_attributes: list[str] | None = None,
        decision_threshold: float = 0.5,
    ):
        """
        Args:
            baseline_predictions: Dict with "y_prob" and "y_true" arrays.
            golden_test_set_name: Identifier for the locked test set.
            baseline_model_name: Name of the baseline model.
            criteria: Acceptance criteria thresholds.
            sensitive_attributes: Demographic columns for slice-based testing.
            decision_threshold: Classification threshold.
        """
        self.baseline_y_true = np.asarray(baseline_predictions["y_true"])
        self.baseline_y_prob = np.asarray(baseline_predictions["y_prob"])
        self.golden_test_set_name = golden_test_set_name
        self.baseline_model_name = baseline_model_name
        self.criteria = criteria or AcceptanceCriteria()
        self.sensitive_attributes = sensitive_attributes or []
        self.decision_threshold = decision_threshold

        # Pre-compute baseline metrics
        self._baseline_metrics = self._compute_metrics(
            self.baseline_y_true, self.baseline_y_prob
        )

    # ------------------------------------------------------------------
    # Main Run
    # ------------------------------------------------------------------

    def run(
        self,
        new_predictions: dict[str, np.ndarray],
        new_model_name: str = "candidate",
        demographics: pd.DataFrame | None = None,
        critical_case_mask: np.ndarray | None = None,
        pccp_change_category: str = "",
    ) -> RegressionTestReport:
        """
        Run full regression test suite.

        Args:
            new_predictions: Dict with "y_prob" array (and optionally "y_true").
            new_model_name: Name for the candidate model.
            demographics: DataFrame with demographic columns for slice testing.
            critical_case_mask: Boolean array flagging never-miss critical cases.
            pccp_change_category: PCCP change category (for documentation).

        Returns:
            RegressionTestReport with verdict and all test results.
        """
        y_true = np.asarray(
            new_predictions.get("y_true", self.baseline_y_true)
        )
        y_prob = np.asarray(new_predictions["y_prob"])

        report = RegressionTestReport(
            baseline_model_name=self.baseline_model_name,
            new_model_name=new_model_name,
            golden_test_set=self.golden_test_set_name,
            pccp_change_category=pccp_change_category,
        )

        # Compute new model metrics
        new_metrics = self._compute_metrics(y_true, y_prob)

        # Overall regression tests
        report.test_results.extend(
            self._run_overall_regression(new_metrics, "overall")
        )

        # Minimum floor tests (absolute performance requirements)
        report.test_results.extend(
            self._run_floor_tests(new_metrics, "overall")
        )

        # Critical case regression
        if critical_case_mask is not None:
            report.test_results.extend(
                self._run_critical_case_tests(
                    y_true, y_prob, critical_case_mask
                )
            )

        # Slice-based regression (demographic subgroups)
        if demographics is not None:
            for attr in self.sensitive_attributes:
                if attr not in demographics.columns:
                    continue
                report.test_results.extend(
                    self._run_slice_regression(y_true, y_prob, demographics, attr)
                )

        # Statistical significance of regression
        report.test_results.extend(
            self._run_statistical_regression_test(y_true, y_prob)
        )

        # Summary stats
        report.summary_stats = {
            "baseline_metrics": {k: round(v, 4) for k, v in self._baseline_metrics.items()},
            "new_metrics": {k: round(v, 4) for k, v in new_metrics.items()},
            "n_test_samples": len(y_true),
            "n_positive": int(y_true.sum()),
        }

        logger.info(
            "Regression test verdict: %s (%d critical, %d errors, %d warnings)",
            report.verdict(),
            len(report.critical_failures),
            len(report.errors),
            len(report.warnings),
        )
        return report

    # ------------------------------------------------------------------
    # Regression Tests
    # ------------------------------------------------------------------

    def _run_overall_regression(
        self, new_metrics: dict[str, float], slice_name: str
    ) -> list[RegressionTestResult]:
        results = []
        test_configs = [
            ("auroc", "roc_auc",
             self.criteria.auroc_degradation_error, self.criteria.auroc_degradation_critical),
            ("f1", "f1_score", self.criteria.f1_degradation_error, None),
            ("sensitivity", "sensitivity", self.criteria.sensitivity_degradation_error, None),
            ("specificity", "specificity", self.criteria.specificity_degradation_error, None),
            ("auprc", "auprc", self.criteria.auprc_degradation_error, None),
        ]

        for metric, test_name, error_threshold, critical_threshold in test_configs:
            baseline = self._baseline_metrics.get(metric)
            new = new_metrics.get(metric)
            if baseline is None or new is None:
                continue

            delta = new - baseline

            # Determine severity
            if critical_threshold and delta < -critical_threshold:
                severity = "critical"
                passed = False
            elif delta < -error_threshold:
                severity = "error"
                passed = False
            elif delta < -error_threshold * 0.5:
                severity = "warning"
                passed = False
            else:
                severity = "info"
                passed = True

            results.append(RegressionTestResult(
                test_name=f"regression_{test_name}_{slice_name}",
                metric=metric,
                baseline_value=round(float(baseline), 4),
                new_value=round(float(new), 4),
                delta=round(float(delta), 4),
                threshold=-error_threshold,
                passed=passed,
                severity=severity,
                message=(f"{metric}: {baseline:.4f} → {new:.4f} (Δ={delta:+.4f}). "
                         f"Threshold: -{error_threshold:.4f}"),
                slice_name=slice_name,
            ))

        return results

    def _run_floor_tests(
        self, new_metrics: dict[str, float], slice_name: str
    ) -> list[RegressionTestResult]:
        """Test absolute minimum performance floors."""
        results = []
        floor_tests = [
            ("auroc", self.criteria.auroc_minimum),
            ("sensitivity", self.criteria.sensitivity_minimum),
            ("specificity", self.criteria.specificity_minimum),
        ]

        for metric, floor in floor_tests:
            val = new_metrics.get(metric)
            if val is None:
                continue
            passed = val >= floor
            results.append(RegressionTestResult(
                test_name=f"floor_{metric}_{slice_name}",
                metric=metric,
                baseline_value=floor,
                new_value=round(float(val), 4),
                delta=round(float(val) - float(floor), 4),
                threshold=0.0,
                passed=passed,
                severity="critical" if not passed else "info",
                message=f"{metric} = {val:.4f} (floor: {floor:.4f})",
                slice_name=slice_name,
            ))

        return results

    def _run_critical_case_tests(
        self,
        y_true: np.ndarray,
        y_prob_new: np.ndarray,
        critical_mask: np.ndarray,
    ) -> list[RegressionTestResult]:
        """
        Ensure new model doesn't introduce false negatives on critical cases
        that were correctly identified by baseline. Zero-tolerance.
        """
        y_pred_baseline = (self.baseline_y_prob >= self.decision_threshold).astype(int)
        y_pred_new = (y_prob_new >= self.decision_threshold).astype(int)

        # Cases that are: critical AND true positive AND baseline caught them
        critical_pos = critical_mask & (y_true == 1)
        baseline_tp = (y_pred_baseline == 1) & critical_pos
        new_fn_on_baseline_tp = baseline_tp & (y_pred_new == 0)

        n_new_critical_fn = int(new_fn_on_baseline_tp.sum())
        passed = n_new_critical_fn <= self.criteria.critical_fn_new_max

        return [RegressionTestResult(
            test_name="critical_case_false_negative_regression",
            metric="new_critical_false_negatives",
            baseline_value=0.0,
            new_value=float(n_new_critical_fn),
            delta=float(n_new_critical_fn),
            threshold=float(self.criteria.critical_fn_new_max),
            passed=passed,
            severity="critical" if not passed else "info",
            message=(f"NEW model introduces {n_new_critical_fn} false negatives on "
                     f"critical cases that baseline model caught correctly. "
                     f"Maximum allowed: {self.criteria.critical_fn_new_max}"),
            slice_name="critical_cases",
        )]

    def _run_slice_regression(
        self,
        y_true: np.ndarray,
        y_prob_new: np.ndarray,
        demographics: pd.DataFrame,
        attribute: str,
    ) -> list[RegressionTestResult]:
        """Per-demographic-subgroup regression testing."""
        results = []

        for group in demographics[attribute].dropna().unique():
            mask = (demographics[attribute].values == group)
            if mask.sum() < 30:
                continue

            baseline_metrics = self._compute_metrics(
                self.baseline_y_true[mask], self.baseline_y_prob[mask]
            )
            new_metrics = self._compute_metrics(y_true[mask], y_prob_new[mask])

            slice_name = f"{attribute}={group}"

            for metric, threshold in [
                ("auroc", self.criteria.slice_auroc_degradation_error),
                ("sensitivity", self.criteria.slice_sensitivity_degradation_error),
            ]:
                baseline_val = baseline_metrics.get(metric)
                new_val = new_metrics.get(metric)
                if baseline_val is None or new_val is None:
                    continue

                delta = new_val - baseline_val
                passed = delta >= -threshold

                results.append(RegressionTestResult(
                    test_name=f"slice_regression_{metric}_{slice_name}",
                    metric=metric,
                    baseline_value=round(float(baseline_val), 4),
                    new_value=round(float(new_val), 4),
                    delta=round(float(delta), 4),
                    threshold=-threshold,
                    passed=passed,
                    severity="error" if not passed else "info",
                    message=(f"[{slice_name}] {metric}: {baseline_val:.4f} → "
                             f"{new_val:.4f} (Δ={delta:+.4f})"),
                    slice_name=slice_name,
                ))

        return results

    def _run_statistical_regression_test(
        self,
        y_true: np.ndarray,
        y_prob_new: np.ndarray,
    ) -> list[RegressionTestResult]:
        """
        McNemar's test for significant difference between baseline and new model errors.
        Tests whether the models make significantly different errors — not just metric deltas.
        """
        y_pred_baseline = (self.baseline_y_prob >= self.decision_threshold).astype(int)
        y_pred_new = (y_prob_new >= self.decision_threshold).astype(int)

        # Cases where models disagree
        baseline_correct = (y_pred_baseline == y_true)
        new_correct = (y_pred_new == y_true)

        # b: baseline wrong, new correct
        # c: baseline correct, new wrong
        b = int((~baseline_correct & new_correct).sum())
        c = int((baseline_correct & ~new_correct).sum())

        # McNemar's test statistic
        if b + c == 0:
            return []

        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
        p_value = float(1 - stats.chi2.cdf(chi2, df=1))

        return [RegressionTestResult(
            test_name="mcnemar_error_difference",
            metric="error_pattern_p_value",
            baseline_value=float(b),   # cases new model fixed
            new_value=float(c),        # cases new model broke
            delta=float(b - c),
            threshold=0.0,
            passed=c <= b,  # concern if new model breaks more than it fixes
            severity="warning" if c > b else "info",
            message=(f"McNemar's test: new model FIXED {b} errors, BROKE {c} (p={p_value:.4f}). "
                     f"{'New model introduces more errors than it fixes!' if c > b else 'Net improvement.'}"),
            slice_name="overall",
        )]

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _compute_metrics(
        self, y_true: np.ndarray, y_prob: np.ndarray
    ) -> dict[str, float]:
        metrics: dict[str, float] = {}
        if len(np.unique(y_true)) < 2:
            return {"auroc": float("nan"), "f1": float("nan")}

        try:
            metrics["auroc"] = roc_auc_score(y_true, y_prob)
        except Exception:
            metrics["auroc"] = float("nan")
        try:
            metrics["auprc"] = average_precision_score(y_true, y_prob)
        except Exception:
            metrics["auprc"] = float("nan")

        y_pred = (y_prob >= self.decision_threshold).astype(int)
        from sklearn.metrics import confusion_matrix, accuracy_score
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn, fp, fn, tp = 0, 0, 0, int(y_true.sum())

        metrics["sensitivity"] = tp / max(tp + fn, 1)
        metrics["specificity"] = tn / max(tn + fp, 1)
        metrics["ppv"] = tp / max(tp + fp, 1)
        metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics["brier_score"] = brier_score_loss(y_true, y_prob)

        return {k: round(float(v), 4) for k, v in metrics.items()}

    def save_baseline(self, path: str | Path) -> None:
        """Save baseline predictions to file for future regression tests."""
        baseline_data = {
            "model_name": self.baseline_model_name,
            "golden_test_set": self.golden_test_set_name,
            "timestamp": datetime.utcnow().isoformat(),
            "y_true": self.baseline_y_true.tolist(),
            "y_prob": self.baseline_y_prob.tolist(),
            "metrics": self._baseline_metrics,
        }
        Path(path).write_text(json.dumps(baseline_data))
        logger.info("Baseline saved to %s", path)

    @classmethod
    def load_baseline(cls, path: str | Path) -> "ModelRegressionTester":
        """Load a previously saved baseline."""
        data = json.loads(Path(path).read_text())
        return cls(
            baseline_predictions={
                "y_true": np.array(data["y_true"]),
                "y_prob": np.array(data["y_prob"]),
            },
            golden_test_set_name=data["golden_test_set"],
            baseline_model_name=data["model_name"],
        )
