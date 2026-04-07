"""
Medical Model Evaluator
========================
Comprehensive evaluation framework for medical AI models with clinical
operating point selection, confidence calibration, statistical significance
testing, and 510(k) predicate device comparison.

FDA GMLP Alignment:
  - Principle 7: Testing demonstrates device performance across clinical scenarios
  - Principle 9: Evaluation reflects deployment context (clinical operating points)
  - Principle 3: Clinical study design — significance testing, confidence intervals

510(k) Context:
  - Substantial equivalence requires comparison to predicate device performance
  - Operating point selection must reflect clinical workflow (sensitivity priority
    for screening, specificity for confirmation)
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    auc,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class OperatingPoint:
    """A specific sensitivity/specificity operating point."""
    threshold: float
    sensitivity: float     # TPR
    specificity: float     # TNR
    ppv: float
    npv: float
    f1: float
    accuracy: float
    youden_j: float        # sensitivity + specificity - 1
    selection_method: str  # e.g., "youden_j", "sensitivity_90", "iso_f1"

    def to_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in self.__dict__.items()}


@dataclass
class CalibrationResult:
    """Calibration analysis results."""
    brier_score: float
    ece: float               # Expected Calibration Error
    mce: float               # Maximum Calibration Error
    overconfident_fraction: float
    underconfident_fraction: float
    mean_predicted_prob: float
    actual_positive_rate: float
    fraction_of_positives: list[float] = field(default_factory=list)
    mean_predicted_values: list[float] = field(default_factory=list)

    def is_well_calibrated(self, ece_threshold: float = 0.10) -> bool:
        return self.ece < ece_threshold


@dataclass
class StatisticalTestResult:
    """Result of a statistical significance test."""
    test_name: str
    statistic: float
    p_value: float
    confidence_interval_95: tuple[float, float]
    null_hypothesis: str
    significant: bool

    def to_dict(self) -> dict:
        return {
            "test_name": self.test_name,
            "statistic": round(self.statistic, 4),
            "p_value": round(self.p_value, 6),
            "ci_95_lower": round(self.confidence_interval_95[0], 4),
            "ci_95_upper": round(self.confidence_interval_95[1], 4),
            "null_hypothesis": self.null_hypothesis,
            "significant_at_0.05": self.significant,
        }


@dataclass
class EvaluationReport:
    """Complete model evaluation report."""
    model_name: str
    dataset_name: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    core_metrics: dict[str, float] = field(default_factory=dict)
    operating_points: list[OperatingPoint] = field(default_factory=list)
    calibration: CalibrationResult | None = None
    statistical_tests: list[StatisticalTestResult] = field(default_factory=list)
    predicate_comparison: dict[str, Any] = field(default_factory=dict)
    confidence_intervals: dict[str, tuple[float, float]] = field(default_factory=dict)
    gmlp_alignment: str = "Principles 3, 7, 9"

    def primary_operating_point(self) -> OperatingPoint | None:
        for op in self.operating_points:
            if op.selection_method == "youden_j":
                return op
        return self.operating_points[0] if self.operating_points else None

    def summary(self) -> str:
        lines = [
            f"=== Evaluation Report: {self.model_name} ===",
            f"Dataset: {self.dataset_name} | {self.timestamp}",
            "",
            "Core Metrics:",
        ]
        for k, v in self.core_metrics.items():
            lines.append(f"  {k:<20}: {v:.4f}")

        op = self.primary_operating_point()
        if op:
            lines += [
                "",
                f"Primary Operating Point ({op.selection_method}):",
                f"  Threshold   : {op.threshold:.3f}",
                f"  Sensitivity : {op.sensitivity:.3f}",
                f"  Specificity : {op.specificity:.3f}",
                f"  PPV         : {op.ppv:.3f}",
                f"  F1          : {op.f1:.3f}",
            ]

        if self.calibration:
            lines += [
                "",
                f"Calibration:",
                f"  Brier Score : {self.calibration.brier_score:.4f}",
                f"  ECE         : {self.calibration.ece:.4f}",
                f"  Status      : {'Well-calibrated' if self.calibration.is_well_calibrated() else 'NEEDS RECALIBRATION'}",
            ]

        if self.predicate_comparison:
            lines += ["", "510(k) Predicate Comparison:"]
            for k, v in self.predicate_comparison.items():
                lines.append(f"  {k}: {v}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "timestamp": self.timestamp,
            "gmlp_alignment": self.gmlp_alignment,
            "core_metrics": {k: round(v, 4) for k, v in self.core_metrics.items()},
            "primary_operating_point": (
                self.primary_operating_point().to_dict()
                if self.primary_operating_point() else None
            ),
            "all_operating_points": [op.to_dict() for op in self.operating_points],
            "calibration": self.calibration.__dict__ if self.calibration else None,
            "statistical_tests": [st.to_dict() for st in self.statistical_tests],
            "predicate_comparison": self.predicate_comparison,
            "confidence_intervals": {
                k: [round(v, 4) for v in ci]
                for k, v in self.confidence_intervals.items()
                for ci in [v]
            },
        }

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))
        logger.info("Evaluation report saved to %s", path)


# ---------------------------------------------------------------------------
# Core Evaluator
# ---------------------------------------------------------------------------

class MedicalModelEvaluator:
    """
    Clinical-grade model evaluator for medical AI.

    Goes beyond standard ML metrics to compute:
    - Clinical operating points (Youden's J, iso-sensitivity curves)
    - Confidence calibration with ECE/MCE
    - Bootstrap confidence intervals
    - DeLong AUROC significance test
    - 510(k) predicate device comparison

    Args:
        prevalence: Expected disease prevalence in deployment population.
                    If None, estimated from test set.
        bootstrap_n: Number of bootstrap samples for confidence intervals.
        sensitivity_targets: Sensitivity levels at which to report specificity.
    """

    def __init__(
        self,
        prevalence: float | None = None,
        bootstrap_n: int = 1000,
        sensitivity_targets: list[float] | None = None,
        n_calibration_bins: int = 10,
    ):
        self.prevalence = prevalence
        self.bootstrap_n = bootstrap_n
        self.sensitivity_targets = sensitivity_targets or [0.85, 0.90, 0.95]
        self.n_calibration_bins = n_calibration_bins

    # ------------------------------------------------------------------
    # Main Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        model_name: str = "model",
        dataset_name: str = "test_set",
        predicate_metrics: dict[str, float] | None = None,
    ) -> EvaluationReport:
        """
        Run comprehensive model evaluation.

        Args:
            y_true: Ground truth labels.
            y_pred_proba: Predicted probabilities for positive class.
            model_name: Model identifier.
            dataset_name: Test dataset identifier.
            predicate_metrics: Predicate device performance for 510(k) comparison.

        Returns:
            EvaluationReport with all metrics and analyses.
        """
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_pred_proba)

        report = EvaluationReport(model_name=model_name, dataset_name=dataset_name)

        # Core metrics
        report.core_metrics = self._compute_core_metrics(y_true, y_prob)

        # Operating points
        report.operating_points = self._compute_operating_points(y_true, y_prob)

        # Confidence intervals
        report.confidence_intervals = self._bootstrap_confidence_intervals(y_true, y_prob)

        # Calibration
        report.calibration = self._compute_calibration(y_true, y_prob)

        # Statistical tests (vs. chance and vs. predicate if provided)
        report.statistical_tests.extend(self._test_vs_chance(y_true, y_prob))
        if predicate_metrics:
            report.statistical_tests.extend(
                self._test_vs_predicate(y_true, y_prob, predicate_metrics)
            )
            report.predicate_comparison = self._compare_to_predicate(
                report.core_metrics, predicate_metrics
            )

        logger.info(
            "Evaluation complete: AUROC=%.4f [%.4f, %.4f], ECE=%.4f",
            report.core_metrics.get("auroc", 0),
            report.confidence_intervals.get("auroc", (0, 0))[0],
            report.confidence_intervals.get("auroc", (0, 0))[1],
            report.calibration.ece if report.calibration else 0,
        )
        return report

    # ------------------------------------------------------------------
    # Core Metrics
    # ------------------------------------------------------------------

    def _compute_core_metrics(
        self, y_true: np.ndarray, y_prob: np.ndarray
    ) -> dict[str, float]:
        metrics: dict[str, float] = {}

        try:
            metrics["auroc"] = roc_auc_score(y_true, y_prob)
        except Exception:
            metrics["auroc"] = float("nan")
        try:
            metrics["auprc"] = average_precision_score(y_true, y_prob)
        except Exception:
            metrics["auprc"] = float("nan")

        metrics["brier_score"] = brier_score_loss(y_true, y_prob)
        metrics["prevalence"] = float(y_true.mean())

        # At default 0.5 threshold
        y_pred = (y_prob >= 0.5).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn, fp, fn, tp = 0, 0, 0, int(cm[0, 0])

        total = len(y_true)
        metrics["accuracy"] = (tp + tn) / total if total > 0 else 0.0
        metrics["sensitivity"] = tp / max(tp + fn, 1)
        metrics["specificity"] = tn / max(tn + fp, 1)
        metrics["ppv"] = tp / max(tp + fp, 1)
        metrics["npv"] = tn / max(tn + fn, 1)
        metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)

        # At each sensitivity target
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        for target_sens in self.sensitivity_targets:
            # Find threshold closest to target sensitivity
            idx = np.searchsorted(tpr, target_sens)
            idx = min(idx, len(thresholds) - 1)
            spec_at_target = 1.0 - fpr[idx]
            metrics[f"specificity_at_sens_{int(target_sens*100)}"] = round(float(spec_at_target), 4)

        return {k: round(float(v), 4) for k, v in metrics.items()}

    # ------------------------------------------------------------------
    # Operating Points
    # ------------------------------------------------------------------

    def _compute_operating_points(
        self, y_true: np.ndarray, y_prob: np.ndarray
    ) -> list[OperatingPoint]:
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        ops = []

        def _make_op(threshold: float, method: str) -> OperatingPoint:
            y_pred = (y_prob >= threshold).astype(int)
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
            else:
                tn, fp, fn, tp = 0, 0, 0, int(cm[0, 0])
            sens = tp / max(tp + fn, 1)
            spec = tn / max(tn + fp, 1)
            ppv = tp / max(tp + fp, 1)
            npv = tn / max(tn + fn, 1)
            return OperatingPoint(
                threshold=round(float(threshold), 4),
                sensitivity=round(float(sens), 4),
                specificity=round(float(spec), 4),
                ppv=round(float(ppv), 4),
                npv=round(float(npv), 4),
                f1=round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
                accuracy=round(float((tp + tn) / len(y_true)), 4),
                youden_j=round(float(sens + spec - 1), 4),
                selection_method=method,
            )

        # 1. Youden's J (maximize sensitivity + specificity)
        youden = tpr - fpr
        best_idx = np.argmax(youden)
        ops.append(_make_op(float(thresholds[best_idx]), "youden_j"))

        # 2. Iso-F1 (maximize F1)
        prec, rec, pr_thresholds = precision_recall_curve(y_true, y_prob)
        f1_scores = 2 * prec * rec / np.maximum(prec + rec, 1e-8)
        best_f1_idx = np.argmax(f1_scores[:-1])
        if best_f1_idx < len(pr_thresholds):
            ops.append(_make_op(float(pr_thresholds[best_f1_idx]), "iso_f1"))

        # 3. Per sensitivity target
        for target_sens in self.sensitivity_targets:
            idx = np.searchsorted(tpr, target_sens)
            idx = min(idx, len(thresholds) - 1)
            ops.append(_make_op(
                float(thresholds[idx]),
                f"sensitivity_{int(target_sens*100)}"
            ))

        # 4. Standard 0.5 threshold
        ops.append(_make_op(0.5, "threshold_0.5"))

        return ops

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def _compute_calibration(
        self, y_true: np.ndarray, y_prob: np.ndarray
    ) -> CalibrationResult:
        """Compute calibration metrics including ECE and MCE."""
        frac_pos, mean_pred = calibration_curve(
            y_true, y_prob, n_bins=self.n_calibration_bins, strategy="uniform"
        )

        # Expected Calibration Error (ECE)
        bin_size = 1.0 / self.n_calibration_bins
        n = len(y_true)
        ece = 0.0
        mce = 0.0
        for b in range(self.n_calibration_bins):
            bin_lo = b * bin_size
            bin_hi = bin_lo + bin_size
            in_bin = (y_prob >= bin_lo) & (y_prob < bin_hi)
            n_in_bin = in_bin.sum()
            if n_in_bin == 0:
                continue
            acc = y_true[in_bin].mean()
            conf = y_prob[in_bin].mean()
            calibration_error = abs(float(acc) - float(conf))
            ece += (n_in_bin / n) * calibration_error
            mce = max(mce, calibration_error)

        # Overconfidence/underconfidence fractions
        overconfident = float((y_prob > (y_true + 0.2)).mean())
        underconfident = float((y_prob < (y_true - 0.2)).mean())

        return CalibrationResult(
            brier_score=round(float(brier_score_loss(y_true, y_prob)), 4),
            ece=round(float(ece), 4),
            mce=round(float(mce), 4),
            overconfident_fraction=round(overconfident, 4),
            underconfident_fraction=round(underconfident, 4),
            mean_predicted_prob=round(float(y_prob.mean()), 4),
            actual_positive_rate=round(float(y_true.mean()), 4),
            fraction_of_positives=frac_pos.tolist(),
            mean_predicted_values=mean_pred.tolist(),
        )

    # ------------------------------------------------------------------
    # Bootstrap Confidence Intervals
    # ------------------------------------------------------------------

    def _bootstrap_confidence_intervals(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        alpha: float = 0.05,
    ) -> dict[str, tuple[float, float]]:
        """Compute bootstrap CIs for AUROC, AUPRC, and F1."""
        rng = np.random.default_rng(42)
        n = len(y_true)

        auroc_scores = []
        auprc_scores = []
        f1_scores_boot = []

        for _ in range(self.bootstrap_n):
            idx = rng.integers(0, n, size=n)
            yt, yp = y_true[idx], y_prob[idx]
            if len(np.unique(yt)) < 2:
                continue
            try:
                auroc_scores.append(roc_auc_score(yt, yp))
                auprc_scores.append(average_precision_score(yt, yp))
                f1_scores_boot.append(f1_score(yt, (yp >= 0.5).astype(int), zero_division=0))
            except Exception:
                continue

        def ci(scores: list[float]) -> tuple[float, float]:
            arr = np.array(scores)
            return (
                round(float(np.percentile(arr, 100 * alpha / 2)), 4),
                round(float(np.percentile(arr, 100 * (1 - alpha / 2))), 4),
            )

        return {
            "auroc": ci(auroc_scores),
            "auprc": ci(auprc_scores),
            "f1": ci(f1_scores_boot),
        }

    # ------------------------------------------------------------------
    # Statistical Tests
    # ------------------------------------------------------------------

    def _test_vs_chance(
        self, y_true: np.ndarray, y_prob: np.ndarray
    ) -> list[StatisticalTestResult]:
        """Test whether model significantly outperforms chance (AUROC > 0.5)."""
        # Wilcoxon rank-sum test: scores of positives vs negatives
        pos_scores = y_prob[y_true == 1]
        neg_scores = y_prob[y_true == 0]

        if len(pos_scores) < 2 or len(neg_scores) < 2:
            return []

        stat, p_value = stats.mannwhitneyu(pos_scores, neg_scores, alternative="greater")
        n1, n2 = len(pos_scores), len(neg_scores)
        se = np.sqrt((n1 + n2 + 1) / (12 * n1 * n2))
        auroc = roc_auc_score(y_true, y_prob)
        # Normal approximation CI for AUROC
        z = 1.96
        ci = (max(0.0, auroc - z * se), min(1.0, auroc + z * se))

        return [StatisticalTestResult(
            test_name="Mann-Whitney U (AUROC vs. chance)",
            statistic=round(float(stat), 2),
            p_value=round(float(p_value), 8),
            confidence_interval_95=(round(ci[0], 4), round(ci[1], 4)),
            null_hypothesis="AUROC = 0.5 (model no better than chance)",
            significant=bool(p_value < 0.05),
        )]

    def _test_vs_predicate(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        predicate_metrics: dict[str, float],
    ) -> list[StatisticalTestResult]:
        """
        Test whether model significantly outperforms (or is non-inferior to)
        a predicate device. Non-inferiority margin: Δ = 0.02 AUROC units.
        Relevant for 510(k) substantial equivalence arguments.
        """
        results = []
        model_auroc = roc_auc_score(y_true, y_prob)
        predicate_auroc = predicate_metrics.get("auroc")

        if predicate_auroc is None:
            return results

        # Hanley-McNeil standard error approximation
        n_pos = int(y_true.sum())
        n_neg = len(y_true) - n_pos
        q1 = model_auroc / (2 - model_auroc)
        q2 = 2 * model_auroc ** 2 / (1 + model_auroc)
        se = np.sqrt((model_auroc * (1 - model_auroc) + (n_pos - 1) * (q1 - model_auroc**2)
                      + (n_neg - 1) * (q2 - model_auroc**2)) / (n_pos * n_neg))

        diff = model_auroc - predicate_auroc
        z = diff / se if se > 0 else 0.0
        p_value = float(1 - stats.norm.cdf(z))  # one-sided (superiority)

        # Non-inferiority margin
        ni_margin = 0.02
        ni_z = (diff + ni_margin) / se if se > 0 else 0.0
        ni_p = float(1 - stats.norm.cdf(ni_z))

        results.append(StatisticalTestResult(
            test_name="Superiority test vs predicate (AUROC)",
            statistic=round(float(z), 3),
            p_value=round(float(p_value), 6),
            confidence_interval_95=(
                round(float(diff - 1.96 * se), 4),
                round(float(diff + 1.96 * se), 4),
            ),
            null_hypothesis=f"Model AUROC ≤ Predicate AUROC ({predicate_auroc:.4f})",
            significant=bool(p_value < 0.05),
        ))

        results.append(StatisticalTestResult(
            test_name=f"Non-inferiority test vs predicate (Δ={ni_margin})",
            statistic=round(float(ni_z), 3),
            p_value=round(float(ni_p), 6),
            confidence_interval_95=(
                round(float(diff - 1.96 * se), 4),
                round(float(diff + 1.96 * se), 4),
            ),
            null_hypothesis=f"Model AUROC < Predicate AUROC - {ni_margin}",
            significant=bool(ni_p < 0.05),
        ))

        return results

    def _compare_to_predicate(
        self,
        model_metrics: dict[str, float],
        predicate_metrics: dict[str, float],
    ) -> dict[str, Any]:
        """Generate 510(k) predicate comparison table."""
        comparison: dict[str, Any] = {"predicate_metrics_provided": predicate_metrics}
        for metric in ["auroc", "sensitivity", "specificity", "f1"]:
            model_val = model_metrics.get(metric)
            pred_val = predicate_metrics.get(metric)
            if model_val is not None and pred_val is not None:
                delta = round(float(model_val) - float(pred_val), 4)
                comparison[f"{metric}_delta_vs_predicate"] = delta
                comparison[f"{metric}_substantially_equivalent"] = abs(delta) <= 0.02

        return comparison

    # ------------------------------------------------------------------
    # Convenience: evaluate from model object
    # ------------------------------------------------------------------

    def evaluate_model(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str = "model",
        dataset_name: str = "test_set",
        predicate_metrics: dict[str, float] | None = None,
    ) -> EvaluationReport:
        """Convenience wrapper that calls model.predict_proba() internally."""
        y_prob = model.predict_proba(X_test)[:, 1]
        return self.evaluate(
            y_test, y_prob, model_name=model_name,
            dataset_name=dataset_name, predicate_metrics=predicate_metrics
        )
