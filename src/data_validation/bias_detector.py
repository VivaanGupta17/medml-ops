"""
Demographic Bias Detector
==========================
FDA-required subgroup analysis and fairness metrics for medical AI models.

Computes performance stratification by age group, biological sex, and ethnicity;
calculates equalized odds, demographic parity, and calibration curves per subgroup;
and produces automated HTML bias reports ready for inclusion in 510(k) submissions.

FDA GMLP Alignment:
  - Principle 6: Model design and training — address bias across demographic groups
  - Principle 3: Clinical study design — ensure representative populations
  - Principle 7: Human-AI collaboration — disclose model limitations by subgroup

References:
  - FDA AI/ML Action Plan (Jan 2021)
  - Hardt et al. "Equality of Opportunity in Supervised Learning" (NeurIPS 2016)
  - Gebru et al. "Datasheets for Datasets" (2021)
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
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.calibration import calibration_curve

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class SubgroupMetrics:
    """Performance metrics for a single demographic subgroup."""
    attribute: str
    group: str
    n_samples: int
    n_positive: int
    prevalence: float
    auroc: float | None
    auprc: float | None
    accuracy: float
    sensitivity: float  # recall / TPR
    specificity: float  # TNR
    ppv: float          # precision
    npv: float
    f1: float
    brier_score: float | None
    support: int

    def to_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in self.__dict__.items()}


@dataclass
class FairnessMetrics:
    """Fairness metrics comparing subgroup pairs."""
    attribute: str
    reference_group: str
    comparison_group: str
    demographic_parity_gap: float        # |P(ŷ=1|A=a) - P(ŷ=1|A=b)|
    equalized_odds_tpr_gap: float        # |TPR_a - TPR_b|
    equalized_odds_fpr_gap: float        # |FPR_a - FPR_b|
    equal_opportunity_gap: float         # |TPR_a - TPR_b| (same as TPR gap)
    auroc_gap: float | None
    calibration_gap: float | None        # |E[ŷ|A=a] - E[ŷ|A=b]| among positives

    @property
    def max_gap(self) -> float:
        gaps = [self.demographic_parity_gap, self.equalized_odds_tpr_gap,
                self.equalized_odds_fpr_gap]
        return max(g for g in gaps if g is not None)

    @property
    def flag_level(self) -> str:
        """Flag severity based on FDA guidance thresholds."""
        m = self.max_gap
        if m > 0.10:
            return "HIGH"
        elif m > 0.05:
            return "MODERATE"
        else:
            return "LOW"

    def to_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in self.__dict__.items()}


@dataclass
class BiasReport:
    """Complete bias analysis report for an ML model."""
    model_name: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    subgroup_metrics: list[SubgroupMetrics] = field(default_factory=list)
    fairness_metrics: list[FairnessMetrics] = field(default_factory=list)
    overall_metrics: dict[str, float] = field(default_factory=dict)
    flag_summary: dict[str, str] = field(default_factory=dict)
    gmlp_alignment: str = "Principles 3, 6, 7"

    def has_high_bias_flags(self) -> bool:
        return any(fm.flag_level == "HIGH" for fm in self.fairness_metrics)

    def summary(self) -> str:
        lines = [
            f"=== Bias Analysis Report: {self.model_name} ===",
            f"Timestamp: {self.timestamp}",
            f"",
            f"Overall Performance:",
        ]
        for k, v in self.overall_metrics.items():
            lines.append(f"  {k:<20}: {v:.4f}")

        lines += ["", "Fairness Flags:"]
        for fm in self.fairness_metrics:
            lines.append(
                f"  [{fm.flag_level:8s}] {fm.attribute} | "
                f"{fm.reference_group} vs {fm.comparison_group} | "
                f"TPR gap: {fm.equalized_odds_tpr_gap:.3f}, "
                f"Dem. parity: {fm.demographic_parity_gap:.3f}"
            )

        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "timestamp": self.timestamp,
            "gmlp_alignment": self.gmlp_alignment,
            "overall_metrics": {k: round(v, 4) for k, v in self.overall_metrics.items()},
            "has_high_bias_flags": self.has_high_bias_flags(),
            "subgroup_metrics": [m.to_dict() for m in self.subgroup_metrics],
            "fairness_metrics": [fm.to_dict() for fm in self.fairness_metrics],
        }

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))
        logger.info("Bias report saved to %s", path)

    def save_html(self, path: str | Path) -> None:
        """Generate a styled HTML bias report."""
        html = self._render_html()
        Path(path).write_text(html)
        logger.info("HTML bias report saved to %s", path)

    def _render_html(self) -> str:
        flag_colors = {"HIGH": "#dc2626", "MODERATE": "#d97706", "LOW": "#16a34a"}

        subgroup_rows = ""
        for sm in self.subgroup_metrics:
            auroc_str = f"{sm.auroc:.3f}" if sm.auroc is not None else "N/A"
            subgroup_rows += f"""
            <tr>
                <td>{sm.attribute}</td>
                <td><strong>{sm.group}</strong></td>
                <td>{sm.n_samples:,}</td>
                <td>{sm.prevalence:.1%}</td>
                <td>{auroc_str}</td>
                <td>{sm.sensitivity:.3f}</td>
                <td>{sm.specificity:.3f}</td>
                <td>{sm.f1:.3f}</td>
            </tr>"""

        fairness_rows = ""
        for fm in self.fairness_metrics:
            color = flag_colors[fm.flag_level]
            fairness_rows += f"""
            <tr>
                <td>{fm.attribute}</td>
                <td>{fm.reference_group}</td>
                <td>{fm.comparison_group}</td>
                <td>{fm.demographic_parity_gap:.4f}</td>
                <td>{fm.equalized_odds_tpr_gap:.4f}</td>
                <td>{fm.equalized_odds_fpr_gap:.4f}</td>
                <td style="color:{color}; font-weight:bold;">{fm.flag_level}</td>
            </tr>"""

        overall_rows = "".join(
            f'<tr><td>{k}</td><td>{v:.4f}</td></tr>'
            for k, v in self.overall_metrics.items()
        )

        high_flag_warning = ""
        if self.has_high_bias_flags():
            high_flag_warning = """
            <div class="warning-banner">
                ⚠ HIGH BIAS FLAGS DETECTED — Review required before deployment per FDA GMLP Principle 6.
            </div>"""

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Bias Analysis Report – {self.model_name}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
          max-width: 1200px; margin: 0 auto; padding: 24px; color: #111; }}
  h1 {{ color: #1e3a5f; border-bottom: 3px solid #1e3a5f; padding-bottom: 8px; }}
  h2 {{ color: #1e3a5f; margin-top: 32px; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 14px; }}
  th {{ background: #1e3a5f; color: white; padding: 10px 12px; text-align: left; }}
  td {{ padding: 8px 12px; border-bottom: 1px solid #e5e7eb; }}
  tr:hover {{ background: #f8fafc; }}
  .meta {{ color: #6b7280; font-size: 13px; margin-bottom: 24px; }}
  .warning-banner {{ background: #fef2f2; border: 2px solid #dc2626; border-radius: 6px;
                     padding: 16px; color: #dc2626; font-weight: bold; margin: 16px 0; }}
  .gmlp-note {{ background: #eff6ff; border-left: 4px solid #3b82f6; padding: 12px 16px;
                color: #1e40af; font-size: 13px; margin: 16px 0; border-radius: 0 4px 4px 0; }}
</style>
</head>
<body>
<h1>Bias Analysis Report</h1>
<div class="meta">
  <strong>Model:</strong> {self.model_name} &nbsp;|&nbsp;
  <strong>Generated:</strong> {self.timestamp} &nbsp;|&nbsp;
  <strong>GMLP Alignment:</strong> {self.gmlp_alignment}
</div>
{high_flag_warning}
<div class="gmlp-note">
  <strong>FDA GMLP Note:</strong> This report implements Principle 6 (Model design and training)
  and Principle 3 (Clinical study design) by stratifying performance across demographic subgroups
  to identify and document potential bias before deployment.
</div>

<h2>Overall Model Performance</h2>
<table>
  <tr><th>Metric</th><th>Value</th></tr>
  {overall_rows}
</table>

<h2>Subgroup Performance</h2>
<table>
  <tr>
    <th>Attribute</th><th>Group</th><th>N</th><th>Prevalence</th>
    <th>AUROC</th><th>Sensitivity</th><th>Specificity</th><th>F1</th>
  </tr>
  {subgroup_rows}
</table>

<h2>Fairness Metrics</h2>
<table>
  <tr>
    <th>Attribute</th><th>Reference</th><th>Comparison</th>
    <th>Dem. Parity Gap</th><th>EqOdds TPR Gap</th>
    <th>EqOdds FPR Gap</th><th>Flag</th>
  </tr>
  {fairness_rows}
</table>

<div class="gmlp-note">
  <strong>Threshold guidance:</strong>
  LOW (&lt;5% gap) — acceptable &nbsp;|&nbsp;
  MODERATE (5–10%) — document and monitor &nbsp;|&nbsp;
  HIGH (&gt;10%) — address before submission
</div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Core Detector
# ---------------------------------------------------------------------------

class DemographicBiasDetector:
    """
    Analyzes model predictions for demographic bias across protected attributes.

    Computes per-subgroup performance metrics and pairwise fairness metrics
    including equalized odds, demographic parity, and calibration gaps.

    Args:
        sensitive_attributes: List of column names representing protected attributes
            (e.g., ["age_group", "sex", "ethnicity"]).
        reference_groups: Dict mapping attribute names to their reference group
            (e.g., {"sex": "M", "ethnicity": "White"}).
        min_subgroup_size: Minimum samples required to compute metrics for a subgroup.
        high_risk_threshold: Maximum allowed fairness gap before flagging HIGH risk.

    Example::

        detector = DemographicBiasDetector(
            sensitive_attributes=["age_group", "sex", "ethnicity"],
            reference_groups={"sex": "M"}
        )
        report = detector.analyze(
            y_true=labels,
            y_pred_proba=probabilities,
            demographics=demographic_df,
            model_name="chest_xray_v2"
        )
        report.save_html("reports/bias_audit.html")
    """

    def __init__(
        self,
        sensitive_attributes: list[str],
        reference_groups: dict[str, str] | None = None,
        min_subgroup_size: int = 30,
        high_risk_threshold: float = 0.10,
        decision_threshold: float = 0.5,
    ):
        self.sensitive_attributes = sensitive_attributes
        self.reference_groups = reference_groups or {}
        self.min_subgroup_size = min_subgroup_size
        self.high_risk_threshold = high_risk_threshold
        self.decision_threshold = decision_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        y_true: np.ndarray | pd.Series,
        y_pred_proba: np.ndarray | pd.Series,
        demographics: pd.DataFrame,
        model_name: str = "model",
        y_pred_binary: np.ndarray | pd.Series | None = None,
    ) -> BiasReport:
        """
        Run full bias analysis.

        Args:
            y_true: Ground truth binary labels (0/1).
            y_pred_proba: Predicted probabilities for the positive class.
            demographics: DataFrame with demographic columns. Must align with y_true.
            model_name: Model identifier for the report.
            y_pred_binary: Optional binary predictions. If None, derived from threshold.

        Returns:
            BiasReport with subgroup metrics and fairness assessments.
        """
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_pred_proba)
        y_pred = (
            np.asarray(y_pred_binary)
            if y_pred_binary is not None
            else (y_prob >= self.decision_threshold).astype(int)
        )

        # Validate alignment
        assert len(y_true) == len(y_prob) == len(demographics), (
            "y_true, y_pred_proba, and demographics must have the same length"
        )

        report = BiasReport(model_name=model_name)

        # Overall metrics
        report.overall_metrics = self._compute_overall_metrics(y_true, y_prob, y_pred)

        # Per-attribute, per-subgroup analysis
        for attr in self.sensitive_attributes:
            if attr not in demographics.columns:
                logger.warning("Attribute '%s' not found in demographics DataFrame — skipping", attr)
                continue

            groups = demographics[attr].dropna().unique()
            subgroup_metrics_map: dict[str, SubgroupMetrics] = {}

            for group in sorted(groups, key=str):
                mask = demographics[attr].values == group
                if mask.sum() < self.min_subgroup_size:
                    logger.info(
                        "Skipping %s=%s: only %d samples (min=%d)",
                        attr, group, mask.sum(), self.min_subgroup_size
                    )
                    continue

                sm = self._compute_subgroup_metrics(
                    y_true[mask], y_prob[mask], y_pred[mask], attr, str(group)
                )
                report.subgroup_metrics.append(sm)
                subgroup_metrics_map[str(group)] = sm

            # Pairwise fairness metrics
            ref_group = self.reference_groups.get(attr)
            if ref_group is None and subgroup_metrics_map:
                # Default to majority group as reference
                ref_group = max(subgroup_metrics_map,
                                key=lambda g: subgroup_metrics_map[g].n_samples)

            if ref_group and ref_group in subgroup_metrics_map:
                for grp, sm in subgroup_metrics_map.items():
                    if grp == ref_group:
                        continue
                    fm = self._compute_fairness_metrics(
                        subgroup_metrics_map[ref_group], sm, attr, ref_group, grp,
                        y_true, y_prob, demographics[attr].values, ref_group, grp
                    )
                    report.fairness_metrics.append(fm)

        # Intersectional analysis (age × sex if both present)
        if len(self.sensitive_attributes) >= 2:
            intersectional = self._intersectional_analysis(
                y_true, y_prob, y_pred, demographics
            )
            report.subgroup_metrics.extend(intersectional)

        logger.info(
            "Bias analysis complete: %d subgroups, %d fairness comparisons, "
            "high flags: %s",
            len(report.subgroup_metrics),
            len(report.fairness_metrics),
            report.has_high_bias_flags(),
        )
        return report

    # ------------------------------------------------------------------
    # Metric Computation
    # ------------------------------------------------------------------

    def _compute_overall_metrics(
        self, y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray
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
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)
        metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
        metrics["recall_sensitivity"] = recall_score(y_true, y_pred, zero_division=0)
        metrics["brier_score"] = brier_score_loss(y_true, y_prob)
        metrics["prevalence"] = float(y_true.mean())
        return {k: round(float(v), 4) for k, v in metrics.items()}

    def _compute_subgroup_metrics(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        y_pred: np.ndarray,
        attribute: str,
        group: str,
    ) -> SubgroupMetrics:
        n = len(y_true)
        n_pos = int(y_true.sum())

        # Confusion matrix components
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = (cm.ravel() if cm.shape == (2, 2)
                          else (cm[0, 0], 0, 0, cm[1, 1] if cm.shape[0] > 1 else 0))

        sensitivity = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        ppv = tp / max(tp + fp, 1)
        npv = tn / max(tn + fn, 1)

        try:
            auroc = roc_auc_score(y_true, y_prob) if n_pos > 0 and n_pos < n else None
        except Exception:
            auroc = None

        try:
            auprc = average_precision_score(y_true, y_prob) if n_pos > 0 else None
        except Exception:
            auprc = None

        try:
            brier = brier_score_loss(y_true, y_prob)
        except Exception:
            brier = None

        return SubgroupMetrics(
            attribute=attribute,
            group=group,
            n_samples=n,
            n_positive=n_pos,
            prevalence=round(float(y_true.mean()), 4),
            auroc=round(float(auroc), 4) if auroc is not None else None,
            auprc=round(float(auprc), 4) if auprc is not None else None,
            accuracy=round(float(accuracy_score(y_true, y_pred)), 4),
            sensitivity=round(float(sensitivity), 4),
            specificity=round(float(specificity), 4),
            ppv=round(float(ppv), 4),
            npv=round(float(npv), 4),
            f1=round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
            brier_score=round(float(brier), 4) if brier is not None else None,
            support=n,
        )

    def _compute_fairness_metrics(
        self,
        ref_sm: SubgroupMetrics,
        cmp_sm: SubgroupMetrics,
        attribute: str,
        ref_group: str,
        cmp_group: str,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        attr_values: np.ndarray,
        ref_val: str,
        cmp_val: str,
    ) -> FairnessMetrics:
        ref_mask = attr_values == ref_val
        cmp_mask = attr_values == cmp_val

        # Demographic Parity: |P(ŷ=1|A=ref) - P(ŷ=1|A=cmp)|
        ref_pred = (y_prob[ref_mask] >= self.decision_threshold).mean()
        cmp_pred = (y_prob[cmp_mask] >= self.decision_threshold).mean()
        dem_parity_gap = abs(float(ref_pred) - float(cmp_pred))

        # Equalized Odds: TPR and FPR gaps
        tpr_gap = abs(ref_sm.sensitivity - cmp_sm.sensitivity)
        fpr_ref = 1.0 - ref_sm.specificity
        fpr_cmp = 1.0 - cmp_sm.specificity
        fpr_gap = abs(fpr_ref - fpr_cmp)

        # AUROC gap
        auroc_gap = (
            abs(ref_sm.auroc - cmp_sm.auroc)
            if ref_sm.auroc is not None and cmp_sm.auroc is not None
            else None
        )

        # Calibration gap: mean predicted probability among true positives
        ref_pos_mask = ref_mask & (y_true == 1)
        cmp_pos_mask = cmp_mask & (y_true == 1)
        if ref_pos_mask.sum() > 0 and cmp_pos_mask.sum() > 0:
            cal_gap = abs(
                float(y_prob[ref_pos_mask].mean()) - float(y_prob[cmp_pos_mask].mean())
            )
        else:
            cal_gap = None

        return FairnessMetrics(
            attribute=attribute,
            reference_group=ref_group,
            comparison_group=cmp_group,
            demographic_parity_gap=round(dem_parity_gap, 4),
            equalized_odds_tpr_gap=round(tpr_gap, 4),
            equalized_odds_fpr_gap=round(fpr_gap, 4),
            equal_opportunity_gap=round(tpr_gap, 4),
            auroc_gap=round(float(auroc_gap), 4) if auroc_gap is not None else None,
            calibration_gap=round(float(cal_gap), 4) if cal_gap is not None else None,
        )

    def _intersectional_analysis(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        y_pred: np.ndarray,
        demographics: pd.DataFrame,
    ) -> list[SubgroupMetrics]:
        """
        Compute intersectional subgroup metrics (e.g., elderly Black women).
        Uses first two sensitive attributes available.
        """
        results = []
        avail = [a for a in self.sensitive_attributes[:2] if a in demographics.columns]
        if len(avail) < 2:
            return results

        attr1, attr2 = avail[0], avail[1]
        for g1 in demographics[attr1].dropna().unique():
            for g2 in demographics[attr2].dropna().unique():
                mask = (demographics[attr1].values == g1) & (demographics[attr2].values == g2)
                if mask.sum() < self.min_subgroup_size:
                    continue
                sm = self._compute_subgroup_metrics(
                    y_true[mask], y_prob[mask], y_pred[mask],
                    f"{attr1}×{attr2}", f"{g1}×{g2}"
                )
                results.append(sm)

        return results

    # ------------------------------------------------------------------
    # Calibration Analysis
    # ------------------------------------------------------------------

    def compute_calibration_by_group(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        demographics: pd.DataFrame,
        attribute: str,
        n_bins: int = 10,
    ) -> dict[str, dict[str, list[float]]]:
        """
        Compute calibration curves per demographic subgroup.
        Returns dict of {group: {"mean_predicted": [...], "fraction_positive": [...]}}.
        """
        if attribute not in demographics.columns:
            raise ValueError(f"Attribute '{attribute}' not in demographics")

        results = {}
        for group in demographics[attribute].dropna().unique():
            mask = demographics[attribute].values == group
            if mask.sum() < self.min_subgroup_size:
                continue
            fraction_pos, mean_pred = calibration_curve(
                y_true[mask], y_prob[mask], n_bins=n_bins, strategy="uniform"
            )
            results[str(group)] = {
                "mean_predicted": mean_pred.tolist(),
                "fraction_positive": fraction_pos.tolist(),
            }

        return results

    # ------------------------------------------------------------------
    # FDA Subgroup Analysis Report (510k-ready)
    # ------------------------------------------------------------------

    def generate_fda_subgroup_table(self, report: BiasReport) -> pd.DataFrame:
        """
        Generate an FDA submission-ready subgroup performance table.
        Formatted per FDA guidance for AI/ML-based SaMD submissions.
        """
        rows = []
        for sm in report.subgroup_metrics:
            rows.append({
                "Demographic Attribute": sm.attribute,
                "Subgroup": sm.group,
                "N": sm.n_samples,
                "Positive Cases": sm.n_positive,
                "Prevalence": f"{sm.prevalence:.1%}",
                "AUROC": f"{sm.auroc:.3f}" if sm.auroc else "N/A",
                "Sensitivity": f"{sm.sensitivity:.3f}",
                "Specificity": f"{sm.specificity:.3f}",
                "PPV": f"{sm.ppv:.3f}",
                "F1 Score": f"{sm.f1:.3f}",
                "Brier Score": f"{sm.brier_score:.3f}" if sm.brier_score else "N/A",
            })

        df = pd.DataFrame(rows)
        return df


# ---------------------------------------------------------------------------
# Standalone Analysis Functions
# ---------------------------------------------------------------------------

def compute_disparate_impact_ratio(
    y_pred: np.ndarray,
    attribute: np.ndarray,
    privileged_group: str,
    unprivileged_group: str,
) -> float:
    """
    Disparate Impact Ratio = P(ŷ=1|unprivileged) / P(ŷ=1|privileged).
    Value < 0.80 is typically considered discriminatory (80% rule).
    """
    priv_rate = y_pred[attribute == privileged_group].mean()
    unpriv_rate = y_pred[attribute == unprivileged_group].mean()
    if priv_rate == 0:
        return float("inf")
    return float(unpriv_rate / priv_rate)


def compute_counterfactual_fairness(
    model_predict_fn,
    X: pd.DataFrame,
    attribute: str,
    groups: list[str],
    n_samples: int = 100,
) -> dict[str, float]:
    """
    Estimate counterfactual fairness by flipping demographic attribute values
    and measuring prediction change.

    Returns dict of {group: mean_absolute_prediction_change}.
    """
    results = {}
    sample_idx = np.random.choice(len(X), min(n_samples, len(X)), replace=False)
    X_sample = X.iloc[sample_idx].copy()

    baseline_preds = model_predict_fn(X_sample)

    for group in groups:
        X_modified = X_sample.copy()
        X_modified[attribute] = group
        modified_preds = model_predict_fn(X_modified)
        results[group] = float(np.abs(baseline_preds - modified_preds).mean())

    return results


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run demographic bias analysis")
    parser.add_argument("--predictions", required=True, help="CSV with y_true, y_prob, demographics")
    parser.add_argument("--attributes", nargs="+", default=["age_group", "sex", "ethnicity"])
    parser.add_argument("--output-json", help="Save JSON report")
    parser.add_argument("--output-html", help="Save HTML report")
    parser.add_argument("--model-name", default="model_v1")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    df = pd.read_csv(args.predictions)
    y_true = df["y_true"].values
    y_prob = df["y_prob"].values
    demographics = df[[c for c in args.attributes if c in df.columns]]

    detector = DemographicBiasDetector(
        sensitive_attributes=args.attributes,
        decision_threshold=args.threshold,
    )
    report = detector.analyze(y_true, y_prob, demographics, model_name=args.model_name)
    print(report.summary())

    if args.output_json:
        report.save_json(args.output_json)
    if args.output_html:
        report.save_html(args.output_html)
