"""
Production Drift Monitor
=========================
Real-time data and prediction drift monitoring for deployed medical AI models.

Uses Population Stability Index (PSI), Kolmogorov-Smirnov tests, chi-square tests,
and Evidently AI integration to detect when production data diverges from training.

Exposes a FastAPI endpoint for programmatic monitoring queries and alerting.

FDA GMLP Alignment:
  - Principle 10: Transparency and monitoring of deployed model performance
  - Principle 9: Evaluation reflects real-world deployment conditions
  - 21 CFR 820.198: Complaint handling and post-market surveillance
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# Evidently AI (optional — graceful degradation)
try:
    from evidently import ColumnMapping
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, ClassificationPreset
    from evidently.metrics import DatasetDriftMetric
    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False
    logger.warning("Evidently AI not installed. Install: pip install evidently")


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class FeatureDriftResult:
    """Drift result for a single feature."""
    feature_name: str
    drift_detected: bool
    test_used: str
    statistic: float
    p_value: float | None
    psi: float | None
    reference_mean: float | None
    current_mean: float | None
    severity: str  # "none" | "low" | "moderate" | "high"

    PSI_THRESHOLDS = {"none": 0.0, "low": 0.10, "moderate": 0.25, "high": float("inf")}

    def to_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in self.__dict__.items() if not k.startswith("PSI")}


@dataclass
class DriftReport:
    """Complete drift monitoring report."""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    reference_period: str = ""
    current_period: str = ""
    n_reference_samples: int = 0
    n_current_samples: int = 0
    n_features_tested: int = 0
    n_features_drifted: int = 0
    overall_drift_detected: bool = False
    overall_drift_share: float = 0.0  # fraction of features drifted
    feature_results: list[FeatureDriftResult] = field(default_factory=list)
    prediction_drift: dict[str, Any] = field(default_factory=dict)
    label_drift: dict[str, Any] = field(default_factory=dict)
    alert_level: str = "GREEN"  # "GREEN" | "YELLOW" | "RED"
    recommended_actions: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"=== Drift Report [{self.alert_level}] ===",
            f"Timestamp : {self.timestamp}",
            f"Reference : {self.reference_period} (N={self.n_reference_samples:,})",
            f"Current   : {self.current_period} (N={self.n_current_samples:,})",
            f"",
            f"Features  : {self.n_features_drifted}/{self.n_features_tested} drifted "
            f"({self.overall_drift_share:.1%})",
            f"",
            "Drifted Features:",
        ]
        for fr in self.feature_results:
            if fr.drift_detected:
                lines.append(
                    f"  ⚠ {fr.feature_name:<25} PSI={fr.psi or 0:.3f} "
                    f"p={fr.p_value or 0:.4f} [{fr.severity.upper()}]"
                )
        if self.recommended_actions:
            lines += ["", "Recommended Actions:"]
            for action in self.recommended_actions:
                lines.append(f"  → {action}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "reference_period": self.reference_period,
            "current_period": self.current_period,
            "n_reference_samples": self.n_reference_samples,
            "n_current_samples": self.n_current_samples,
            "n_features_tested": self.n_features_tested,
            "n_features_drifted": self.n_features_drifted,
            "overall_drift_detected": self.overall_drift_detected,
            "overall_drift_share": round(self.overall_drift_share, 4),
            "alert_level": self.alert_level,
            "recommended_actions": self.recommended_actions,
            "feature_results": [fr.to_dict() for fr in self.feature_results],
            "prediction_drift": self.prediction_drift,
            "label_drift": self.label_drift,
        }

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))
        logger.info("Drift report saved to %s", path)


# ---------------------------------------------------------------------------
# Drift Monitor
# ---------------------------------------------------------------------------

class DriftMonitor:
    """
    Production drift monitor for medical AI deployments.

    Monitors:
    1. Feature/covariate drift (PSI + KS test)
    2. Prediction drift (distribution of model outputs)
    3. Label drift (if ground truth available — delayed labels)
    4. Demographic shift (age, sex composition changes)

    Triggers alerts when drift exceeds configurable thresholds.

    Example::

        monitor = DriftMonitor(reference_df=training_df, feature_columns=features)
        report = monitor.run_check(current_df=production_df)
        if report.alert_level == "RED":
            send_alert(report)
    """

    def __init__(
        self,
        reference_df: pd.DataFrame,
        feature_columns: list[str] | None = None,
        categorical_columns: list[str] | None = None,
        prediction_column: str = "y_prob",
        label_column: str | None = None,
        # Thresholds
        psi_warning_threshold: float = 0.10,
        psi_alert_threshold: float = 0.25,
        ks_p_value_threshold: float = 0.05,
        chi2_p_value_threshold: float = 0.05,
        drift_share_warning: float = 0.20,   # >20% features drifted → warning
        drift_share_alert: float = 0.50,     # >50% features drifted → alert
    ):
        self.reference_df = reference_df.copy()
        self.feature_columns = feature_columns or [
            c for c in reference_df.columns
            if c not in [prediction_column, label_column]
            and reference_df[c].dtype in [np.float64, np.float32, np.int64, np.int32]
        ]
        self.categorical_columns = categorical_columns or [
            c for c in reference_df.columns
            if c in (feature_columns or []) and reference_df[c].dtype == object
        ]
        self.prediction_column = prediction_column
        self.label_column = label_column
        self.psi_warning = psi_warning_threshold
        self.psi_alert = psi_alert_threshold
        self.ks_p_threshold = ks_p_value_threshold
        self.chi2_p_threshold = chi2_p_value_threshold
        self.drift_share_warning = drift_share_warning
        self.drift_share_alert = drift_share_alert

    # ------------------------------------------------------------------
    # Main Check
    # ------------------------------------------------------------------

    def run_check(
        self,
        current_df: pd.DataFrame,
        reference_period: str = "",
        current_period: str = "",
    ) -> DriftReport:
        """
        Run full drift analysis on current production data vs. reference.

        Args:
            current_df: Recent production data batch.
            reference_period: Label for reference period (e.g., "2024-Q1-train").
            current_period: Label for current period (e.g., "2024-11").

        Returns:
            DriftReport with alert level and recommended actions.
        """
        report = DriftReport(
            reference_period=reference_period or "training",
            current_period=current_period or datetime.utcnow().strftime("%Y-%m"),
            n_reference_samples=len(self.reference_df),
            n_current_samples=len(current_df),
        )

        # 1. Numeric feature drift (PSI + KS)
        numeric_results = self._check_numeric_drift(current_df)
        report.feature_results.extend(numeric_results)

        # 2. Categorical feature drift (chi-square + PSI)
        cat_results = self._check_categorical_drift(current_df)
        report.feature_results.extend(cat_results)

        # 3. Prediction distribution drift
        if self.prediction_column in current_df.columns and \
           self.prediction_column in self.reference_df.columns:
            report.prediction_drift = self._check_prediction_drift(current_df)

        # 4. Label drift (if labels available)
        if self.label_column and self.label_column in current_df.columns:
            report.label_drift = self._check_label_drift(current_df)

        # 5. Aggregate
        report.n_features_tested = len(report.feature_results)
        report.n_features_drifted = sum(1 for r in report.feature_results if r.drift_detected)
        report.overall_drift_share = (
            report.n_features_drifted / max(report.n_features_tested, 1)
        )
        report.overall_drift_detected = report.n_features_drifted > 0

        # 6. Alert level
        report.alert_level = self._compute_alert_level(report)
        report.recommended_actions = self._generate_recommendations(report)

        logger.info(
            "Drift check: %d/%d features drifted, alert=%s",
            report.n_features_drifted, report.n_features_tested, report.alert_level
        )
        return report

    # ------------------------------------------------------------------
    # Numeric Drift
    # ------------------------------------------------------------------

    def _check_numeric_drift(self, current_df: pd.DataFrame) -> list[FeatureDriftResult]:
        results = []
        common_cols = [c for c in self.feature_columns
                       if c in current_df.columns and c not in self.categorical_columns]

        for col in common_cols:
            ref_vals = self.reference_df[col].dropna().values
            cur_vals = current_df[col].dropna().values

            if len(ref_vals) < 10 or len(cur_vals) < 10:
                continue

            # KS test
            ks_stat, ks_p = stats.ks_2samp(ref_vals, cur_vals)

            # PSI
            psi = self._compute_psi(ref_vals, cur_vals)

            # Determine severity from PSI
            if psi > self.psi_alert:
                severity = "high"
            elif psi > self.psi_warning:
                severity = "moderate"
            elif ks_p < self.ks_p_threshold:
                severity = "low"
            else:
                severity = "none"

            drift_detected = (ks_p < self.ks_p_threshold) or (psi > self.psi_warning)

            results.append(FeatureDriftResult(
                feature_name=col,
                drift_detected=drift_detected,
                test_used="KS + PSI",
                statistic=round(float(ks_stat), 4),
                p_value=round(float(ks_p), 6),
                psi=round(float(psi), 4),
                reference_mean=round(float(ref_vals.mean()), 4),
                current_mean=round(float(cur_vals.mean()), 4),
                severity=severity,
            ))

        return results

    def _check_categorical_drift(self, current_df: pd.DataFrame) -> list[FeatureDriftResult]:
        results = []
        for col in self.categorical_columns:
            if col not in current_df.columns:
                continue

            ref_counts = self.reference_df[col].value_counts()
            cur_counts = current_df[col].value_counts()
            all_cats = sorted(set(ref_counts.index) | set(cur_counts.index))

            if len(all_cats) < 2:
                continue

            ref_arr = np.array([ref_counts.get(c, 0) for c in all_cats], dtype=float)
            cur_arr = np.array([cur_counts.get(c, 0) for c in all_cats], dtype=float)

            # Scale current to reference size for chi-square
            scale = ref_arr.sum() / max(cur_arr.sum(), 1)
            cur_scaled = cur_arr * scale

            try:
                chi2, p_value = stats.chisquare(f_obs=cur_scaled, f_exp=ref_arr)
            except Exception:
                continue

            # PSI for categorical
            eps = 1e-8
            ref_pct = (ref_arr + eps) / (ref_arr.sum() + eps)
            cur_pct = (cur_arr + eps) / (cur_arr.sum() + eps)
            psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))

            drift_detected = p_value < self.chi2_p_threshold

            results.append(FeatureDriftResult(
                feature_name=col,
                drift_detected=drift_detected,
                test_used="chi-square + PSI",
                statistic=round(float(chi2), 4),
                p_value=round(float(p_value), 6),
                psi=round(float(psi), 4),
                reference_mean=None,
                current_mean=None,
                severity="moderate" if drift_detected else "none",
            ))

        return results

    # ------------------------------------------------------------------
    # Prediction Drift
    # ------------------------------------------------------------------

    def _check_prediction_drift(self, current_df: pd.DataFrame) -> dict[str, Any]:
        """Monitor drift in model output distribution."""
        ref_preds = self.reference_df[self.prediction_column].dropna().values
        cur_preds = current_df[self.prediction_column].dropna().values

        ks_stat, ks_p = stats.ks_2samp(ref_preds, cur_preds)
        psi = self._compute_psi(ref_preds, cur_preds)

        # Positive rate shift
        ref_pos_rate = float((ref_preds >= 0.5).mean())
        cur_pos_rate = float((cur_preds >= 0.5).mean())

        return {
            "drift_detected": bool(ks_p < self.ks_p_threshold or psi > self.psi_warning),
            "ks_statistic": round(float(ks_stat), 4),
            "ks_p_value": round(float(ks_p), 6),
            "psi": round(float(psi), 4),
            "reference_mean_score": round(float(ref_preds.mean()), 4),
            "current_mean_score": round(float(cur_preds.mean()), 4),
            "reference_positive_rate": round(ref_pos_rate, 4),
            "current_positive_rate": round(cur_pos_rate, 4),
            "positive_rate_delta": round(cur_pos_rate - ref_pos_rate, 4),
        }

    # ------------------------------------------------------------------
    # Label Drift
    # ------------------------------------------------------------------

    def _check_label_drift(self, current_df: pd.DataFrame) -> dict[str, Any]:
        """Monitor shift in observed label prevalence (delayed ground truth)."""
        ref_labels = self.reference_df[self.label_column].dropna().values
        cur_labels = current_df[self.label_column].dropna().values

        ref_prev = float(ref_labels.mean())
        cur_prev = float(cur_labels.mean())

        chi2, p_value = stats.chisquare(
            f_obs=[cur_labels.sum(), len(cur_labels) - cur_labels.sum()],
            f_exp=[ref_prev * len(cur_labels), (1 - ref_prev) * len(cur_labels)],
        )

        return {
            "drift_detected": bool(p_value < self.chi2_p_threshold),
            "reference_prevalence": round(ref_prev, 4),
            "current_prevalence": round(cur_prev, 4),
            "prevalence_delta": round(cur_prev - ref_prev, 4),
            "chi2_statistic": round(float(chi2), 4),
            "p_value": round(float(p_value), 6),
        }

    # ------------------------------------------------------------------
    # Evidently AI Integration
    # ------------------------------------------------------------------

    def run_evidently_report(
        self, current_df: pd.DataFrame, output_path: str | Path
    ) -> None:
        """
        Generate an Evidently AI HTML drift report.
        Provides richer visualizations than the built-in checks.
        """
        if not EVIDENTLY_AVAILABLE:
            logger.warning("Evidently not installed — skipping HTML report")
            return

        target = self.label_column
        prediction = self.prediction_column

        column_mapping = ColumnMapping(
            target=target,
            prediction=prediction,
            numerical_features=[c for c in self.feature_columns
                                 if c not in self.categorical_columns],
            categorical_features=self.categorical_columns,
        )

        report = Report(metrics=[
            DataDriftPreset(),
            ClassificationPreset() if target else DatasetDriftMetric(),
        ])

        report.run(
            reference_data=self.reference_df,
            current_data=current_df,
            column_mapping=column_mapping,
        )

        report.save_html(str(output_path))
        logger.info("Evidently report saved to %s", output_path)

    # ------------------------------------------------------------------
    # Alert Logic
    # ------------------------------------------------------------------

    def _compute_alert_level(self, report: DriftReport) -> str:
        """Determine alert level from aggregate drift metrics."""
        high_psi_features = [
            r for r in report.feature_results
            if r.psi is not None and r.psi > self.psi_alert
        ]
        pred_drift = report.prediction_drift.get("drift_detected", False)

        if len(high_psi_features) >= 3 or pred_drift:
            return "RED"
        elif report.overall_drift_share > self.drift_share_warning:
            return "YELLOW"
        else:
            return "GREEN"

    def _generate_recommendations(self, report: DriftReport) -> list[str]:
        """Generate actionable recommendations based on drift findings."""
        actions = []

        if report.alert_level == "RED":
            actions.append(
                "URGENT: Significant distribution shift detected. "
                "Evaluate model performance on recent labeled cases before continuing deployment."
            )
            actions.append(
                "Consider model re-training with recent data following PCCP protocol."
            )

        if report.overall_drift_share > self.drift_share_warning:
            drifted = [r.feature_name for r in report.feature_results if r.drift_detected]
            actions.append(
                f"Investigate data pipeline for features: {', '.join(drifted[:5])}"
            )

        if report.prediction_drift.get("positive_rate_delta", 0) > 0.10:
            actions.append(
                "Prediction rate has increased significantly. "
                "Verify label quality or check for data preprocessing changes."
            )

        if not actions:
            actions.append(
                "No critical drift detected. Continue scheduled monitoring."
            )

        return actions

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_psi(
        reference: np.ndarray, current: np.ndarray, n_bins: int = 10
    ) -> float:
        bins = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
        bins[0] = -np.inf
        bins[-1] = np.inf
        eps = 1e-8
        ref_pct = np.histogram(reference, bins=bins)[0] / max(len(reference), 1) + eps
        cur_pct = np.histogram(current, bins=bins)[0] / max(len(current), 1) + eps
        return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


# ---------------------------------------------------------------------------
# FastAPI Monitoring Endpoint
# ---------------------------------------------------------------------------

try:
    from fastapi import FastAPI
    from pydantic import BaseModel as PydanticModel

    monitor_app = FastAPI(title="MedML-Ops Drift Monitor", version="1.0.0")
    _monitor_instance: DriftMonitor | None = None

    class DriftCheckRequest(PydanticModel):
        current_data_path: str
        reference_period: str = ""
        current_period: str = ""
        output_html: str | None = None

    @monitor_app.get("/health")
    async def monitor_health():
        return {"status": "healthy", "evidently_available": EVIDENTLY_AVAILABLE}

    @monitor_app.post("/check")
    async def run_drift_check(req: DriftCheckRequest):
        global _monitor_instance
        if _monitor_instance is None:
            return {"error": "Monitor not initialized. Load reference data first."}

        current_df = pd.read_csv(req.current_data_path)
        report = _monitor_instance.run_check(
            current_df, req.reference_period, req.current_period
        )

        if req.output_html:
            _monitor_instance.run_evidently_report(current_df, req.output_html)

        return report.to_dict()

    @monitor_app.get("/alerts")
    async def get_recent_alerts():
        log_path = Path(os.environ.get("DRIFT_LOG_PATH", "logs/drift_reports.jsonl"))
        if not log_path.exists():
            return {"alerts": []}
        lines = log_path.read_text().strip().split("\n")[-50:]
        return {"alerts": [json.loads(l) for l in lines if l]}

    # Alias for docker-compose service
    app = monitor_app

except ImportError:
    pass
