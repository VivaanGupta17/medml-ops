"""
Unit tests for DemographicBiasDetector.
"""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_validation.bias_detector import DemographicBiasDetector, BiasReport


def make_biased_predictions(n: int = 500, seed: int = 42):
    """Create synthetic predictions with deliberate sex-based bias."""
    rng = np.random.default_rng(seed)
    sex = rng.choice(["M", "F"], n)
    age_group = rng.choice(["18-40", "41-65", "66+"], n)
    ethnicity = rng.choice(["White", "Black", "Hispanic", "Asian"], n,
                            p=[0.60, 0.15, 0.15, 0.10])

    y_true = rng.binomial(1, 0.35, n)

    # Introduce deliberate bias: lower sensitivity for "Black" group
    y_prob = np.where(y_true == 1, rng.uniform(0.55, 0.95, n), rng.uniform(0.05, 0.45, n))
    # Degrade performance for Black ethnicity
    black_mask = ethnicity == "Black"
    y_prob[black_mask & (y_true == 1)] *= 0.7  # Systematically lower scores for positives

    y_prob = np.clip(y_prob, 0.01, 0.99)
    demographics = pd.DataFrame({
        "sex": sex,
        "age_group": age_group,
        "ethnicity": ethnicity,
    })
    return y_true, y_prob, demographics


class TestDemographicBiasDetector:
    def test_basic_analysis(self):
        y_true, y_prob, demo = make_biased_predictions()
        detector = DemographicBiasDetector(
            sensitive_attributes=["sex", "age_group", "ethnicity"],
            min_subgroup_size=30,
        )
        report = detector.analyze(y_true, y_prob, demo, model_name="test_model")
        assert isinstance(report, BiasReport)
        assert len(report.subgroup_metrics) > 0
        assert len(report.fairness_metrics) > 0

    def test_subgroup_metrics_computed(self):
        y_true, y_prob, demo = make_biased_predictions()
        detector = DemographicBiasDetector(
            sensitive_attributes=["sex"],
            min_subgroup_size=30,
        )
        report = detector.analyze(y_true, y_prob, demo)
        sex_groups = [sm for sm in report.subgroup_metrics if sm.attribute == "sex"]
        assert len(sex_groups) >= 2  # M and F
        for sg in sex_groups:
            assert sg.n_samples > 0
            assert 0 <= sg.sensitivity <= 1
            assert 0 <= sg.specificity <= 1

    def test_deliberate_bias_detected(self):
        """Artificially biased data should produce high fairness flags."""
        y_true, y_prob, demo = make_biased_predictions()
        detector = DemographicBiasDetector(
            sensitive_attributes=["ethnicity"],
            reference_groups={"ethnicity": "White"},
            min_subgroup_size=20,
            high_risk_threshold=0.10,
        )
        report = detector.analyze(y_true, y_prob, demo)
        # With deliberate 30% degradation for Black group, expect HIGH flag
        # (may vary by random seed, but should have at least MODERATE)
        flags = [fm.flag_level for fm in report.fairness_metrics]
        assert "HIGH" in flags or "MODERATE" in flags

    def test_overall_metrics_present(self):
        y_true, y_prob, demo = make_biased_predictions()
        detector = DemographicBiasDetector(sensitive_attributes=["sex"])
        report = detector.analyze(y_true, y_prob, demo)
        assert "auroc" in report.overall_metrics
        assert "f1" in report.overall_metrics
        assert report.overall_metrics["auroc"] > 0.5

    def test_report_serialization(self, tmp_path):
        y_true, y_prob, demo = make_biased_predictions()
        detector = DemographicBiasDetector(sensitive_attributes=["sex"])
        report = detector.analyze(y_true, y_prob, demo)

        json_path = tmp_path / "bias_report.json"
        report.save_json(json_path)
        assert json_path.exists()

        html_path = tmp_path / "bias_report.html"
        report.save_html(html_path)
        assert html_path.exists()
        html_content = html_path.read_text()
        assert "Bias Analysis Report" in html_content

    def test_small_subgroup_skipped(self):
        """Subgroups below min_size should be skipped gracefully."""
        y_true, y_prob, demo = make_biased_predictions(n=100)
        # With n=100 and min_size=200, no subgroup should be computed
        detector = DemographicBiasDetector(
            sensitive_attributes=["ethnicity"],
            min_subgroup_size=200,
        )
        report = detector.analyze(y_true, y_prob, demo)
        assert len(report.subgroup_metrics) == 0

    def test_fda_subgroup_table(self):
        y_true, y_prob, demo = make_biased_predictions()
        detector = DemographicBiasDetector(sensitive_attributes=["sex"])
        report = detector.analyze(y_true, y_prob, demo)
        df = detector.generate_fda_subgroup_table(report)
        assert isinstance(df, pd.DataFrame)
        assert "Demographic Attribute" in df.columns
        assert len(df) > 0
