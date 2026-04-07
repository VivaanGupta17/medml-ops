"""
Unit tests for MedicalModelEvaluator.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.model_evaluator import MedicalModelEvaluator, EvaluationReport


def make_predictions(n: int = 300, seed: int = 42, auroc_target: float = 0.85):
    """Generate synthetic predictions with approximate target AUROC."""
    rng = np.random.default_rng(seed)
    y_true = rng.binomial(1, 0.30, n)
    signal_strength = auroc_target * 2 - 1  # 0.85 → 0.70
    y_prob = np.where(
        y_true == 1,
        rng.beta(5 * signal_strength, 2, n),
        rng.beta(2, 5 * signal_strength, n),
    )
    y_prob = np.clip(y_prob, 0.01, 0.99)
    return y_true, y_prob


class TestMedicalModelEvaluator:
    def test_basic_evaluation(self):
        y_true, y_prob = make_predictions()
        evaluator = MedicalModelEvaluator(bootstrap_n=100)
        report = evaluator.evaluate(y_true, y_prob, model_name="test")
        assert isinstance(report, EvaluationReport)
        assert report.core_metrics["auroc"] > 0.5
        assert len(report.operating_points) > 0
        assert report.calibration is not None

    def test_core_metrics_in_range(self):
        y_true, y_prob = make_predictions()
        evaluator = MedicalModelEvaluator(bootstrap_n=50)
        report = evaluator.evaluate(y_true, y_prob)
        m = report.core_metrics
        assert 0 <= m["auroc"] <= 1
        assert 0 <= m["sensitivity"] <= 1
        assert 0 <= m["specificity"] <= 1
        assert 0 <= m["ppv"] <= 1
        assert 0 <= m["f1"] <= 1
        assert 0 <= m["brier_score"] <= 1

    def test_operating_points_computed(self):
        y_true, y_prob = make_predictions()
        evaluator = MedicalModelEvaluator(
            bootstrap_n=50,
            sensitivity_targets=[0.85, 0.90],
        )
        report = evaluator.evaluate(y_true, y_prob)
        methods = {op.selection_method for op in report.operating_points}
        assert "youden_j" in methods
        assert "iso_f1" in methods

    def test_youden_j_maximizes_sensitivity_plus_specificity(self):
        y_true, y_prob = make_predictions()
        evaluator = MedicalModelEvaluator(bootstrap_n=50)
        report = evaluator.evaluate(y_true, y_prob)
        youden_op = next(op for op in report.operating_points if op.selection_method == "youden_j")
        standard_op = next(op for op in report.operating_points if op.selection_method == "threshold_0.5")
        # Youden J should be >= threshold 0.5 J
        assert youden_op.youden_j >= standard_op.youden_j - 0.01  # small tolerance

    def test_bootstrap_confidence_intervals(self):
        y_true, y_prob = make_predictions(n=300)
        evaluator = MedicalModelEvaluator(bootstrap_n=200)
        report = evaluator.evaluate(y_true, y_prob)
        assert "auroc" in report.confidence_intervals
        ci = report.confidence_intervals["auroc"]
        assert ci[0] < ci[1]
        assert ci[0] > 0
        assert ci[1] < 1

    def test_calibration_metrics(self):
        y_true, y_prob = make_predictions()
        evaluator = MedicalModelEvaluator(bootstrap_n=50)
        report = evaluator.evaluate(y_true, y_prob)
        cal = report.calibration
        assert cal is not None
        assert 0 <= cal.brier_score <= 1
        assert 0 <= cal.ece <= 1
        assert 0 <= cal.mce <= 1

    def test_predicate_comparison(self):
        y_true, y_prob = make_predictions(n=400, auroc_target=0.88)
        evaluator = MedicalModelEvaluator(bootstrap_n=50)
        report = evaluator.evaluate(
            y_true, y_prob,
            predicate_metrics={"auroc": 0.85, "sensitivity": 0.80},
        )
        assert len(report.statistical_tests) > 1
        assert "delta_vs_predicate" in str(report.predicate_comparison)

    def test_report_json_serialization(self, tmp_path):
        y_true, y_prob = make_predictions()
        evaluator = MedicalModelEvaluator(bootstrap_n=50)
        report = evaluator.evaluate(y_true, y_prob, dataset_name="test_set")
        json_path = tmp_path / "eval_report.json"
        report.save_json(json_path)
        assert json_path.exists()
        import json
        data = json.loads(json_path.read_text())
        assert "core_metrics" in data
        assert "auroc" in data["core_metrics"]

    def test_summary_string(self):
        y_true, y_prob = make_predictions()
        evaluator = MedicalModelEvaluator(bootstrap_n=50)
        report = evaluator.evaluate(y_true, y_prob)
        summary = report.summary()
        assert "AUROC" in summary or "auroc" in summary.lower()
        assert "Sensitivity" in summary or "sensitivity" in summary.lower()
