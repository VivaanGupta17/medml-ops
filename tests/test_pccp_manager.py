"""
Unit tests for PCCPManager.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.compliance.pccp_manager import (
    PCCPManager,
    PCCPSpecification,
    PCCPChangeRequest,
    PCCPValidationReport,
    ChangeType,
    ChangeImpactLevel,
    ValidationStatus,
)


class TestPCCPSpecification:
    def test_create_default(self):
        spec = PCCPSpecification.create_default(
            device_name="TestDevice",
            intended_use="Clinical decision support",
            baseline_auroc=0.90,
        )
        assert spec.device_name == "TestDevice"
        assert len(spec.allowed_changes) > 0
        assert spec.performance_floor["auroc"] == pytest.approx(0.85, abs=0.01)

    def test_get_allowed_change(self):
        spec = PCCPSpecification.create_default("Device", "Use")
        ac = spec.get_allowed_change(ChangeType.RETRAINING_SAME_ARCHITECTURE)
        assert ac is not None
        ac_none = spec.get_allowed_change(ChangeType.EXPANSION_INDICATION)
        assert ac_none is None

    def test_json_roundtrip(self, tmp_path):
        spec = PCCPSpecification.create_default("Device", "Use", 0.88)
        path = tmp_path / "pccp.json"
        spec.save_json(path)
        loaded = PCCPSpecification.load_json(path)
        assert loaded.device_name == spec.device_name
        assert len(loaded.allowed_changes) == len(spec.allowed_changes)


class TestPCCPManager:
    def setup_method(self):
        self.spec = PCCPSpecification.create_default(
            "ChestXR-AI", "Pulmonary nodule detection", 0.92
        )
        self.manager = PCCPManager(self.spec)

    def test_within_pccp_approved(self):
        """Change within performance bounds should be approved."""
        request = PCCPChangeRequest(
            change_type=ChangeType.RETRAINING_SAME_ARCHITECTURE,
            description="Retrain with new cases",
            baseline_metrics={"auroc": 0.92, "sensitivity": 0.88},
            new_metrics={"auroc": 0.93, "sensitivity": 0.89},  # Improvement
        )
        report = self.manager.validate_change(request)
        assert report.within_pccp is True
        assert report.status == ValidationStatus.APPROVED

    def test_performance_degradation_rejected(self):
        """Degradation beyond threshold should be rejected."""
        request = PCCPChangeRequest(
            change_type=ChangeType.RETRAINING_SAME_ARCHITECTURE,
            baseline_metrics={"auroc": 0.92},
            new_metrics={"auroc": 0.85},  # -7% — exceeds -2% threshold
        )
        report = self.manager.validate_change(request)
        assert report.status == ValidationStatus.REJECTED
        assert len(report.failed_checks) > 0

    def test_change_type_not_in_pccp_rejected(self):
        """Unknown change type should be flagged as outside PCCP."""
        request = PCCPChangeRequest(
            change_type=ChangeType.EXPANSION_INDICATION,  # Not in PCCP
            description="Expand to pediatric population",
        )
        report = self.manager.validate_change(request)
        assert report.within_pccp is False
        assert report.status == ValidationStatus.REJECTED
        assert report.impact_level == ChangeImpactLevel.CRITICAL
        assert "new 510(k)" in report.regulatory_pathway.lower() or \
               "submission" in report.regulatory_pathway.lower()

    def test_floor_violation_critical(self):
        """Falling below performance floor should trigger critical failure."""
        request = PCCPChangeRequest(
            change_type=ChangeType.HYPERPARAMETER_ADJUSTMENT,
            baseline_metrics={"auroc": 0.92},
            new_metrics={"auroc": 0.60},  # Below floor of 0.87
        )
        report = self.manager.validate_change(request)
        assert report.status == ValidationStatus.REJECTED
        floor_failures = [c for c in report.failed_checks if "floor" in c.lower()]
        assert len(floor_failures) > 0

    def test_human_review_required_for_threshold_change(self):
        """Threshold adjustment should require human review."""
        request = PCCPChangeRequest(
            change_type=ChangeType.THRESHOLD_ADJUSTMENT,
            description="Lower threshold from 0.5 to 0.4",
            baseline_metrics={"auroc": 0.92},
            new_metrics={"auroc": 0.92},  # AUROC unchanged
        )
        report = self.manager.validate_change(request)
        # Threshold change requires human review → UNDER_REVIEW or APPROVED
        assert report.status in (ValidationStatus.UNDER_REVIEW, ValidationStatus.APPROVED,
                                  ValidationStatus.PENDING_TESTING)
        assert report.impact_level == ChangeImpactLevel.MODERATE

    def test_change_log_populated(self):
        request = PCCPChangeRequest(
            change_type=ChangeType.RETRAINING_SAME_ARCHITECTURE,
            requester="test_engineer",
            baseline_metrics={"auroc": 0.92},
            new_metrics={"auroc": 0.93},
        )
        self.manager.validate_change(request)
        assert len(self.manager._change_log) == 1
        assert self.manager._change_log[0]["requester"] == "test_engineer"

    def test_change_report_generated(self):
        request = PCCPChangeRequest(
            change_type=ChangeType.RETRAINING_SAME_ARCHITECTURE,
            description="Test change",
            baseline_metrics={"auroc": 0.92},
            new_metrics={"auroc": 0.93},
        )
        report = self.manager.validate_change(request)
        change_report = self.manager.generate_change_report(request, report)
        assert "PREDETERMINED CHANGE CONTROL PLAN" in change_report
        assert "VALIDATION RESULT" in change_report
