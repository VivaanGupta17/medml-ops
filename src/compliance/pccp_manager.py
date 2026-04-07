"""
PCCP Manager — Predetermined Change Control Plan
=================================================
Implements the FDA's Predetermined Change Control Plan (PCCP) framework for
AI/ML-based Software as a Medical Device (SaMD).

The PCCP allows manufacturers to pre-define allowed modifications to their
AI/ML models — and the testing/validation procedures for those modifications —
so that post-market updates can be implemented without requiring a new 510(k)
submission for each change, as long as the change falls within the pre-approved
protocol.

References:
  - FDA Draft Guidance: "Marketing Submission Recommendations for a
    Predetermined Change Control Plan for Artificial Intelligence/Machine
    Learning (AI/ML)-Enabled Device Software Functions" (2023)
    https://www.fda.gov/regulatory-information/search-fda-guidance-documents/marketing-submission-recommendations-predetermined-change-control-plan-artificial

FDA GMLP Alignment:
  - Principle 5: Re-training practices transparency
  - Principle 10: Monitoring and updating deployed models
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ChangeType(str, Enum):
    """FDA PCCP modification type categories."""
    RETRAINING_SAME_ARCHITECTURE = "retraining_same_architecture"
    RETRAINING_NEW_DATA = "retraining_new_data"
    HYPERPARAMETER_ADJUSTMENT = "hyperparameter_adjustment"
    THRESHOLD_ADJUSTMENT = "threshold_adjustment"
    PREPROCESSING_CHANGE = "preprocessing_change"
    ARCHITECTURE_CHANGE = "architecture_change"       # May require new submission
    LABEL_REFINEMENT = "label_refinement"
    EXPANSION_INDICATION = "expansion_indication"     # Likely requires new submission
    SITE_EXPANSION = "site_expansion"


class ChangeImpactLevel(str, Enum):
    """Risk level of a proposed model change."""
    MINOR = "minor"          # Within PCCP — auto-approved with testing
    MODERATE = "moderate"    # Within PCCP — requires human review
    MAJOR = "major"          # May exceed PCCP — escalate to regulatory team
    CRITICAL = "critical"    # Requires new 510(k) or supplemental submission


class ValidationStatus(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    UNDER_REVIEW = "under_review"
    PENDING_TESTING = "pending_testing"


# ---------------------------------------------------------------------------
# Allowed Change Specification
# ---------------------------------------------------------------------------

@dataclass
class AllowedChange:
    """
    Defines a pre-approved modification type within the PCCP.
    Specifies what changes are allowed, under what conditions,
    and what testing is required to confirm acceptability.
    """
    change_type: ChangeType
    description: str
    conditions: list[str] = field(default_factory=list)
    required_testing: list[str] = field(default_factory=list)
    performance_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    requires_human_review: bool = False
    requires_regulatory_notification: bool = False
    max_allowed_performance_degradation: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "change_type": self.change_type.value,
            "description": self.description,
            "conditions": self.conditions,
            "required_testing": self.required_testing,
            "performance_bounds": {
                k: list(v) for k, v in self.performance_bounds.items()
            },
            "requires_human_review": self.requires_human_review,
            "requires_regulatory_notification": self.requires_regulatory_notification,
            "max_allowed_performance_degradation": self.max_allowed_performance_degradation,
        }


@dataclass
class PCCPSpecification:
    """
    Complete PCCP specification for a medical AI device.
    Defines all allowed modifications and their validation requirements.
    """
    device_name: str
    device_version: str
    intended_use: str
    sponsor: str = ""
    submission_number: str = ""
    creation_date: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    allowed_changes: list[AllowedChange] = field(default_factory=list)
    locked_test_set_hash: str = ""      # SHA-256 of golden test set
    performance_floor: dict[str, float] = field(default_factory=dict)
    monitoring_protocol: dict[str, Any] = field(default_factory=dict)

    def get_allowed_change(self, change_type: ChangeType) -> AllowedChange | None:
        for ac in self.allowed_changes:
            if ac.change_type == change_type:
                return ac
        return None

    def to_dict(self) -> dict:
        return {
            "device_name": self.device_name,
            "device_version": self.device_version,
            "intended_use": self.intended_use,
            "sponsor": self.sponsor,
            "submission_number": self.submission_number,
            "creation_date": self.creation_date,
            "locked_test_set_hash": self.locked_test_set_hash,
            "performance_floor": self.performance_floor,
            "monitoring_protocol": self.monitoring_protocol,
            "allowed_changes": [ac.to_dict() for ac in self.allowed_changes],
        }

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load_json(cls, path: str | Path) -> "PCCPSpecification":
        data = json.loads(Path(path).read_text())
        spec = cls(
            device_name=data["device_name"],
            device_version=data["device_version"],
            intended_use=data["intended_use"],
            sponsor=data.get("sponsor", ""),
            submission_number=data.get("submission_number", ""),
            locked_test_set_hash=data.get("locked_test_set_hash", ""),
            performance_floor=data.get("performance_floor", {}),
        )
        for ac_data in data.get("allowed_changes", []):
            ac = AllowedChange(
                change_type=ChangeType(ac_data["change_type"]),
                description=ac_data["description"],
                conditions=ac_data.get("conditions", []),
                required_testing=ac_data.get("required_testing", []),
                max_allowed_performance_degradation=ac_data.get(
                    "max_allowed_performance_degradation", {}
                ),
                requires_human_review=ac_data.get("requires_human_review", False),
            )
            spec.allowed_changes.append(ac)
        return spec

    @classmethod
    def create_default(
        cls,
        device_name: str,
        intended_use: str,
        baseline_auroc: float = 0.90,
    ) -> "PCCPSpecification":
        """
        Create a sensible default PCCP for a binary classification medical AI model.
        Suitable as a starting template — customize thresholds before submission.
        """
        spec = cls(
            device_name=device_name,
            device_version="1.0",
            intended_use=intended_use,
            performance_floor={
                "auroc": max(0.75, baseline_auroc - 0.05),
                "sensitivity": 0.80,
                "specificity": 0.70,
            },
        )

        # Pre-approved changes
        spec.allowed_changes = [
            AllowedChange(
                change_type=ChangeType.RETRAINING_SAME_ARCHITECTURE,
                description=(
                    "Re-train existing model architecture on expanded dataset "
                    "from same sites/scanners with same label definitions."
                ),
                conditions=[
                    "New dataset must pass schema validation and drift check vs. original training data",
                    "New training data must be collected with same protocol as original",
                    "At least 500 new samples required",
                ],
                required_testing=[
                    "Regression test on golden test set (AUROC degradation < 2%)",
                    "Bias analysis on new training population demographics",
                    "Model card updated with new training data description",
                ],
                max_allowed_performance_degradation={"auroc": 0.02, "sensitivity": 0.03},
                requires_human_review=False,
                requires_regulatory_notification=False,
            ),
            AllowedChange(
                change_type=ChangeType.HYPERPARAMETER_ADJUSTMENT,
                description=(
                    "Adjust model hyperparameters (learning rate, regularization, "
                    "tree depth, etc.) without changing model architecture or training data."
                ),
                conditions=[
                    "Architecture and training data remain unchanged",
                    "Change must improve or maintain primary metric on validation set",
                ],
                required_testing=[
                    "Cross-validation on original training set",
                    "Regression test on golden test set",
                ],
                max_allowed_performance_degradation={"auroc": 0.01},
                requires_human_review=False,
                requires_regulatory_notification=False,
            ),
            AllowedChange(
                change_type=ChangeType.THRESHOLD_ADJUSTMENT,
                description=(
                    "Adjust the decision threshold (sensitivity/specificity operating point) "
                    "without retraining. Threshold must remain within the pre-validated ROC curve."
                ),
                conditions=[
                    "New threshold must lie on the existing validated ROC curve",
                    "Sensitivity must remain ≥ 0.80 for screening applications",
                    "Rationale for threshold change must be clinically motivated",
                ],
                required_testing=[
                    "Compute sensitivity/specificity at new threshold on golden test set",
                    "Update model card with new operating point",
                ],
                max_allowed_performance_degradation={},
                requires_human_review=True,
                requires_regulatory_notification=False,
            ),
            AllowedChange(
                change_type=ChangeType.RETRAINING_NEW_DATA,
                description=(
                    "Re-train with data from new sites, new scanner models, or new "
                    "patient demographics not in original training data."
                ),
                conditions=[
                    "New site data must be from same intended use population",
                    "New data must pass DICOM metadata validation if applicable",
                    "Bias analysis must show no new HIGH flags",
                ],
                required_testing=[
                    "Full evaluation on golden test set",
                    "Demographic bias analysis on new combined dataset",
                    "Multi-site performance breakdown",
                    "Regression test: AUROC degradation < 3%",
                ],
                max_allowed_performance_degradation={"auroc": 0.03},
                requires_human_review=True,
                requires_regulatory_notification=True,
            ),
            AllowedChange(
                change_type=ChangeType.PREPROCESSING_CHANGE,
                description=(
                    "Change preprocessing steps (normalization, augmentation, "
                    "image resizing) without changing training data or model architecture."
                ),
                conditions=[
                    "Change must be validated on at least 200 samples before full retraining",
                    "No new input requirements introduced for clinical users",
                ],
                required_testing=[
                    "Shadow testing on held-out validation set",
                    "Regression test on golden test set",
                ],
                max_allowed_performance_degradation={"auroc": 0.02},
                requires_human_review=True,
                requires_regulatory_notification=False,
            ),
        ]

        return spec


# ---------------------------------------------------------------------------
# Change Request
# ---------------------------------------------------------------------------

@dataclass
class PCCPChangeRequest:
    """A proposed model change submitted for PCCP validation."""
    change_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    change_type: ChangeType = ChangeType.RETRAINING_SAME_ARCHITECTURE
    description: str = ""
    requester: str = ""
    submitted_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    proposed_changes: dict[str, Any] = field(default_factory=dict)
    test_results: dict[str, Any] = field(default_factory=dict)
    baseline_metrics: dict[str, float] = field(default_factory=dict)
    new_metrics: dict[str, float] = field(default_factory=dict)
    justification: str = ""
    urgency: str = "routine"  # "routine" | "urgent" | "emergency"

    def to_dict(self) -> dict:
        return {k: (v.value if isinstance(v, Enum) else v)
                for k, v in self.__dict__.items()}


# ---------------------------------------------------------------------------
# Validation Report
# ---------------------------------------------------------------------------

@dataclass
class PCCPValidationReport:
    """Outcome of validating a change request against the PCCP."""
    change_request_id: str
    change_type: str
    status: ValidationStatus
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    within_pccp: bool = False
    impact_level: ChangeImpactLevel = ChangeImpactLevel.MINOR
    passed_checks: list[str] = field(default_factory=list)
    failed_checks: list[str] = field(default_factory=list)
    required_actions: list[str] = field(default_factory=list)
    regulatory_pathway: str = ""
    approval_notes: str = ""
    reviewer: str = ""

    def summary(self) -> str:
        lines = [
            f"=== PCCP Validation Report ===",
            f"Change ID      : {self.change_request_id}",
            f"Change Type    : {self.change_type}",
            f"Status         : {self.status.value.upper()}",
            f"Within PCCP    : {'YES' if self.within_pccp else 'NO'}",
            f"Impact Level   : {self.impact_level.value.upper()}",
            f"Regulatory Path: {self.regulatory_pathway}",
            "",
            f"Passed Checks  : {len(self.passed_checks)}",
            f"Failed Checks  : {len(self.failed_checks)}",
        ]
        for c in self.passed_checks:
            lines.append(f"  ✓ {c}")
        for c in self.failed_checks:
            lines.append(f"  ✗ {c}")
        if self.required_actions:
            lines += ["", "Required Actions:"]
            for a in self.required_actions:
                lines.append(f"  → {a}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "change_request_id": self.change_request_id,
            "change_type": self.change_type,
            "status": self.status.value,
            "timestamp": self.timestamp,
            "within_pccp": self.within_pccp,
            "impact_level": self.impact_level.value,
            "passed_checks": self.passed_checks,
            "failed_checks": self.failed_checks,
            "required_actions": self.required_actions,
            "regulatory_pathway": self.regulatory_pathway,
            "approval_notes": self.approval_notes,
        }

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))
        logger.info("PCCP validation report saved to %s", path)


# ---------------------------------------------------------------------------
# Core PCCP Manager
# ---------------------------------------------------------------------------

class PCCPManager:
    """
    Manages the Predetermined Change Control Plan workflow.

    Validates proposed model changes against pre-defined protocols,
    determines impact level, generates change control documentation,
    and decides whether regulatory notification is required.

    Example::

        # Load or create PCCP specification
        spec = PCCPSpecification.create_default(
            device_name="ChestXR-AI",
            intended_use="Detection of pulmonary nodules"
        )
        manager = PCCPManager(spec)

        # Propose a change
        change_request = PCCPChangeRequest(
            change_type=ChangeType.RETRAINING_SAME_ARCHITECTURE,
            description="Re-train with 2000 new cases from site B",
            baseline_metrics={"auroc": 0.92},
            new_metrics={"auroc": 0.93, "sensitivity": 0.88},
        )

        # Validate it
        report = manager.validate_change(change_request)
        print(report.summary())
    """

    def __init__(self, specification: PCCPSpecification):
        self.spec = specification
        self._change_log: list[dict] = []

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_change(self, request: PCCPChangeRequest) -> PCCPValidationReport:
        """
        Validate a proposed model change against the PCCP specification.

        Returns a PCCPValidationReport with status and required actions.
        """
        allowed = self.spec.get_allowed_change(request.change_type)

        report = PCCPValidationReport(
            change_request_id=request.change_id,
            change_type=request.change_type.value,
            status=ValidationStatus.UNDER_REVIEW,
        )

        if allowed is None:
            # Change type not in PCCP — requires new submission
            report.status = ValidationStatus.REJECTED
            report.within_pccp = False
            report.impact_level = ChangeImpactLevel.CRITICAL
            report.regulatory_pathway = "New 510(k) or supplemental submission required"
            report.failed_checks.append(
                f"Change type '{request.change_type.value}' not in PCCP specification"
            )
            report.required_actions = [
                "Contact regulatory team to assess submission requirements",
                "Consider expanding PCCP to cover this change type for future updates",
            ]
            logger.warning(
                "PCCP: Change type '%s' not in specification — rejection",
                request.change_type.value
            )
            self._log_change(request, report)
            return report

        report.within_pccp = True
        passed: list[str] = []
        failed: list[str] = []
        actions: list[str] = []

        # --- Performance degradation checks ---
        for metric, max_deg in allowed.max_allowed_performance_degradation.items():
            baseline = request.baseline_metrics.get(metric)
            new = request.new_metrics.get(metric)
            if baseline is not None and new is not None:
                delta = new - baseline
                if delta < -max_deg:
                    failed.append(
                        f"Performance degradation: {metric} dropped {abs(delta):.4f} "
                        f"(max allowed: {max_deg:.4f})"
                    )
                else:
                    passed.append(
                        f"Performance maintained: {metric} Δ={delta:+.4f} "
                        f"(threshold: -{max_deg:.4f})"
                    )

        # --- Absolute performance floor ---
        for metric, floor in self.spec.performance_floor.items():
            val = request.new_metrics.get(metric)
            if val is not None:
                if val < floor:
                    failed.append(
                        f"Performance floor violation: {metric}={val:.4f} < {floor:.4f}"
                    )
                else:
                    passed.append(f"Performance floor met: {metric}={val:.4f} ≥ {floor:.4f}")

        # --- Required testing checklist ---
        for test in allowed.required_testing:
            test_key = test.lower().replace(" ", "_")[:40]
            if test_key in request.test_results:
                test_result = request.test_results[test_key]
                if test_result.get("passed", False):
                    passed.append(f"Required test passed: {test}")
                else:
                    failed.append(
                        f"Required test failed: {test} ({test_result.get('message', '')})"
                    )
            else:
                actions.append(f"Complete required testing: {test}")

        # --- Condition verification (manual attestation) ---
        for condition in allowed.conditions:
            passed.append(f"[Condition — requires manual verification] {condition}")

        # --- Determine impact level ---
        if failed:
            impact = ChangeImpactLevel.MAJOR
        elif allowed.requires_human_review:
            impact = ChangeImpactLevel.MODERATE
        else:
            impact = ChangeImpactLevel.MINOR

        # --- Final status ---
        if failed:
            status = ValidationStatus.REJECTED
            reg_path = (
                "Performance degradation outside PCCP bounds — re-training or "
                "new submission required"
            )
        elif actions:
            status = ValidationStatus.PENDING_TESTING
            reg_path = "Pending completion of required testing"
        elif allowed.requires_regulatory_notification:
            status = ValidationStatus.APPROVED
            reg_path = "Regulatory notification required — submit PAS before deployment"
            actions.append("Submit Prior Approval Supplement (PAS) to FDA before deploying")
        elif allowed.requires_human_review:
            status = ValidationStatus.UNDER_REVIEW
            reg_path = "Internal review required before deployment"
        else:
            status = ValidationStatus.APPROVED
            reg_path = "No regulatory action required — change within PCCP bounds"

        report.status = status
        report.impact_level = impact
        report.regulatory_pathway = reg_path
        report.passed_checks = passed
        report.failed_checks = failed
        report.required_actions = actions

        logger.info(
            "PCCP validation: %s — status=%s, impact=%s",
            request.change_type.value, status.value, impact.value
        )
        self._log_change(request, report)
        return report

    # ------------------------------------------------------------------
    # Change Log
    # ------------------------------------------------------------------

    def _log_change(
        self, request: PCCPChangeRequest, report: PCCPValidationReport
    ) -> None:
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "change_id": request.change_id,
            "change_type": request.change_type.value,
            "requester": request.requester,
            "status": report.status.value,
            "within_pccp": report.within_pccp,
            "impact_level": report.impact_level.value,
            "description": request.description,
        }
        self._change_log.append(entry)

    def export_change_log(self, path: str | Path) -> None:
        """Export full change log for regulatory audit trail."""
        Path(path).write_text(json.dumps(self._change_log, indent=2))
        logger.info("PCCP change log exported to %s", path)

    def generate_change_report(
        self, request: PCCPChangeRequest, report: PCCPValidationReport
    ) -> str:
        """
        Generate a formatted change control report suitable for
        FDA submission documentation.
        """
        return f"""
PREDETERMINED CHANGE CONTROL PLAN — CHANGE REPORT
==================================================
Device:          {self.spec.device_name} v{self.spec.device_version}
Change ID:       {request.change_id}
Submitted:       {request.submitted_at}
Requester:       {request.requester}
Change Type:     {request.change_type.value}
Description:     {request.description}
Urgency:         {request.urgency}

VALIDATION RESULT
-----------------
Status:          {report.status.value.upper()}
Within PCCP:     {'YES' if report.within_pccp else 'NO'}
Impact Level:    {report.impact_level.value.upper()}
Regulatory Path: {report.regulatory_pathway}

PERFORMANCE SUMMARY
-------------------
{self._format_metrics_comparison(request.baseline_metrics, request.new_metrics)}

PASSED CHECKS ({len(report.passed_checks)})
{chr(10).join(f'  ✓ {c}' for c in report.passed_checks)}

FAILED CHECKS ({len(report.failed_checks)})
{chr(10).join(f'  ✗ {c}' for c in report.failed_checks) or '  None'}

REQUIRED ACTIONS
{chr(10).join(f'  → {a}' for a in report.required_actions) or '  None — proceed to deployment'}

CHANGE JUSTIFICATION
--------------------
{request.justification or '(Not provided)'}

---
Generated by MedML-Ops PCCP Manager | {datetime.utcnow().isoformat()}
PCCP Specification: {self.spec.device_name} | Submission: {self.spec.submission_number or 'Pending'}
"""

    @staticmethod
    def _format_metrics_comparison(
        baseline: dict[str, float], new: dict[str, float]
    ) -> str:
        if not baseline and not new:
            return "  No metrics provided"
        all_metrics = sorted(set(list(baseline.keys()) + list(new.keys())))
        lines = [f"  {'Metric':<20} {'Baseline':>10} {'New':>10} {'Delta':>10}"]
        lines.append("  " + "-" * 52)
        for m in all_metrics:
            b = baseline.get(m, float("nan"))
            n = new.get(m, float("nan"))
            d = n - b if not (b != b or n != n) else float("nan")
            d_str = f"{d:+.4f}" if d == d else "N/A"
            lines.append(f"  {m:<20} {b:>10.4f} {n:>10.4f} {d_str:>10}")
        return "\n".join(lines)
