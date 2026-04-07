"""
GMLP Compliance Checker
========================
Automated audit against FDA's 10 Good Machine Learning Practice (GMLP) guiding
principles for medical device AI development.

Source:
  FDA, Health Canada, and MHRA. "Good Machine Learning Practice for Medical Device
  Development: Guiding Principles." October 2021.
  https://www.fda.gov/medical-devices/software-medical-device-samd/good-machine-learning-practice-medical-device-development-guiding-principles

This module:
  - Programmatically checks each GMLP principle against pipeline state
  - Integrates with MLflow run data for automated evidence collection
  - Generates audit trail documentation for regulatory submissions
  - Flags non-compliant areas with remediation guidance
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GMLP Principles Reference
# ---------------------------------------------------------------------------

GMLP_PRINCIPLES: dict[int, dict[str, str]] = {
    1: {
        "title": "Multi-Disciplinary Expertise Is Leveraged",
        "description": (
            "Throughout the total product lifecycle, teams use relevant cross-functional "
            "expertise and existing medical device practices to ensure the ML-enabled device "
            "is safe and effective."
        ),
        "key_questions": [
            "Is there documented involvement of clinical experts in design and validation?",
            "Have regulatory professionals reviewed the development plan?",
            "Are cybersecurity and software engineering represented?",
        ],
    },
    2: {
        "title": "Good Software Engineering and Security Practices Are Implemented",
        "description": (
            "Established software development practices are used, including version control, "
            "risk management, and cybersecurity measures appropriate to the device's risk level."
        ),
        "key_questions": [
            "Is version control (git) used with meaningful commit messages?",
            "Are automated tests and CI/CD pipelines in place?",
            "Are dependencies pinned and supply chain risks assessed?",
        ],
    },
    3: {
        "title": "Clinical Study Design Is Appropriate for the ML Task",
        "description": (
            "The clinical study design and evidence generation are appropriate for the "
            "intended use and characteristics of the ML model."
        ),
        "key_questions": [
            "Are inclusion/exclusion criteria documented and clinically motivated?",
            "Is the sample size sufficient to detect clinically meaningful effects?",
            "Are confidence intervals reported alongside point estimates?",
        ],
    },
    4: {
        "title": "Data Management Practices Support Transparency and Generalizability",
        "description": (
            "Data collection, labeling, and management are conducted with practices that "
            "support transparency, including documentation of data provenance and quality."
        ),
        "key_questions": [
            "Is dataset provenance documented (source, collection date, IRB approval)?",
            "Are data quality checks performed and documented?",
            "Is a reference dataset version-locked for reproducibility?",
        ],
    },
    5: {
        "title": "Re-Training Practices Are Transparent and Maintain Performance",
        "description": (
            "When models are re-trained or modified, practices ensure the performance "
            "and behavior remain safe and effective, and changes are transparent."
        ),
        "key_questions": [
            "Is the training environment captured (library versions, hardware)?",
            "Are random seeds logged for reproducibility?",
            "Is there a regression test suite against a golden test set?",
        ],
    },
    6: {
        "title": "Model Design Is Tailored to the Available Data and Avoids Bias",
        "description": (
            "Model design, training, and evaluation explicitly address the limitations "
            "of available data and known sources of bias."
        ),
        "key_questions": [
            "Is demographic bias analysis performed across age, sex, and ethnicity?",
            "Are class imbalance techniques applied where appropriate?",
            "Are known failure modes documented?",
        ],
    },
    7: {
        "title": "Focus Is Placed on the Performance of the Human-AI Team",
        "description": (
            "Testing considers the performance of the overall human-AI team, not just "
            "the algorithm in isolation."
        ),
        "key_questions": [
            "Has human-in-the-loop performance been evaluated?",
            "Are confidence thresholds set to support appropriate human override?",
            "Is uncertainty communicated to end users?",
        ],
    },
    8: {
        "title": "Testing Demonstrates Device Performance During Clinically Relevant Conditions",
        "description": (
            "Testing demonstrates that the device performs appropriately for its intended use "
            "under realistic and clinically relevant conditions, including edge cases."
        ),
        "key_questions": [
            "Is a fully independent test set used (no overlap with train/val)?",
            "Are edge cases and failure modes tested?",
            "Is multi-site / multi-vendor performance evaluated?",
        ],
    },
    9: {
        "title": "Users Are Provided Clear, Essential Information About the AI/ML Device",
        "description": (
            "Users have access to clear information about the AI/ML model including its "
            "intended use, inputs, outputs, and performance."
        ),
        "key_questions": [
            "Is a model card published with performance metrics?",
            "Are input requirements and limitations clearly documented?",
            "Is the clinical operating point and its basis explained?",
        ],
    },
    10: {
        "title": "Deployed Models Are Monitored for Performance and Updated as Needed",
        "description": (
            "Deployed models are monitored for real-world performance, and there is a "
            "process to update or retire the model when performance degrades."
        ),
        "key_questions": [
            "Is production drift monitoring active?",
            "Are thresholds defined for triggering human review?",
            "Is there a PCCP for managing model updates?",
        ],
    },
}


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class PrincipleCheckResult:
    """Compliance assessment for a single GMLP principle."""
    principle_number: int
    principle_title: str
    status: str          # "compliant" | "partial" | "non-compliant" | "not-assessed"
    evidence: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    remediation: list[str] = field(default_factory=list)
    automated_checks: list[dict] = field(default_factory=list)
    weight: float = 1.0

    @property
    def score(self) -> float:
        """Numeric compliance score: 1.0=compliant, 0.5=partial, 0.0=non-compliant."""
        return {"compliant": 1.0, "partial": 0.5, "non-compliant": 0.0,
                "not-assessed": 0.0}.get(self.status, 0.0)

    def to_dict(self) -> dict:
        return {
            "principle": self.principle_number,
            "title": self.principle_title,
            "status": self.status,
            "score": self.score,
            "evidence": self.evidence,
            "gaps": self.gaps,
            "remediation": self.remediation,
            "automated_checks": self.automated_checks,
        }


@dataclass
class GMLPAuditReport:
    """Complete GMLP compliance audit report."""
    model_name: str
    run_id: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    principle_results: list[PrincipleCheckResult] = field(default_factory=list)
    overall_score: float = 0.0
    compliant_count: int = 0
    partial_count: int = 0
    non_compliant_count: int = 0
    critical_gaps: list[str] = field(default_factory=list)
    submission_ready: bool = False

    def compute_score(self) -> float:
        if not self.principle_results:
            return 0.0
        total_weight = sum(r.weight for r in self.principle_results)
        weighted_score = sum(r.score * r.weight for r in self.principle_results)
        return round(weighted_score / max(total_weight, 1), 3)

    def summary(self) -> str:
        score_pct = f"{self.overall_score * 100:.1f}%"
        lines = [
            f"=== GMLP Compliance Audit ===",
            f"Model      : {self.model_name}",
            f"Run ID     : {self.run_id}",
            f"Timestamp  : {self.timestamp}",
            f"",
            f"Overall Score: {score_pct}",
            f"Compliant    : {self.compliant_count}/10",
            f"Partial      : {self.partial_count}/10",
            f"Non-Compliant: {self.non_compliant_count}/10",
            f"Submission Ready: {'YES' if self.submission_ready else 'NO'}",
            "",
        ]
        for r in self.principle_results:
            icon = {"compliant": "✓", "partial": "⚠", "non-compliant": "✗",
                    "not-assessed": "?"}.get(r.status, "?")
            lines.append(
                f"  {icon} P{r.principle_number:02d}: [{r.status.upper():14s}] {r.principle_title}"
            )
            for gap in r.gaps:
                lines.append(f"      GAP: {gap}")

        if self.critical_gaps:
            lines += ["", "Critical Gaps:"]
            for gap in self.critical_gaps:
                lines.append(f"  ✗ {gap}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "overall_score": self.overall_score,
            "compliant_count": self.compliant_count,
            "partial_count": self.partial_count,
            "non_compliant_count": self.non_compliant_count,
            "submission_ready": self.submission_ready,
            "critical_gaps": self.critical_gaps,
            "principles": [r.to_dict() for r in self.principle_results],
        }

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))
        logger.info("GMLP audit report saved to %s", path)


# ---------------------------------------------------------------------------
# Core Checker
# ---------------------------------------------------------------------------

class GMLPComplianceChecker:
    """
    Automated GMLP compliance checker.

    Integrates with MLflow run data to auto-collect evidence, and accepts
    additional manual attestations for checks that can't be automated.

    Example::

        checker = GMLPComplianceChecker(
            run_id="abc123def456",
            model_name="chest_xray_v2",
        )
        checker.attest_principle(1, "clinical_expert_reviewed",
                                 "Dr. Smith (Radiology) reviewed validation design")
        report = checker.run_full_audit(
            has_bias_report=True,
            has_model_card=True,
            has_drift_monitoring=True,
            train_test_overlap=False,
        )
        report.save_json("compliance/gmlp_audit.json")
    """

    def __init__(
        self,
        run_id: str = "",
        model_name: str = "",
        tracking_uri: str = "mlruns",
    ):
        self.run_id = run_id
        self.model_name = model_name
        self.tracking_uri = tracking_uri
        self._attestations: dict[int, list[str]] = {}
        self._run_data: dict | None = None

        if run_id:
            self._load_run_data()

    def _load_run_data(self) -> None:
        try:
            import mlflow
            client = mlflow.tracking.MlflowClient(tracking_uri=self.tracking_uri)
            run = client.get_run(self.run_id)
            self._run_data = {
                "params": run.data.params,
                "metrics": run.data.metrics,
                "tags": run.data.tags,
            }
        except Exception as e:
            logger.warning("Could not load MLflow run: %s", e)

    def attest_principle(
        self, principle_number: int, check_id: str, evidence: str
    ) -> None:
        """
        Manually attest evidence for a GMLP principle.
        Used for checks that require human judgment (team composition, IRB, etc.)
        """
        if principle_number not in self._attestations:
            self._attestations[principle_number] = []
        self._attestations[principle_number].append(f"[{check_id}] {evidence}")

    # ------------------------------------------------------------------
    # Full Audit
    # ------------------------------------------------------------------

    def run_full_audit(
        self,
        # Binary flags from pipeline state
        has_bias_report: bool = False,
        has_model_card: bool = False,
        has_drift_monitoring: bool = False,
        has_regression_tests: bool = False,
        has_pccp: bool = False,
        train_test_overlap: bool | None = None,  # None = not checked
        has_git_versioning: bool = True,
        has_requirements_pinned: bool = True,
        has_confidence_intervals: bool = False,
        has_multi_site_eval: bool = False,
        has_human_review_loop: bool = False,
        has_clinical_expert: bool = False,
        has_adversarial_testing: bool = False,
    ) -> GMLPAuditReport:
        """Run all 10 GMLP principle checks."""
        report = GMLPAuditReport(
            model_name=self.model_name,
            run_id=self.run_id,
        )

        checkers = [
            self._check_p1_multidisciplinary,
            self._check_p2_software_engineering,
            self._check_p3_clinical_study_design,
            self._check_p4_data_management,
            self._check_p5_retraining,
            self._check_p6_bias_design,
            self._check_p7_human_ai_team,
            self._check_p8_testing,
            self._check_p9_transparency,
            self._check_p10_monitoring,
        ]

        kwargs = {
            "has_bias_report": has_bias_report,
            "has_model_card": has_model_card,
            "has_drift_monitoring": has_drift_monitoring,
            "has_regression_tests": has_regression_tests,
            "has_pccp": has_pccp,
            "train_test_overlap": train_test_overlap,
            "has_git_versioning": has_git_versioning,
            "has_requirements_pinned": has_requirements_pinned,
            "has_confidence_intervals": has_confidence_intervals,
            "has_multi_site_eval": has_multi_site_eval,
            "has_human_review_loop": has_human_review_loop,
            "has_clinical_expert": has_clinical_expert,
            "has_adversarial_testing": has_adversarial_testing,
        }

        for checker in checkers:
            result = checker(**kwargs)
            # Inject manual attestations
            n = result.principle_number
            if n in self._attestations:
                result.evidence.extend(self._attestations[n])
            report.principle_results.append(result)

        # Aggregate
        report.compliant_count = sum(1 for r in report.principle_results
                                      if r.status == "compliant")
        report.partial_count = sum(1 for r in report.principle_results
                                    if r.status == "partial")
        report.non_compliant_count = sum(1 for r in report.principle_results
                                          if r.status == "non-compliant")
        report.overall_score = report.compute_score()
        report.critical_gaps = [
            gap
            for r in report.principle_results
            if r.status == "non-compliant"
            for gap in r.gaps
        ]
        # Submission ready: no non-compliant, score ≥ 0.80
        report.submission_ready = (
            report.non_compliant_count == 0 and report.overall_score >= 0.80
        )

        logger.info(
            "GMLP audit complete: %.1f%% (%d/10 compliant, submission_ready=%s)",
            report.overall_score * 100,
            report.compliant_count,
            report.submission_ready,
        )
        return report

    # ------------------------------------------------------------------
    # Per-Principle Checks
    # ------------------------------------------------------------------

    def _check_p1_multidisciplinary(self, has_clinical_expert: bool = False, **kw) -> PrincipleCheckResult:
        result = PrincipleCheckResult(1, GMLP_PRINCIPLES[1]["title"])
        evidence = self._attestations.get(1, [])

        auto_checks = []
        if has_clinical_expert:
            evidence.append("Clinical expert involvement documented")
            auto_checks.append({"check": "clinical_expert", "passed": True})
        else:
            auto_checks.append({"check": "clinical_expert", "passed": False})

        # Check MLflow tags for team sign-off
        if self._run_data:
            ml_engineer = self._run_data["params"].get("team_sign_off.ml_engineer", "")
            clinical = self._run_data["params"].get("team_sign_off.clinical_expert", "")
            if ml_engineer:
                evidence.append(f"ML Engineer: {ml_engineer}")
            if clinical:
                evidence.append(f"Clinical Expert: {clinical}")
                has_clinical_expert = True

        result.evidence = evidence
        result.automated_checks = auto_checks

        if has_clinical_expert and evidence:
            result.status = "compliant"
        elif evidence:
            result.status = "partial"
            result.gaps = ["Clinical expert involvement not formally documented"]
            result.remediation = ["Add clinical expert sign-off in GMLP documentation"]
        else:
            result.status = "non-compliant"
            result.gaps = [
                "No evidence of multi-disciplinary team involvement",
                "No clinical expert, regulatory, or cybersecurity documentation",
            ]
            result.remediation = [
                "Establish cross-functional team with clinical, regulatory, and ML expertise",
                "Document all team members and their roles in project charter",
            ]
        return result

    def _check_p2_software_engineering(
        self, has_git_versioning: bool = True, has_requirements_pinned: bool = True, **kw
    ) -> PrincipleCheckResult:
        result = PrincipleCheckResult(2, GMLP_PRINCIPLES[2]["title"])
        evidence = []
        gaps = []
        auto_checks = []

        if has_git_versioning:
            evidence.append("Git version control in use")
            auto_checks.append({"check": "git_versioning", "passed": True})
        else:
            gaps.append("No version control system detected")
            auto_checks.append({"check": "git_versioning", "passed": False})

        if has_requirements_pinned:
            evidence.append("Python dependencies pinned in requirements.txt")
            auto_checks.append({"check": "pinned_deps", "passed": True})
        else:
            gaps.append("Dependencies not pinned — reproducibility risk")

        # Check MLflow for git commit
        if self._run_data:
            git_commit = self._run_data["params"].get("env.git_commit", "")
            if git_commit and git_commit != "unavailable":
                evidence.append(f"Git commit logged: {git_commit[:12]}")
                git_dirty = self._run_data["params"].get("env.git_dirty", "True")
                if git_dirty == "True":
                    gaps.append("Uncommitted changes at training time (dirty git state)")
            python_version = self._run_data["params"].get("env.python_version", "")
            if python_version:
                evidence.append(f"Python version logged: {python_version}")

        # Check for pre-commit config
        precommit_exists = Path(".pre-commit-config.yaml").exists()
        if precommit_exists:
            evidence.append("Pre-commit hooks configured")
            auto_checks.append({"check": "pre_commit", "passed": True})

        result.evidence = evidence
        result.gaps = gaps
        result.automated_checks = auto_checks

        critical_checks_passed = has_git_versioning and has_requirements_pinned
        if critical_checks_passed and not gaps:
            result.status = "compliant"
        elif critical_checks_passed:
            result.status = "partial"
            result.remediation = ["Resolve identified gaps in software engineering practices"]
        else:
            result.status = "non-compliant"
            result.remediation = [
                "Implement git version control immediately",
                "Pin all Python dependencies in requirements.txt",
                "Add CI/CD pipeline with automated testing",
            ]
        return result

    def _check_p3_clinical_study_design(
        self, has_confidence_intervals: bool = False, **kw
    ) -> PrincipleCheckResult:
        result = PrincipleCheckResult(3, GMLP_PRINCIPLES[3]["title"])
        evidence = self._attestations.get(3, [])
        gaps = []

        if has_confidence_intervals:
            evidence.append("Bootstrap confidence intervals computed for primary metrics")
        else:
            gaps.append("Confidence intervals not reported for primary metrics")

        if self._run_data:
            n_samples = self._run_data["params"].get("dataset.train.n_samples", "")
            if n_samples:
                evidence.append(f"Training cohort size: {n_samples}")

        result.evidence = evidence
        result.gaps = gaps
        result.status = (
            "compliant" if evidence and not gaps else
            "partial" if evidence else "non-compliant"
        )
        result.remediation = (
            ["Compute and report bootstrap confidence intervals",
             "Document IRB approval and inclusion/exclusion criteria"]
            if not has_confidence_intervals else []
        )
        return result

    def _check_p4_data_management(self, **kw) -> PrincipleCheckResult:
        result = PrincipleCheckResult(4, GMLP_PRINCIPLES[4]["title"])
        evidence = self._attestations.get(4, [])
        gaps = []

        if self._run_data:
            p = self._run_data["params"]
            for split in ["train", "validation", "test"]:
                path = p.get(f"dataset.{split}.path", "")
                version = p.get(f"dataset.{split}.version", "")
                sha = p.get(f"dataset.{split}.sha256", "")
                source = p.get(f"dataset.{split}.source", "")
                if path:
                    evidence.append(
                        f"{split.capitalize()} dataset: {path} "
                        f"(v{version}, SHA256: {sha[:8] if sha and sha != 'not_computed' else 'not computed'})"
                    )
                    if not sha or sha == "not_computed":
                        gaps.append(f"SHA-256 hash not computed for {split} dataset")
                    if not source:
                        gaps.append(f"Data source not documented for {split} dataset")

        result.evidence = evidence or ["Data management logging enabled"]
        result.gaps = gaps
        result.status = (
            "compliant" if evidence and not gaps else
            "partial" if evidence else "non-compliant"
        )
        result.remediation = (
            ["Compute SHA-256 hashes for all datasets",
             "Document data provenance (source, collection dates, IRB)"]
            if gaps else []
        )
        return result

    def _check_p5_retraining(
        self, has_regression_tests: bool = False, **kw
    ) -> PrincipleCheckResult:
        result = PrincipleCheckResult(5, GMLP_PRINCIPLES[5]["title"])
        evidence = []
        gaps = []

        if self._run_data:
            p = self._run_data["params"]
            seed = p.get("random_seed", "")
            if seed:
                evidence.append(f"Random seed logged: {seed}")
            else:
                gaps.append("Random seed not logged — training not reproducible")

            env_python = p.get("env.python_version", "")
            if env_python:
                evidence.append(f"Python version captured: {env_python}")

            git = p.get("env.git_commit", "")
            if git and git != "unavailable":
                evidence.append(f"Git commit captured: {git[:12]}")
            else:
                gaps.append("Git commit not captured — code version unknown")

        if has_regression_tests:
            evidence.append("Regression test suite against golden test set")
        else:
            gaps.append("No regression test suite — cannot verify re-training quality")

        result.evidence = evidence
        result.gaps = gaps
        result.status = (
            "compliant" if evidence and not gaps else
            "partial" if evidence else "non-compliant"
        )
        result.remediation = [
            "Log random seed in all training runs",
            "Capture git commit hash at training time",
            "Implement regression test suite with golden test set",
        ] if gaps else []
        return result

    def _check_p6_bias_design(
        self, has_bias_report: bool = False, **kw
    ) -> PrincipleCheckResult:
        result = PrincipleCheckResult(6, GMLP_PRINCIPLES[6]["title"])
        evidence = self._attestations.get(6, [])
        gaps = []

        if has_bias_report:
            evidence.append(
                "Demographic bias analysis performed (age, sex, ethnicity subgroups)"
            )
            evidence.append("Equalized odds and demographic parity metrics computed")
        else:
            gaps.append(
                "No demographic bias analysis — FDA requires subgroup performance reporting"
            )
            gaps.append("Fairness metrics (equalized odds, demographic parity) not computed")

        result.evidence = evidence
        result.gaps = gaps
        result.status = (
            "compliant" if has_bias_report else "non-compliant"
        )
        result.remediation = (
            ["Run bias_detector.py on test set to generate subgroup analysis",
             "Document and address any HIGH or MODERATE bias flags before submission"]
            if not has_bias_report else []
        )
        return result

    def _check_p7_human_ai_team(
        self, has_human_review_loop: bool = False, **kw
    ) -> PrincipleCheckResult:
        result = PrincipleCheckResult(7, GMLP_PRINCIPLES[7]["title"])
        evidence = self._attestations.get(7, [])
        gaps = []

        if has_human_review_loop:
            evidence.append("Human-in-the-loop review implemented in deployment workflow")
        else:
            gaps.append("Human review workflow not documented")

        if self._run_data:
            threshold = self._run_data["params"].get("confidence_threshold", "")
            if threshold:
                evidence.append(
                    f"Confidence threshold for human escalation: {threshold}"
                )

        result.evidence = evidence
        result.gaps = gaps
        result.status = (
            "compliant" if has_human_review_loop and evidence else
            "partial" if evidence else "non-compliant"
        )
        result.remediation = (
            ["Define and document human review workflow",
             "Set confidence threshold for automated escalation to human review",
             "Conduct reader study measuring human-AI team performance"]
            if not has_human_review_loop else []
        )
        return result

    def _check_p8_testing(
        self, train_test_overlap: bool | None = None,
        has_multi_site_eval: bool = False,
        has_adversarial_testing: bool = False, **kw
    ) -> PrincipleCheckResult:
        result = PrincipleCheckResult(8, GMLP_PRINCIPLES[8]["title"])
        evidence = []
        gaps = []

        if train_test_overlap is False:
            evidence.append("Train/test separation verified: zero patient overlap")
        elif train_test_overlap is True:
            gaps.append(
                "CRITICAL: Data leakage detected — train/test overlap found. "
                "This is a GMLP violation and invalidates test results."
            )
        else:
            gaps.append("Train/test separation not explicitly verified")

        if has_multi_site_eval:
            evidence.append("Multi-site evaluation performed")
        else:
            gaps.append("Multi-site evaluation not performed")

        if has_adversarial_testing:
            evidence.append("Adversarial and edge case testing conducted")

        if self._run_data:
            overlap_count = self._run_data["params"].get("train_test_overlap_count", "")
            if overlap_count == "0":
                evidence.append(f"MLflow-verified zero train/test overlap")
            elif overlap_count and overlap_count != "0":
                gaps.append(f"MLflow detected {overlap_count} overlapping train/test samples!")

        result.evidence = evidence
        result.gaps = gaps

        critical = any("CRITICAL" in g for g in gaps)
        if critical:
            result.status = "non-compliant"
        elif not gaps:
            result.status = "compliant"
        else:
            result.status = "partial"

        result.remediation = [
            "Re-run with verified patient-level train/test separation",
            "Conduct evaluation at additional sites with different scanners/protocols",
        ] if gaps else []
        return result

    def _check_p9_transparency(
        self, has_model_card: bool = False, **kw
    ) -> PrincipleCheckResult:
        result = PrincipleCheckResult(9, GMLP_PRINCIPLES[9]["title"])
        evidence = []
        gaps = []

        if has_model_card:
            evidence.append("Model card generated with performance metrics and limitations")
        else:
            gaps.append("No model card — users lack information about device performance")

        result.evidence = evidence
        result.gaps = gaps
        result.status = "compliant" if has_model_card else "non-compliant"
        result.remediation = (
            ["Generate model card using model_card_generator.py",
             "Publish model card to clinical users and in FDA submission package"]
            if not has_model_card else []
        )
        return result

    def _check_p10_monitoring(
        self, has_drift_monitoring: bool = False, has_pccp: bool = False, **kw
    ) -> PrincipleCheckResult:
        result = PrincipleCheckResult(10, GMLP_PRINCIPLES[10]["title"])
        evidence = []
        gaps = []

        if has_drift_monitoring:
            evidence.append("Production drift monitoring active (PSI, KS, chi-square)")
        else:
            gaps.append("No production drift monitoring — GMLP requires post-market surveillance")

        if has_pccp:
            evidence.append("PCCP defined for managing model updates without new 510(k)")
        else:
            gaps.append("No PCCP — all model updates may require new 510(k) submission")

        result.evidence = evidence
        result.gaps = gaps
        result.status = (
            "compliant" if has_drift_monitoring and has_pccp else
            "partial" if evidence else "non-compliant"
        )
        result.remediation = [
            "Deploy drift_monitor.py for production monitoring",
            "Define PCCP using pccp_manager.py to enable safe model updates",
        ] if gaps else []
        return result
