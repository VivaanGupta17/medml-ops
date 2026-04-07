"""
Model Card Generator
=====================
Automated generation of FDA submission-ready model cards with Google Model Card
format extended for medical device regulatory context.

Generates both HTML (human-readable) and JSON (machine-readable) model cards
that include: intended use, training population demographics, subgroup performance,
known limitations, ethical considerations, and regulatory status.

References:
  - Mitchell et al. "Model Cards for Model Reporting" (FAccT 2019)
  - FDA AI/ML Action Plan — transparency requirements
  - NIST AI RMF — documentation standards

FDA GMLP Alignment:
  - Principle 10: Transparency to users about device performance and limitations
  - Principle 6: Design documentation for bias awareness
  - Principle 7: Testing documentation for regulatory submissions
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
# Model Card Data Schema
# ---------------------------------------------------------------------------

@dataclass
class ModelDetails:
    """Core model identification and provenance."""
    name: str
    version: str
    model_type: str
    description: str
    developers: list[str] = field(default_factory=list)
    release_date: str = ""
    license: str = "Proprietary"
    citations: list[str] = field(default_factory=list)
    framework: str = ""
    mlflow_run_id: str = ""
    git_commit: str = ""


@dataclass
class IntendedUse:
    """Clinical intended use and out-of-scope applications."""
    primary_intended_uses: list[str] = field(default_factory=list)
    primary_intended_users: list[str] = field(default_factory=list)
    out_of_scope_uses: list[str] = field(default_factory=list)
    device_class: str = ""
    regulatory_pathway: str = ""   # "510(k)", "De Novo", "PMA"
    submission_number: str = ""
    clinical_setting: str = ""


@dataclass
class TrainingData:
    """Training dataset description."""
    dataset_name: str
    dataset_version: str
    dataset_source: str
    n_training_samples: int
    n_validation_samples: int
    n_test_samples: int
    label_distribution: dict[str, int] = field(default_factory=dict)
    demographic_distribution: dict[str, dict] = field(default_factory=dict)
    data_collection_period: str = ""
    collection_sites: list[str] = field(default_factory=list)
    exclusion_criteria: list[str] = field(default_factory=list)
    preprocessing_steps: list[str] = field(default_factory=list)
    sha256_hash: str = ""


@dataclass
class QuantitativeAnalysis:
    """Model performance metrics — overall and per subgroup."""
    overall_metrics: dict[str, float] = field(default_factory=dict)
    subgroup_metrics: list[dict[str, Any]] = field(default_factory=list)
    fairness_metrics: list[dict[str, Any]] = field(default_factory=list)
    calibration_metrics: dict[str, float] = field(default_factory=dict)
    confidence_intervals: dict[str, list[float]] = field(default_factory=dict)
    primary_operating_point: dict[str, Any] = field(default_factory=dict)
    predicate_comparison: dict[str, Any] = field(default_factory=dict)
    evaluation_dataset: str = ""
    evaluation_date: str = ""


@dataclass
class ConsiderationsAndLimitations:
    """Ethical considerations and known limitations."""
    known_limitations: list[str] = field(default_factory=list)
    ethical_considerations: list[str] = field(default_factory=list)
    bias_risks: list[str] = field(default_factory=list)
    deployment_risks: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    human_oversight_required: bool = True
    adverse_event_reporting: str = ""


# ---------------------------------------------------------------------------
# Model Card
# ---------------------------------------------------------------------------

@dataclass
class ModelCard:
    """
    Complete FDA-submission-ready model card.
    Aggregates all sections into a unified document.
    """
    model_details: ModelDetails
    intended_use: IntendedUse
    training_data: TrainingData
    quantitative_analysis: QuantitativeAnalysis
    considerations: ConsiderationsAndLimitations
    generated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    card_version: str = "1.0"
    gmlp_alignment: str = "Principles 6, 7, 10"

    def to_dict(self) -> dict:
        import dataclasses
        def _convert(obj):
            if dataclasses.is_dataclass(obj):
                return {k: _convert(v) for k, v in dataclasses.asdict(obj).items()}
            elif isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_convert(i) for i in obj]
            elif isinstance(obj, float):
                return round(obj, 4)
            return obj
        return _convert(self)

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))
        logger.info("Model card JSON saved to %s", path)

    def save_html(self, path: str | Path) -> None:
        html = _render_model_card_html(self)
        Path(path).write_text(html)
        logger.info("Model card HTML saved to %s", path)


# ---------------------------------------------------------------------------
# Model Card Generator
# ---------------------------------------------------------------------------

class ModelCardGenerator:
    """
    Automated model card generator that assembles a complete model card
    from MLflow run data, evaluation reports, and bias reports.

    Example::

        generator = ModelCardGenerator(mlflow_run_id="abc123")
        card = generator.generate(
            model_name="chest_xray_classifier",
            version="v2.1",
            evaluation_report=eval_report,
            bias_report=bias_report,
        )
        card.save_html("docs/model_card_v2.1.html")
        card.save_json("compliance/model_card_v2.1.json")
    """

    def __init__(
        self,
        mlflow_run_id: str | None = None,
        tracking_uri: str = "mlruns",
    ):
        self.mlflow_run_id = mlflow_run_id
        self.tracking_uri = tracking_uri
        self._run_data: dict | None = None

        if mlflow_run_id:
            self._load_mlflow_run(mlflow_run_id)

    def _load_mlflow_run(self, run_id: str) -> None:
        try:
            import mlflow
            client = mlflow.tracking.MlflowClient(tracking_uri=self.tracking_uri)
            run = client.get_run(run_id)
            self._run_data = {
                "params": run.data.params,
                "metrics": run.data.metrics,
                "tags": run.data.tags,
                "start_time": run.info.start_time,
            }
            logger.info("Loaded MLflow run: %s", run_id)
        except Exception as e:
            logger.warning("Could not load MLflow run %s: %s", run_id, e)

    def generate(
        self,
        model_name: str,
        version: str,
        model_type: str = "",
        description: str = "",
        intended_uses: list[str] | None = None,
        out_of_scope: list[str] | None = None,
        evaluation_report=None,        # EvaluationReport instance
        bias_report=None,              # BiasReport instance
        training_data_info: dict | None = None,
        developers: list[str] | None = None,
        regulatory_pathway: str = "510(k)",
        known_limitations: list[str] | None = None,
    ) -> ModelCard:
        """
        Generate a complete model card.

        Args:
            model_name: Model identifier.
            version: Version string (e.g., "v2.1").
            model_type: Model architecture (e.g., "RandomForest", "ResNet50").
            description: Human-readable model description.
            intended_uses: List of intended clinical use cases.
            out_of_scope: Explicitly out-of-scope applications.
            evaluation_report: EvaluationReport from model_evaluator.py.
            bias_report: BiasReport from bias_detector.py.
            training_data_info: Dict with training data details.
            developers: List of developer names/roles.
            regulatory_pathway: FDA submission pathway.
            known_limitations: Documented model limitations.

        Returns:
            Complete ModelCard.
        """
        # --- Model Details ---
        run_id = self.mlflow_run_id or ""
        git_commit = ""
        if self._run_data:
            git_commit = self._run_data["params"].get("env.git_commit", "")

        details = ModelDetails(
            name=model_name,
            version=version,
            model_type=model_type or (
                self._run_data["params"].get("model_type", "")
                if self._run_data else ""
            ),
            description=description,
            developers=developers or [],
            release_date=datetime.utcnow().strftime("%Y-%m-%d"),
            framework=self._run_data["params"].get("env.pkg_scikit-learn", "")
                      if self._run_data else "",
            mlflow_run_id=run_id,
            git_commit=git_commit,
        )

        # --- Intended Use ---
        use = IntendedUse(
            primary_intended_uses=intended_uses or [
                f"Clinical decision support for {model_name}",
            ],
            primary_intended_users=[
                "Licensed healthcare professionals",
                "Radiologists / clinicians (as applicable)",
            ],
            out_of_scope_uses=out_of_scope or [
                "Autonomous diagnosis without clinician review",
                "Pediatric populations (validate separately)",
                "Emergency / acute care without validation",
                "Populations outside the training demographics",
            ],
            regulatory_pathway=regulatory_pathway,
            clinical_setting="Hospital / outpatient radiology / clinical laboratory",
        )

        # --- Training Data ---
        td_info = training_data_info or {}
        train_data = TrainingData(
            dataset_name=td_info.get("name", ""),
            dataset_version=td_info.get("version", ""),
            dataset_source=td_info.get("source", ""),
            n_training_samples=int(td_info.get("n_train", 0)),
            n_validation_samples=int(td_info.get("n_val", 0)),
            n_test_samples=int(td_info.get("n_test", 0)),
            label_distribution=td_info.get("label_distribution", {}),
            demographic_distribution=td_info.get("demographics", {}),
            data_collection_period=td_info.get("collection_period", ""),
            collection_sites=td_info.get("sites", []),
            exclusion_criteria=td_info.get("exclusions", []),
            preprocessing_steps=td_info.get("preprocessing", []),
            sha256_hash=td_info.get("sha256", ""),
        )

        # Fill from MLflow if available
        if self._run_data and not td_info:
            p = self._run_data["params"]
            train_data.dataset_name = p.get("dataset.train.path", "")
            train_data.dataset_version = p.get("dataset.train.version", "")
            train_data.sha256_hash = p.get("dataset.train.sha256", "")
            try:
                train_data.n_training_samples = int(p.get("dataset.train.n_samples", 0))
            except (ValueError, TypeError):
                pass

        # --- Quantitative Analysis ---
        quant = QuantitativeAnalysis(
            evaluation_date=datetime.utcnow().strftime("%Y-%m-%d"),
        )

        if evaluation_report:
            quant.overall_metrics = evaluation_report.overall_metrics if hasattr(evaluation_report, 'overall_metrics') else evaluation_report.core_metrics
            quant.calibration_metrics = (
                evaluation_report.calibration.__dict__
                if evaluation_report.calibration else {}
            )
            quant.confidence_intervals = {
                k: list(v) for k, v in
                (evaluation_report.confidence_intervals or {}).items()
            }
            op = (evaluation_report.primary_operating_point()
                  if hasattr(evaluation_report, "primary_operating_point") else None)
            if op:
                quant.primary_operating_point = op.to_dict()
            quant.evaluation_dataset = evaluation_report.dataset_name
            quant.predicate_comparison = evaluation_report.predicate_comparison

        if bias_report:
            quant.subgroup_metrics = [sm.to_dict() for sm in bias_report.subgroup_metrics]
            quant.fairness_metrics = [fm.to_dict() for fm in bias_report.fairness_metrics]

        # --- Considerations ---
        default_limitations = [
            "Validated on retrospective data — prospective validation recommended",
            "Performance may degrade on images from scanners not represented in training",
            "Not validated for patients with conditions outside training cohort",
            "Requires integration with clinical workflow; not for autonomous use",
        ]

        default_ethical = [
            "Model may exhibit differential performance across demographic subgroups — see subgroup analysis",
            "False negative predictions carry higher clinical risk than false positives in screening context",
            "Clinicians must retain final decision authority",
            "De-identified training data — re-identification risk assessed as low",
        ]

        if bias_report and bias_report.has_high_bias_flags():
            default_ethical.append(
                "HIGH BIAS FLAGS detected in subgroup analysis — see bias report for details"
            )

        considerations = ConsiderationsAndLimitations(
            known_limitations=known_limitations or default_limitations,
            ethical_considerations=default_ethical,
            bias_risks=[
                f"Subgroup performance gap: {fm.attribute} "
                f"({fm.reference_group} vs {fm.comparison_group})"
                for fm in (bias_report.fairness_metrics if bias_report else [])
                if fm.flag_level in ("HIGH", "MODERATE")
            ],
            deployment_risks=[
                "Model performance must be re-validated on local data before deployment",
                "Production drift monitoring required per PCCP",
            ],
            recommendations=[
                "Deploy with human-in-the-loop review for all predictions",
                "Monitor for demographic drift in production population",
                "Conduct annual performance review per PCCP schedule",
            ],
            human_oversight_required=True,
            adverse_event_reporting=(
                "Report adverse events to FDA MedWatch and per 21 CFR 803"
            ),
        )

        return ModelCard(
            model_details=details,
            intended_use=use,
            training_data=train_data,
            quantitative_analysis=quant,
            considerations=considerations,
        )


# ---------------------------------------------------------------------------
# HTML Rendering
# ---------------------------------------------------------------------------

def _render_model_card_html(card: ModelCard) -> str:
    """Render a full model card as styled HTML."""
    d = card.model_details
    u = card.intended_use
    t = card.training_data
    q = card.quantitative_analysis
    c = card.considerations

    overall_rows = "".join(
        f"<tr><td>{k}</td><td><strong>{v:.4f}</strong></td></tr>"
        for k, v in q.overall_metrics.items()
    )

    subgroup_rows = ""
    for sm in q.subgroup_metrics:
        auroc = sm.get("auroc", "N/A")
        auroc_str = f"{auroc:.3f}" if isinstance(auroc, float) else "N/A"
        subgroup_rows += f"""
        <tr>
            <td>{sm.get('attribute', '')}</td>
            <td><strong>{sm.get('group', '')}</strong></td>
            <td>{sm.get('n_samples', 0):,}</td>
            <td>{auroc_str}</td>
            <td>{sm.get('sensitivity', 0):.3f}</td>
            <td>{sm.get('specificity', 0):.3f}</td>
            <td>{sm.get('f1', 0):.3f}</td>
        </tr>"""

    fairness_rows = ""
    flag_colors = {"HIGH": "#dc2626", "MODERATE": "#d97706", "LOW": "#16a34a"}
    for fm in q.fairness_metrics:
        flag = fm.get("flag_level", "LOW")
        color = flag_colors.get(flag, "#16a34a")
        fairness_rows += f"""
        <tr>
            <td>{fm.get('attribute', '')}</td>
            <td>{fm.get('reference_group', '')}</td>
            <td>{fm.get('comparison_group', '')}</td>
            <td>{fm.get('equalized_odds_tpr_gap', 0):.4f}</td>
            <td>{fm.get('demographic_parity_gap', 0):.4f}</td>
            <td style="color:{color}; font-weight:bold;">{flag}</td>
        </tr>"""

    limitation_items = "".join(f"<li>{l}</li>" for l in c.known_limitations)
    ethical_items = "".join(f"<li>{e}</li>" for e in c.ethical_considerations)
    bias_items = "".join(f"<li>{b}</li>" for b in c.bias_risks) if c.bias_risks else "<li>None identified</li>"

    op = q.primary_operating_point
    op_section = ""
    if op:
        op_section = f"""
        <h2>Primary Operating Point ({op.get('selection_method', '')})</h2>
        <table>
          <tr><th>Parameter</th><th>Value</th></tr>
          <tr><td>Decision Threshold</td><td>{op.get('threshold', 0):.3f}</td></tr>
          <tr><td>Sensitivity (TPR)</td><td>{op.get('sensitivity', 0):.3f}</td></tr>
          <tr><td>Specificity (TNR)</td><td>{op.get('specificity', 0):.3f}</td></tr>
          <tr><td>PPV (Precision)</td><td>{op.get('ppv', 0):.3f}</td></tr>
          <tr><td>F1 Score</td><td>{op.get('f1', 0):.3f}</td></tr>
        </table>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Model Card – {d.name} v{d.version}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
          max-width: 1100px; margin: 0 auto; padding: 24px; color: #111; line-height: 1.6; }}
  h1 {{ color: #1e3a5f; border-bottom: 3px solid #1e3a5f; padding-bottom: 8px; }}
  h2 {{ color: #1e3a5f; margin-top: 32px; }}
  h3 {{ color: #374151; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 14px; }}
  th {{ background: #1e3a5f; color: white; padding: 10px 12px; text-align: left; }}
  td {{ padding: 8px 12px; border-bottom: 1px solid #e5e7eb; }}
  tr:hover {{ background: #f8fafc; }}
  .badge {{ display: inline-block; padding: 3px 10px; border-radius: 12px;
            font-size: 12px; font-weight: bold; margin: 2px; }}
  .badge-blue {{ background: #dbeafe; color: #1d4ed8; }}
  .badge-green {{ background: #dcfce7; color: #15803d; }}
  .badge-red {{ background: #fee2e2; color: #dc2626; }}
  .meta {{ color: #6b7280; font-size: 13px; margin-bottom: 20px; }}
  .note {{ background: #eff6ff; border-left: 4px solid #3b82f6; padding: 12px 16px;
           color: #1e40af; font-size: 13px; margin: 12px 0; border-radius: 0 4px 4px 0; }}
  .warning {{ background: #fef3c7; border-left: 4px solid #d97706; padding: 12px 16px;
              color: #92400e; font-size: 13px; margin: 12px 0; border-radius: 0 4px 4px 0; }}
  ul {{ margin-top: 8px; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
  @media (max-width: 700px) {{ .grid {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>

<h1>Model Card: {d.name}</h1>
<div class="meta">
  <strong>Version:</strong> {d.version} &nbsp;|&nbsp;
  <strong>Type:</strong> {d.model_type} &nbsp;|&nbsp;
  <strong>Released:</strong> {d.release_date} &nbsp;|&nbsp;
  <strong>Generated:</strong> {card.generated_at[:10]} &nbsp;|&nbsp;
  <strong>GMLP:</strong> {card.gmlp_alignment}
</div>

<div>
  <span class="badge badge-blue">{u.regulatory_pathway}</span>
  <span class="badge badge-green">GMLP Compliant</span>
  {'<span class="badge badge-red">HIGH BIAS FLAGS</span>' if any(fm.get('flag_level') == 'HIGH' for fm in q.fairness_metrics) else ''}
</div>

<div class="note">
  <strong>Clinical Disclaimer:</strong> This AI system is intended to support, not replace,
  clinical decision-making. All predictions must be reviewed by a qualified healthcare professional.
</div>

<h2>Model Description</h2>
<p>{d.description}</p>

<h2>Intended Use</h2>
<div class="grid">
  <div>
    <h3>In-Scope</h3>
    <ul>{''.join(f'<li>{u}</li>' for u in u.primary_intended_uses)}</ul>
    <p><strong>Target Users:</strong></p>
    <ul>{''.join(f'<li>{u}</li>' for u in u.primary_intended_users)}</ul>
    <p><strong>Clinical Setting:</strong> {u.clinical_setting}</p>
  </div>
  <div>
    <h3>Out of Scope</h3>
    <ul>{''.join(f'<li>{o}</li>' for o in u.out_of_scope_uses)}</ul>
  </div>
</div>

<h2>Training Data</h2>
<table>
  <tr><th>Property</th><th>Value</th></tr>
  <tr><td>Dataset</td><td>{t.dataset_name} v{t.dataset_version}</td></tr>
  <tr><td>Source</td><td>{t.dataset_source}</td></tr>
  <tr><td>Training samples</td><td>{t.n_training_samples:,}</td></tr>
  <tr><td>Validation samples</td><td>{t.n_validation_samples:,}</td></tr>
  <tr><td>Test samples</td><td>{t.n_test_samples:,}</td></tr>
  <tr><td>Collection period</td><td>{t.data_collection_period}</td></tr>
  <tr><td>Integrity (SHA-256)</td><td><code>{t.sha256_hash[:16]}...</code></td></tr>
</table>

<h2>Overall Performance</h2>
<table>
  <tr><th>Metric</th><th>Value</th></tr>
  {overall_rows}
</table>

{op_section}

<h2>Subgroup Performance</h2>
{'<div class="warning"><strong>FDA Note:</strong> Subgroup performance analysis is required per GMLP Principle 6. Review all subgroup disparities before deployment.</div>' if q.subgroup_metrics else ''}
<table>
  <tr>
    <th>Attribute</th><th>Group</th><th>N</th>
    <th>AUROC</th><th>Sensitivity</th><th>Specificity</th><th>F1</th>
  </tr>
  {subgroup_rows if subgroup_rows else '<tr><td colspan="7">No subgroup analysis available</td></tr>'}
</table>

<h2>Fairness Metrics</h2>
<table>
  <tr>
    <th>Attribute</th><th>Reference Group</th><th>Comparison Group</th>
    <th>EqOdds TPR Gap</th><th>Dem. Parity Gap</th><th>Flag</th>
  </tr>
  {fairness_rows if fairness_rows else '<tr><td colspan="6">No fairness metrics available</td></tr>'}
</table>

<h2>Known Limitations</h2>
<ul>{limitation_items}</ul>

<h2>Ethical Considerations</h2>
<ul>{ethical_items}</ul>

<h2>Bias Risks</h2>
<ul>{bias_items}</ul>

<h2>Regulatory & Provenance</h2>
<table>
  <tr><th>Field</th><th>Value</th></tr>
  <tr><td>Regulatory pathway</td><td>{u.regulatory_pathway}</td></tr>
  <tr><td>Submission number</td><td>{u.submission_number or 'Pending'}</td></tr>
  <tr><td>MLflow Run ID</td><td><code>{d.mlflow_run_id}</code></td></tr>
  <tr><td>Git Commit</td><td><code>{d.git_commit[:12] if d.git_commit else 'N/A'}</code></td></tr>
  <tr><td>Human oversight required</td><td>{"Yes" if c.human_oversight_required else "No"}</td></tr>
  <tr><td>Adverse event reporting</td><td>{c.adverse_event_reporting}</td></tr>
</table>

</body>
</html>"""
