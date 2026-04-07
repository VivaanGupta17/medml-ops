#!/usr/bin/env python3
"""
Compliance Report Generator
=============================
Generates a comprehensive compliance report from existing pipeline outputs.
Aggregates validation, evaluation, bias, GMLP audit, and model card data
into a single FDA submission-ready HTML document.

Usage:
    python scripts/generate_report.py
    python scripts/generate_report.py --reports-dir reports/ --output reports/compliance_summary.html
    python scripts/generate_report.py --run-id abc123 --mlflow-uri mlruns
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


def load_json_safe(path: str | Path) -> dict | None:
    """Load a JSON file, returning None if not found."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception as e:
        logger.warning("Could not load %s: %s", path, e)
        return None


def generate_html_report(
    reports_dir: Path,
    output_path: Path,
    experiment_name: str = "",
    run_id: str = "",
) -> None:
    """Generate a comprehensive HTML compliance report from pipeline outputs."""

    # Load available reports
    validation = load_json_safe(reports_dir / "validation_report.json")
    evaluation = load_json_safe(reports_dir / "evaluation_report.json")
    bias = load_json_safe(reports_dir / "bias_report.json")
    regression = load_json_safe(reports_dir / "regression_test_report.json")
    gmlp = load_json_safe(reports_dir.parent / "compliance" / "gmlp_audit.json")
    pipeline_summary = load_json_safe(reports_dir / "pipeline_summary.json")

    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    # ---------------------------------------------------------------------------
    # Section: Validation
    # ---------------------------------------------------------------------------
    if validation:
        val_status = "PASSED" if validation.get("passed") else "FAILED"
        val_color = "#16a34a" if validation.get("passed") else "#dc2626"
        val_section = f"""
        <div class="section">
          <h2>Data Validation</h2>
          <div class="status-badge" style="background:{val_color}20; border-color:{val_color}; color:{val_color}">
            {val_status}
          </div>
          <table>
            <tr><td>Errors</td><td>{validation.get('error_count', 0)}</td></tr>
            <tr><td>Warnings</td><td>{validation.get('warning_count', 0)}</td></tr>
            <tr><td>GMLP Principle</td><td>{validation.get('gmlp_principle', 'Principle 4')}</td></tr>
          </table>
        </div>"""
    else:
        val_section = '<div class="section"><h2>Data Validation</h2><p class="na">Not run</p></div>'

    # ---------------------------------------------------------------------------
    # Section: Evaluation
    # ---------------------------------------------------------------------------
    if evaluation:
        metrics = evaluation.get("core_metrics", {})
        ci = evaluation.get("confidence_intervals", {})
        auroc = metrics.get("auroc", 0)
        auroc_ci = ci.get("auroc", [None, None])
        ci_str = (f"[{auroc_ci[0]:.3f}, {auroc_ci[1]:.3f}]"
                  if all(v is not None for v in auroc_ci) else "")
        metric_rows = "".join(
            f"<tr><td>{k}</td><td>{v:.4f}</td></tr>"
            for k, v in metrics.items()
        )
        op = evaluation.get("primary_operating_point") or {}
        op_section = ""
        if op:
            op_section = f"""
            <h3>Primary Operating Point ({op.get('selection_method', '')})</h3>
            <table>
              <tr><td>Threshold</td><td>{op.get('threshold', 0):.3f}</td></tr>
              <tr><td>Sensitivity</td><td>{op.get('sensitivity', 0):.3f}</td></tr>
              <tr><td>Specificity</td><td>{op.get('specificity', 0):.3f}</td></tr>
              <tr><td>PPV</td><td>{op.get('ppv', 0):.3f}</td></tr>
              <tr><td>F1</td><td>{op.get('f1', 0):.3f}</td></tr>
            </table>"""
        eval_section = f"""
        <div class="section">
          <h2>Model Evaluation</h2>
          <p>AUROC: <strong>{auroc:.4f}</strong> {ci_str}</p>
          <table><tr><th>Metric</th><th>Value</th></tr>{metric_rows}</table>
          {op_section}
        </div>"""
    else:
        eval_section = '<div class="section"><h2>Model Evaluation</h2><p class="na">Not run</p></div>'

    # ---------------------------------------------------------------------------
    # Section: Bias Analysis
    # ---------------------------------------------------------------------------
    if bias:
        has_high = bias.get("has_high_bias_flags", False)
        bias_color = "#dc2626" if has_high else "#16a34a"
        bias_status = "HIGH BIAS FLAGS" if has_high else "PASSED"
        sg = bias.get("subgroup_metrics", [])
        sg_rows = "".join(
            f"<tr><td>{s.get('attribute')}</td><td>{s.get('group')}</td>"
            f"<td>{s.get('n_samples', 0):,}</td><td>{s.get('auroc') or 'N/A'}</td>"
            f"<td>{s.get('sensitivity', 0):.3f}</td><td>{s.get('specificity', 0):.3f}</td></tr>"
            for s in sg
        )
        bias_section = f"""
        <div class="section">
          <h2>Demographic Bias Analysis</h2>
          <div class="status-badge" style="background:{bias_color}20; border-color:{bias_color}; color:{bias_color}">
            {bias_status}
          </div>
          <table>
            <tr><th>Attribute</th><th>Group</th><th>N</th><th>AUROC</th>
                <th>Sensitivity</th><th>Specificity</th></tr>
            {sg_rows or '<tr><td colspan="6">No subgroup data</td></tr>'}
          </table>
        </div>"""
    else:
        bias_section = '<div class="section"><h2>Demographic Bias Analysis</h2><p class="na">Not run</p></div>'

    # ---------------------------------------------------------------------------
    # Section: GMLP Audit
    # ---------------------------------------------------------------------------
    if gmlp:
        score = gmlp.get("overall_score", 0)
        ready = gmlp.get("submission_ready", False)
        gmlp_color = "#16a34a" if ready else ("#d97706" if score >= 0.70 else "#dc2626")
        principle_rows = ""
        for p in gmlp.get("principles", []):
            status = p.get("status", "not-assessed")
            s_color = {"compliant": "#16a34a", "partial": "#d97706",
                       "non-compliant": "#dc2626"}.get(status, "#6b7280")
            principle_rows += (
                f'<tr><td>P{p.get("principle", 0):02d}</td>'
                f'<td>{p.get("title", "")}</td>'
                f'<td style="color:{s_color}; font-weight:bold;">{status.upper()}</td>'
                f'<td>{len(p.get("evidence", []))}</td></tr>'
            )
        gmlp_section = f"""
        <div class="section">
          <h2>GMLP Compliance Audit</h2>
          <div class="status-badge" style="background:{gmlp_color}20; border-color:{gmlp_color}; color:{gmlp_color}">
            {score*100:.1f}% — {'SUBMISSION READY' if ready else 'NOT YET READY'}
          </div>
          <table>
            <tr><th>#</th><th>Principle</th><th>Status</th><th>Evidence Items</th></tr>
            {principle_rows}
          </table>
        </div>"""
    else:
        gmlp_section = '<div class="section"><h2>GMLP Compliance Audit</h2><p class="na">Not run</p></div>'

    # ---------------------------------------------------------------------------
    # Section: Regression Testing
    # ---------------------------------------------------------------------------
    if regression:
        verdict = regression.get("verdict", "")
        reg_color = "#16a34a" if regression.get("passed") else "#dc2626"
        reg_section = f"""
        <div class="section">
          <h2>Regression Testing (PCCP)</h2>
          <div class="status-badge" style="background:{reg_color}20; border-color:{reg_color}; color:{reg_color}">
            {verdict}
          </div>
          <table>
            <tr><td>Baseline</td><td>{regression.get('baseline_model', '')}</td></tr>
            <tr><td>Candidate</td><td>{regression.get('new_model', '')}</td></tr>
            <tr><td>Critical Failures</td><td>{regression.get('critical_failures', 0)}</td></tr>
            <tr><td>Errors</td><td>{regression.get('errors', 0)}</td></tr>
            <tr><td>Warnings</td><td>{regression.get('warnings', 0)}</td></tr>
          </table>
        </div>"""
    else:
        reg_section = '<div class="section"><h2>Regression Testing</h2><p class="na">Not run / no baseline</p></div>'

    # ---------------------------------------------------------------------------
    # Assemble full HTML
    # ---------------------------------------------------------------------------
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>MedML-Ops Compliance Report — {experiment_name or 'Pipeline Run'}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
          max-width: 1100px; margin: 0 auto; padding: 24px; color: #111; }}
  h1 {{ color: #1e3a5f; border-bottom: 3px solid #1e3a5f; padding-bottom: 8px; }}
  h2 {{ color: #1e3a5f; margin-top: 0; }}
  h3 {{ color: #374151; margin-top: 16px; }}
  .section {{ background: #f8fafc; border: 1px solid #e5e7eb; border-radius: 8px;
              padding: 20px 24px; margin-bottom: 20px; }}
  .meta {{ color: #6b7280; font-size: 13px; margin-bottom: 24px; }}
  .status-badge {{ display: inline-block; padding: 6px 16px; border-radius: 6px;
                   border: 2px solid; font-weight: bold; font-size: 14px;
                   margin-bottom: 12px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 14px; margin-top: 8px; }}
  th {{ background: #1e3a5f; color: white; padding: 9px 12px; text-align: left; }}
  td {{ padding: 7px 12px; border-bottom: 1px solid #e5e7eb; }}
  tr:hover {{ background: #f1f5f9; }}
  .na {{ color: #9ca3af; font-style: italic; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  .fda-note {{ background: #eff6ff; border-left: 4px solid #3b82f6;
               padding: 12px 16px; color: #1e40af; font-size: 13px;
               margin: 16px 0; border-radius: 0 4px 4px 0; }}
  @media (max-width: 700px) {{ .grid {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>

<h1>MedML-Ops Compliance Report</h1>
<div class="meta">
  <strong>Experiment:</strong> {experiment_name or 'Pipeline Run'} &nbsp;|&nbsp;
  <strong>Run ID:</strong> {run_id or 'N/A'} &nbsp;|&nbsp;
  <strong>Generated:</strong> {timestamp}
</div>

<div class="fda-note">
  <strong>Regulatory Note:</strong> This report was generated by the MedML-Ops pipeline
  and documents compliance with FDA Good Machine Learning Practice (GMLP) guiding principles.
  It is intended to support — not replace — formal regulatory submissions.
</div>

<div class="grid">
  {val_section}
  {reg_section}
</div>

{eval_section}
{bias_section}
{gmlp_section}

<p class="meta" style="margin-top:32px; text-align:center;">
  Generated by MedML-Ops v1.0 | {timestamp}
</p>

</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
    logger.info("Compliance report saved to %s", output_path)
    print(f"Report generated: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate MedML-Ops compliance report")
    parser.add_argument("--reports-dir", default="reports/",
                        help="Directory containing pipeline output JSON reports")
    parser.add_argument("--output", default="reports/compliance_summary.html",
                        help="Output path for HTML report")
    parser.add_argument("--experiment-name", default="",
                        help="Experiment name for report header")
    parser.add_argument("--run-id", default="",
                        help="MLflow run ID")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    generate_html_report(
        reports_dir=Path(args.reports_dir),
        output_path=Path(args.output),
        experiment_name=args.experiment_name,
        run_id=args.run_id,
    )


if __name__ == "__main__":
    main()
