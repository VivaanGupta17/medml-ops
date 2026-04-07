"""
Medical Dataset Schema Validator
=================================
Great Expectations-style validation for medical datasets with DICOM metadata
support, data drift detection, and FDA GMLP-aligned documentation.

FDA GMLP Alignment:
  - Principle 4: Data management — validates data quality, provenance, schema
  - Principle 9: Evaluation reflects clinical deployment — checks for covariate shift
  - Principle 10: Post-market monitoring — drift detection between train and production
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

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """Result of a single validation check."""
    check_name: str
    passed: bool
    severity: str  # "error" | "warning" | "info"
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    """Aggregated report from a full dataset validation run."""
    dataset_path: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    results: list[ValidationResult] = field(default_factory=list)
    schema_version: str = "1.0"
    gmlp_principle: str = "Principle 4 – Data Management"

    @property
    def passed(self) -> bool:
        return all(r.passed or r.severity != "error" for r in self.results)

    @property
    def error_count(self) -> int:
        return sum(1 for r in self.results if not r.passed and r.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for r in self.results if not r.passed and r.severity == "warning")

    def summary(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        lines = [
            f"=== Validation Report [{status}] ===",
            f"Dataset    : {self.dataset_path}",
            f"Timestamp  : {self.timestamp}",
            f"Errors     : {self.error_count}",
            f"Warnings   : {self.warning_count}",
            "",
        ]
        for r in self.results:
            icon = "✓" if r.passed else ("✗" if r.severity == "error" else "⚠")
            lines.append(f"  {icon} [{r.severity.upper():7s}] {r.check_name}: {r.message}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "dataset_path": self.dataset_path,
            "timestamp": self.timestamp,
            "passed": self.passed,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "gmlp_principle": self.gmlp_principle,
            "results": [
                {
                    "check_name": r.check_name,
                    "passed": r.passed,
                    "severity": r.severity,
                    "message": r.message,
                    "details": r.details,
                }
                for r in self.results
            ],
        }

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))
        logger.info("Validation report saved to %s", path)


# ---------------------------------------------------------------------------
# Schema Enforcement
# ---------------------------------------------------------------------------

class MedicalSchemaSpec:
    """
    Defines expected schema for a medical dataset.

    Example usage::

        spec = MedicalSchemaSpec(
            required_columns=["patient_id", "age", "sex", "label"],
            column_dtypes={"age": "float64", "sex": "category", "label": "int64"},
            column_ranges={"age": (0, 120)},
            categorical_values={"sex": ["M", "F", "Unknown"]},
            max_missing_fraction={"age": 0.02, "label": 0.0},
        )
    """

    def __init__(
        self,
        required_columns: list[str] | None = None,
        column_dtypes: dict[str, str] | None = None,
        column_ranges: dict[str, tuple[float, float]] | None = None,
        categorical_values: dict[str, list[str]] | None = None,
        max_missing_fraction: dict[str, float] | None = None,
        unique_key_columns: list[str] | None = None,
    ):
        self.required_columns = required_columns or []
        self.column_dtypes = column_dtypes or {}
        self.column_ranges = column_ranges or {}
        self.categorical_values = categorical_values or {}
        self.max_missing_fraction = max_missing_fraction or {}
        self.unique_key_columns = unique_key_columns or []

    @classmethod
    def from_json(cls, path: str | Path) -> "MedicalSchemaSpec":
        data = json.loads(Path(path).read_text())
        return cls(**data)

    def to_json(self, path: str | Path) -> None:
        spec = {
            "required_columns": self.required_columns,
            "column_dtypes": self.column_dtypes,
            "column_ranges": self.column_ranges,
            "categorical_values": self.categorical_values,
            "max_missing_fraction": self.max_missing_fraction,
            "unique_key_columns": self.unique_key_columns,
        }
        Path(path).write_text(json.dumps(spec, indent=2))


# ---------------------------------------------------------------------------
# Core Validator
# ---------------------------------------------------------------------------

class MedicalDataValidator:
    """
    Comprehensive data validator for medical ML datasets.

    Performs:
    - Schema enforcement (required columns, types, ranges)
    - Missing value and outlier detection
    - DICOM metadata validation (when applicable)
    - Data drift detection between reference and current datasets
    - Demographic completeness checks (for bias analysis readiness)

    GMLP Mapping:
        Principle 4: Validates data quality, completeness, and provenance.
        Principle 9: Detects covariate shift between train and deployment data.
    """

    def __init__(
        self,
        schema: MedicalSchemaSpec | None = None,
        config_path: str | Path | None = None,
        outlier_z_threshold: float = 3.5,
        drift_p_value_threshold: float = 0.05,
    ):
        self.schema = schema
        self.outlier_z_threshold = outlier_z_threshold
        self.drift_p_value_threshold = drift_p_value_threshold

        if config_path:
            self._load_config(config_path)

    def _load_config(self, config_path: str | Path) -> None:
        import yaml  # lazy import

        cfg = yaml.safe_load(Path(config_path).read_text())
        validation_cfg = cfg.get("data_validation", {})
        schema_cfg = validation_cfg.get("schema", {})
        if schema_cfg:
            self.schema = MedicalSchemaSpec(**schema_cfg)
        self.outlier_z_threshold = validation_cfg.get(
            "outlier_z_threshold", self.outlier_z_threshold
        )
        self.drift_p_value_threshold = validation_cfg.get(
            "drift_p_value_threshold", self.drift_p_value_threshold
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate_dataset(
        self,
        data_path: str | Path | pd.DataFrame,
        reference_path: str | Path | pd.DataFrame | None = None,
    ) -> ValidationReport:
        """
        Run full validation suite on a dataset.

        Args:
            data_path: Path to CSV/Parquet, or a DataFrame directly.
            reference_path: Reference dataset for drift detection (e.g., training set).

        Returns:
            ValidationReport with all check results.
        """
        df = self._load(data_path)
        report = ValidationReport(dataset_path=str(data_path))

        # Core schema checks
        if self.schema:
            report.results.extend(self._check_required_columns(df))
            report.results.extend(self._check_dtypes(df))
            report.results.extend(self._check_ranges(df))
            report.results.extend(self._check_categoricals(df))
            report.results.extend(self._check_missing_values(df))
            report.results.extend(self._check_unique_keys(df))

        # Outlier detection
        report.results.extend(self._check_outliers(df))

        # Demographic completeness (for bias analysis)
        report.results.extend(self._check_demographic_completeness(df))

        # Drift detection
        if reference_path is not None:
            df_ref = self._load(reference_path)
            report.results.extend(self._check_data_drift(df_ref, df))

        logger.info("Validation complete: %d errors, %d warnings",
                    report.error_count, report.warning_count)
        return report

    def validate_dicom_metadata(self, dicom_dir: str | Path) -> ValidationReport:
        """
        Validate DICOM metadata fields required for medical AI compliance.
        Checks for mandatory tags: PatientID, StudyDate, Modality, Manufacturer.
        """
        try:
            import pydicom
        except ImportError:
            warnings.warn("pydicom not installed; DICOM validation skipped.")
            report = ValidationReport(dataset_path=str(dicom_dir))
            report.results.append(ValidationResult(
                check_name="dicom_import",
                passed=False,
                severity="warning",
                message="pydicom not installed — install with: pip install pydicom",
            ))
            return report

        dicom_dir = Path(dicom_dir)
        report = ValidationReport(dataset_path=str(dicom_dir))
        required_tags = ["PatientID", "StudyDate", "Modality", "Manufacturer",
                         "PixelSpacing", "SliceThickness", "ConvolutionKernel"]
        optional_tags = ["PatientAge", "PatientSex", "InstitutionName", "DeviceSerialNumber"]

        dcm_files = list(dicom_dir.rglob("*.dcm"))
        if not dcm_files:
            report.results.append(ValidationResult(
                check_name="dicom_files_found",
                passed=False,
                severity="error",
                message=f"No .dcm files found in {dicom_dir}",
            ))
            return report

        tag_missing_counts: dict[str, int] = {t: 0 for t in required_tags + optional_tags}
        manufacturer_set: set[str] = set()
        modality_set: set[str] = set()

        for dcm_path in dcm_files:
            try:
                ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=True)
                for tag in required_tags + optional_tags:
                    if not hasattr(ds, tag):
                        tag_missing_counts[tag] += 1
                if hasattr(ds, "Manufacturer"):
                    manufacturer_set.add(str(ds.Manufacturer))
                if hasattr(ds, "Modality"):
                    modality_set.add(str(ds.Modality))
            except Exception as e:
                report.results.append(ValidationResult(
                    check_name="dicom_read_error",
                    passed=False,
                    severity="error",
                    message=f"Could not read {dcm_path.name}: {e}",
                ))

        total = len(dcm_files)
        for tag in required_tags:
            missing = tag_missing_counts[tag]
            passed = missing == 0
            report.results.append(ValidationResult(
                check_name=f"dicom_tag_{tag.lower()}",
                passed=passed,
                severity="error" if not passed else "info",
                message=f"{tag}: missing in {missing}/{total} files" if not passed
                        else f"{tag}: present in all {total} files",
                details={"missing_count": missing, "total_files": total},
            ))

        # Multi-site / multi-manufacturer check
        report.results.append(ValidationResult(
            check_name="dicom_manufacturer_diversity",
            passed=True,
            severity="info",
            message=f"Manufacturers found: {', '.join(sorted(manufacturer_set))}",
            details={"manufacturers": list(manufacturer_set)},
        ))

        report.results.append(ValidationResult(
            check_name="dicom_modality_check",
            passed=len(modality_set) > 0,
            severity="info",
            message=f"Modalities: {', '.join(sorted(modality_set))}",
            details={"modalities": list(modality_set)},
        ))

        return report

    # ------------------------------------------------------------------
    # Schema Checks
    # ------------------------------------------------------------------

    def _check_required_columns(self, df: pd.DataFrame) -> list[ValidationResult]:
        results = []
        missing = [c for c in self.schema.required_columns if c not in df.columns]
        results.append(ValidationResult(
            check_name="required_columns",
            passed=len(missing) == 0,
            severity="error",
            message=f"Missing columns: {missing}" if missing else "All required columns present",
            details={"missing": missing, "present": [c for c in self.schema.required_columns
                                                      if c in df.columns]},
        ))
        return results

    def _check_dtypes(self, df: pd.DataFrame) -> list[ValidationResult]:
        results = []
        for col, expected_dtype in self.schema.column_dtypes.items():
            if col not in df.columns:
                continue
            actual = str(df[col].dtype)
            # Flexible matching: float64 matches float32, int64 matches int32, etc.
            passed = actual == expected_dtype or (
                actual.startswith(expected_dtype.rstrip("0123456789"))
            )
            results.append(ValidationResult(
                check_name=f"dtype_{col}",
                passed=passed,
                severity="warning",
                message=f"Column '{col}': expected {expected_dtype}, got {actual}",
                details={"column": col, "expected": expected_dtype, "actual": actual},
            ))
        return results

    def _check_ranges(self, df: pd.DataFrame) -> list[ValidationResult]:
        results = []
        for col, (lo, hi) in self.schema.column_ranges.items():
            if col not in df.columns:
                continue
            out_of_range = df[col].dropna()
            out_of_range = out_of_range[(out_of_range < lo) | (out_of_range > hi)]
            frac = len(out_of_range) / max(len(df), 1)
            results.append(ValidationResult(
                check_name=f"range_{col}",
                passed=len(out_of_range) == 0,
                severity="error",
                message=f"Column '{col}': {len(out_of_range)} values outside [{lo}, {hi}] ({frac:.1%})",
                details={"out_of_range_count": int(len(out_of_range)),
                         "out_of_range_fraction": round(frac, 4),
                         "min_value": float(df[col].min()),
                         "max_value": float(df[col].max())},
            ))
        return results

    def _check_categoricals(self, df: pd.DataFrame) -> list[ValidationResult]:
        results = []
        for col, allowed in self.schema.categorical_values.items():
            if col not in df.columns:
                continue
            actual_vals = set(df[col].dropna().unique())
            unexpected = actual_vals - set(allowed)
            results.append(ValidationResult(
                check_name=f"categorical_{col}",
                passed=len(unexpected) == 0,
                severity="error",
                message=f"Column '{col}': unexpected values {unexpected}" if unexpected
                        else f"Column '{col}': all values in allowed set",
                details={"unexpected": list(unexpected), "allowed": allowed},
            ))
        return results

    def _check_missing_values(self, df: pd.DataFrame) -> list[ValidationResult]:
        results = []
        for col, max_frac in self.schema.max_missing_fraction.items():
            if col not in df.columns:
                continue
            actual_frac = df[col].isna().mean()
            results.append(ValidationResult(
                check_name=f"missing_{col}",
                passed=actual_frac <= max_frac,
                severity="error" if max_frac == 0.0 else "warning",
                message=f"Column '{col}': {actual_frac:.1%} missing (threshold: {max_frac:.1%})",
                details={"column": col, "missing_fraction": round(float(actual_frac), 4),
                         "threshold": max_frac},
            ))
        # Global missing check for all columns
        overall_missing = df.isna().mean()
        high_missing = overall_missing[overall_missing > 0.20]
        if not high_missing.empty:
            results.append(ValidationResult(
                check_name="high_missing_global",
                passed=False,
                severity="warning",
                message=f"Columns with >20% missing: {high_missing.to_dict()}",
                details={"columns": high_missing.to_dict()},
            ))
        return results

    def _check_unique_keys(self, df: pd.DataFrame) -> list[ValidationResult]:
        results = []
        for col in self.schema.unique_key_columns:
            if col not in df.columns:
                continue
            n_dupes = df[col].duplicated().sum()
            results.append(ValidationResult(
                check_name=f"unique_key_{col}",
                passed=n_dupes == 0,
                severity="error",
                message=f"Column '{col}': {n_dupes} duplicate values found",
                details={"duplicate_count": int(n_dupes)},
            ))
        return results

    # ------------------------------------------------------------------
    # Outlier Detection
    # ------------------------------------------------------------------

    def _check_outliers(self, df: pd.DataFrame) -> list[ValidationResult]:
        results = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        outlier_summary: dict[str, int] = {}

        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) < 10:
                continue
            z_scores = np.abs(stats.zscore(series, nan_policy="omit"))
            n_outliers = int((z_scores > self.outlier_z_threshold).sum())
            if n_outliers > 0:
                outlier_summary[col] = n_outliers

        passed = len(outlier_summary) == 0
        results.append(ValidationResult(
            check_name="outlier_detection_zscore",
            passed=passed,
            severity="warning",
            message=f"Outliers (|z| > {self.outlier_z_threshold}) found in: {outlier_summary}"
                    if not passed else "No statistical outliers detected",
            details={"outlier_counts": outlier_summary, "threshold": self.outlier_z_threshold},
        ))

        # IQR-based check as a secondary method
        iqr_outliers: dict[str, int] = {}
        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) < 10:
                continue
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
            n = int(((series < lo) | (series > hi)).sum())
            if n > 0:
                iqr_outliers[col] = n

        results.append(ValidationResult(
            check_name="outlier_detection_iqr",
            passed=len(iqr_outliers) == 0,
            severity="warning",
            message=f"IQR outliers (3×IQR) in: {iqr_outliers}" if iqr_outliers
                    else "No IQR outliers detected",
            details={"outlier_counts": iqr_outliers},
        ))
        return results

    # ------------------------------------------------------------------
    # Demographic Completeness (for Bias Analysis Readiness)
    # ------------------------------------------------------------------

    def _check_demographic_completeness(self, df: pd.DataFrame) -> list[ValidationResult]:
        """
        Check that standard demographic fields are present and reasonably complete.
        Required for FDA-mandated subgroup analysis.
        """
        results = []
        standard_demographics = {
            "age": "Numeric age or age_group",
            "sex": "Biological sex / gender",
            "ethnicity": "Self-reported ethnicity/race",
            "race": "Self-reported race",
        }
        present_demographics = []
        missing_demographics = []

        for col, description in standard_demographics.items():
            # Check for exact column name or common variants
            variants = [col, f"{col}_group", f"patient_{col}", f"subject_{col}"]
            found = any(v in df.columns for v in variants)
            if found:
                present_demographics.append(col)
            else:
                missing_demographics.append(col)

        results.append(ValidationResult(
            check_name="demographic_completeness",
            passed=len(missing_demographics) == 0,
            severity="warning",
            message=(f"Missing demographic columns for bias analysis: {missing_demographics}. "
                     f"FDA guidance requires subgroup performance analysis.")
                    if missing_demographics
                    else "All standard demographic fields present — bias analysis ready",
            details={
                "present": present_demographics,
                "missing": missing_demographics,
                "gmlp_note": "FDA GMLP Principle 6 requires bias awareness across demographic groups",
            },
        ))
        return results

    # ------------------------------------------------------------------
    # Data Drift Detection
    # ------------------------------------------------------------------

    def _check_data_drift(
        self, reference: pd.DataFrame, current: pd.DataFrame
    ) -> list[ValidationResult]:
        """
        Detect distribution shift between reference (training) and current (deployment) data.
        Uses KS test for continuous features and chi-square for categoricals.
        """
        results = []
        numeric_cols = [c for c in reference.select_dtypes(include=[np.number]).columns
                        if c in current.columns]
        cat_cols = [c for c in reference.select_dtypes(include=["object", "category"]).columns
                    if c in current.columns]

        drifted_numeric: dict[str, dict] = {}
        drifted_categorical: dict[str, dict] = {}

        # Kolmogorov-Smirnov test for numeric features
        for col in numeric_cols:
            ref_vals = reference[col].dropna().values
            cur_vals = current[col].dropna().values
            if len(ref_vals) < 5 or len(cur_vals) < 5:
                continue
            ks_stat, p_value = stats.ks_2samp(ref_vals, cur_vals)
            if p_value < self.drift_p_value_threshold:
                drifted_numeric[col] = {
                    "ks_statistic": round(float(ks_stat), 4),
                    "p_value": round(float(p_value), 6),
                    "ref_mean": round(float(np.mean(ref_vals)), 4),
                    "cur_mean": round(float(np.mean(cur_vals)), 4),
                }

        if drifted_numeric:
            results.append(ValidationResult(
                check_name="numeric_drift_ks_test",
                passed=False,
                severity="warning",
                message=f"Distribution drift detected in {len(drifted_numeric)} numeric columns: "
                        f"{list(drifted_numeric.keys())}",
                details={"drifted_columns": drifted_numeric,
                         "p_value_threshold": self.drift_p_value_threshold,
                         "gmlp_note": "GMLP Principle 10 – Monitor deployed model for covariate shift"},
            ))
        else:
            results.append(ValidationResult(
                check_name="numeric_drift_ks_test",
                passed=True,
                severity="info",
                message=f"No significant drift in {len(numeric_cols)} numeric features "
                        f"(KS test p > {self.drift_p_value_threshold})",
                details={"tested_columns": numeric_cols},
            ))

        # Chi-square test for categorical features
        for col in cat_cols:
            ref_counts = reference[col].value_counts()
            cur_counts = current[col].value_counts()
            all_cats = set(ref_counts.index) | set(cur_counts.index)
            ref_arr = np.array([ref_counts.get(c, 0) for c in all_cats])
            cur_arr = np.array([cur_counts.get(c, 0) for c in all_cats])
            # Normalize to same scale before chi-square
            cur_arr_scaled = cur_arr * (ref_arr.sum() / max(cur_arr.sum(), 1))
            # Filter zero-expected cells
            mask = ref_arr > 0
            if mask.sum() < 2:
                continue
            try:
                chi2, p_value = stats.chisquare(
                    f_obs=cur_arr_scaled[mask], f_exp=ref_arr[mask]
                )
                if p_value < self.drift_p_value_threshold:
                    drifted_categorical[col] = {
                        "chi2_statistic": round(float(chi2), 4),
                        "p_value": round(float(p_value), 6),
                    }
            except Exception:
                pass

        if drifted_categorical:
            results.append(ValidationResult(
                check_name="categorical_drift_chi2",
                passed=False,
                severity="warning",
                message=f"Categorical drift in {len(drifted_categorical)} columns: "
                        f"{list(drifted_categorical.keys())}",
                details={"drifted_columns": drifted_categorical},
            ))
        else:
            results.append(ValidationResult(
                check_name="categorical_drift_chi2",
                passed=True,
                severity="info",
                message=f"No significant drift in {len(cat_cols)} categorical features",
                details={"tested_columns": cat_cols},
            ))

        # PSI (Population Stability Index) for key features
        psi_results = self._compute_psi_batch(reference, current, numeric_cols[:10])
        results.extend(psi_results)

        return results

    def _compute_psi(
        self, reference: np.ndarray, current: np.ndarray, n_bins: int = 10
    ) -> float:
        """
        Compute Population Stability Index.
        PSI < 0.10: No significant change
        PSI 0.10–0.25: Moderate change, monitor
        PSI > 0.25: Major change, investigate
        """
        bins = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
        bins[0] = -np.inf
        bins[-1] = np.inf

        ref_counts = np.histogram(reference, bins=bins)[0]
        cur_counts = np.histogram(current, bins=bins)[0]

        # Add small epsilon to avoid log(0)
        eps = 1e-8
        ref_pct = ref_counts / (ref_counts.sum() + eps) + eps
        cur_pct = cur_counts / (cur_counts.sum() + eps) + eps

        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
        return float(psi)

    def _compute_psi_batch(
        self, reference: pd.DataFrame, current: pd.DataFrame, columns: list[str]
    ) -> list[ValidationResult]:
        results = []
        psi_scores: dict[str, float] = {}
        for col in columns:
            if col not in reference.columns or col not in current.columns:
                continue
            ref_vals = reference[col].dropna().values
            cur_vals = current[col].dropna().values
            if len(ref_vals) < 20 or len(cur_vals) < 20:
                continue
            psi = self._compute_psi(ref_vals, cur_vals)
            psi_scores[col] = round(psi, 4)

        high_psi = {c: v for c, v in psi_scores.items() if v > 0.25}
        moderate_psi = {c: v for c, v in psi_scores.items() if 0.10 <= v <= 0.25}

        results.append(ValidationResult(
            check_name="psi_drift_analysis",
            passed=len(high_psi) == 0,
            severity="error" if high_psi else ("warning" if moderate_psi else "info"),
            message=(f"High PSI (>0.25) in {list(high_psi.keys())} — major distribution shift. "
                     f"Moderate PSI (0.10–0.25) in {list(moderate_psi.keys())}.")
                    if high_psi or moderate_psi
                    else "All PSI scores below 0.10 — stable distributions",
            details={
                "psi_scores": psi_scores,
                "interpretation": {
                    "< 0.10": "Stable",
                    "0.10 - 0.25": "Moderate shift — monitor",
                    "> 0.25": "Major shift — investigate before deployment",
                },
            },
        ))
        return results

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _load(source: str | Path | pd.DataFrame) -> pd.DataFrame:
        if isinstance(source, pd.DataFrame):
            return source
        path = Path(source)
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        elif path.suffix == ".csv":
            return pd.read_csv(path)
        elif path.suffix == ".json":
            return pd.read_json(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate a medical ML dataset")
    parser.add_argument("dataset", help="Path to dataset (CSV, Parquet, JSON)")
    parser.add_argument("--reference", help="Reference dataset for drift detection")
    parser.add_argument("--schema", help="Schema spec JSON file")
    parser.add_argument("--output", help="Save report to JSON", default=None)
    parser.add_argument("--dicom-dir", help="DICOM directory for metadata validation")
    args = parser.parse_args()

    schema = MedicalSchemaSpec.from_json(args.schema) if args.schema else None
    validator = MedicalDataValidator(schema=schema)

    report = validator.validate_dataset(args.dataset, reference_path=args.reference)
    print(report.summary())

    if args.dicom_dir:
        dicom_report = validator.validate_dicom_metadata(args.dicom_dir)
        print("\n" + dicom_report.summary())

    if args.output:
        report.save_json(args.output)
        print(f"\nReport saved to {args.output}")
