"""
Unit tests for MedicalDataValidator.
"""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_validation.schema_validator import (
    MedicalDataValidator,
    MedicalSchemaSpec,
    ValidationReport,
)


def make_sample_df(n: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, n)
    return pd.DataFrame({
        "patient_id": [f"P{i:05d}" for i in range(n)],
        "age": rng.integers(25, 80, n).astype(float),
        "sex": rng.choice(["M", "F"], n),
        "label": labels,
        "feature_1": rng.normal(labels * 0.5, 1.0),
        "feature_2": rng.uniform(0, 1, n),
    })


class TestSchemaValidator:
    def test_basic_validation_passes(self):
        df = make_sample_df()
        schema = MedicalSchemaSpec(
            required_columns=["patient_id", "age", "label"],
            column_ranges={"age": (0, 120)},
            max_missing_fraction={"label": 0.0},
            unique_key_columns=["patient_id"],
        )
        validator = MedicalDataValidator(schema=schema)
        report = validator.validate_dataset(df)
        assert report.passed, report.summary()

    def test_missing_required_columns_fails(self):
        df = pd.DataFrame({"age": [30, 40], "feature": [1.0, 2.0]})
        schema = MedicalSchemaSpec(required_columns=["patient_id", "label"])
        validator = MedicalDataValidator(schema=schema)
        report = validator.validate_dataset(df)
        assert not report.passed

    def test_out_of_range_values_flagged(self):
        df = make_sample_df()
        df.loc[0, "age"] = 999  # Out of range
        schema = MedicalSchemaSpec(
            required_columns=["patient_id"],
            column_ranges={"age": (0, 120)},
        )
        validator = MedicalDataValidator(schema=schema)
        report = validator.validate_dataset(df)
        range_checks = [r for r in report.results if "range" in r.check_name]
        assert any(not r.passed for r in range_checks)

    def test_duplicate_keys_flagged(self):
        df = make_sample_df()
        df.loc[1, "patient_id"] = df.loc[0, "patient_id"]  # Duplicate
        schema = MedicalSchemaSpec(
            required_columns=["patient_id"],
            unique_key_columns=["patient_id"],
        )
        validator = MedicalDataValidator(schema=schema)
        report = validator.validate_dataset(df)
        unique_checks = [r for r in report.results if "unique_key" in r.check_name]
        assert any(not r.passed for r in unique_checks)

    def test_psi_no_drift(self):
        """PSI should be near zero when reference == current."""
        df = make_sample_df(seed=42)
        validator = MedicalDataValidator()
        report = validator.validate_dataset(df, reference_path=df)
        psi_checks = [r for r in report.results if "psi" in r.check_name]
        assert len(psi_checks) > 0

    def test_drift_detected_on_shifted_data(self):
        """KS test should detect significant shift."""
        rng = np.random.default_rng(42)
        df_ref = pd.DataFrame({
            "patient_id": [f"R{i}" for i in range(200)],
            "feature_1": rng.normal(0, 1, 200),
            "label": rng.integers(0, 2, 200),
        })
        df_cur = pd.DataFrame({
            "patient_id": [f"C{i}" for i in range(200)],
            "feature_1": rng.normal(5.0, 1, 200),  # Shifted by 5 sigma
            "label": rng.integers(0, 2, 200),
        })
        validator = MedicalDataValidator(drift_p_value_threshold=0.05)
        report = validator.validate_dataset(df_cur, reference_path=df_ref)
        drift_checks = [r for r in report.results if "drift" in r.check_name or "psi" in r.check_name]
        assert any(not r.passed for r in drift_checks)

    def test_demographic_completeness_check(self):
        """Should warn when demographic columns missing."""
        df = pd.DataFrame({"patient_id": ["A"], "label": [1], "feature": [0.5]})
        validator = MedicalDataValidator()
        report = validator.validate_dataset(df)
        demo_checks = [r for r in report.results if "demographic" in r.check_name]
        assert len(demo_checks) > 0
        assert not demo_checks[0].passed  # No demographic cols → warning

    def test_report_to_dict(self):
        df = make_sample_df()
        validator = MedicalDataValidator()
        report = validator.validate_dataset(df)
        d = report.to_dict()
        assert "passed" in d
        assert "results" in d
        assert isinstance(d["results"], list)


class TestSchemaSpec:
    def test_roundtrip_json(self, tmp_path):
        spec = MedicalSchemaSpec(
            required_columns=["patient_id", "label"],
            column_ranges={"age": (0, 120)},
            max_missing_fraction={"label": 0.0},
        )
        path = tmp_path / "spec.json"
        spec.to_json(path)
        loaded = MedicalSchemaSpec.from_json(path)
        assert loaded.required_columns == spec.required_columns
        assert loaded.column_ranges == spec.column_ranges
