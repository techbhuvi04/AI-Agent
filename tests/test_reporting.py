import json

import pandas as pd
import pytest

from recon.engine import build_exception_report, export_audit_trail


@pytest.fixture
def result_df():
    return pd.DataFrame({
        "payment_id": ["P1", "P2", "P3"],
        "net": [100.0, 200.0, 300.0],
        "settled_at": ["2025-01-10", "2025-01-10", "2025-01-11"],
        "assigned_utr": ["UTR_1", "UTR_1", None],
        "assigned_tier": pd.array([2, 2, pd.NA], dtype="Int64"),
        "assigned_confidence": [1.0, 1.0, pd.NA],
    })


@pytest.fixture
def bank_df():
    return pd.DataFrame({
        "utr": ["UTR_1", "UTR_2"],
        "credit": [300.0, 60000.0],
        "value_date": ["2025-01-10", "2025-01-12"],
    })


class TestBuildExceptionReport:
    def test_only_uncleared_credits_appear(self, result_df, bank_df):
        report = build_exception_report(
            result_df, bank_df, {"UTR_1": [0, 1]}, run_date="2025-01-15"
        )
        assert list(report["credit_utr"]) == ["UTR_2"]

    def test_age_days_from_run_date(self, result_df, bank_df):
        report = build_exception_report(
            result_df, bank_df, {"UTR_1": [0, 1]}, run_date="2025-01-15"
        )
        assert int(report.iloc[0]["age_days"]) == 3

    def test_materiality_thresholds(self, result_df, bank_df):
        report = build_exception_report(
            result_df, bank_df, {"UTR_1": [0, 1]}, run_date="2025-01-15"
        )
        # 60000 > 50000 → HIGH
        assert report.iloc[0]["materiality"] == "HIGH"

    def test_evidence_is_valid_json(self, result_df, bank_df):
        report = build_exception_report(
            result_df, bank_df, {"UTR_1": [0, 1]}, run_date="2025-01-15"
        )
        evidence = json.loads(report.iloc[0]["evidence"])
        assert "closest_entry_ids" in evidence
        assert "closest_sum" in evidence

    def test_suggested_action_matches_break_code(self, result_df, bank_df):
        report = build_exception_report(
            result_df, bank_df, {"UTR_1": [0, 1]}, run_date="2025-01-15"
        )
        row = report.iloc[0]
        expected = {
            "WINDOW_DEFICIT": "Expand settlement window to T+4",
            "NO_CANDIDATES": "Check for missing settlements data",
            "UNRESOLVED": "Escalate to finance ops",
        }
        if row["break_code"] in expected:
            assert row["suggested_action"] == expected[row["break_code"]]

    def test_empty_when_all_cleared(self, result_df, bank_df):
        cleared = {"UTR_1": [0, 1], "UTR_2": [2]}
        report = build_exception_report(result_df, bank_df, cleared, run_date="2025-01-15")
        assert len(report) == 0


class TestExportAuditTrail:
    def test_one_row_per_payment(self, result_df, bank_df):
        report = build_exception_report(
            result_df, bank_df, {"UTR_1": [0, 1]}, run_date="2025-01-15"
        )
        audit = export_audit_trail(result_df, {"UTR_1": [0, 1]}, report)
        assert len(audit) == len(result_df)

    def test_schema_columns_present(self, result_df, bank_df):
        audit = export_audit_trail(result_df, {"UTR_1": [0, 1]}, None)
        expected = [
            "run_id", "entry_id", "payment_id", "assigned_utr", "assigned_tier",
            "tier_name", "reason_code", "confidence", "evidence", "reconciled_at",
        ]
        assert list(audit.columns) == expected

    def test_unassigned_marked_unresolved(self, result_df, bank_df):
        audit = export_audit_trail(result_df, {"UTR_1": [0, 1]}, None)
        unassigned = audit[audit["payment_id"] == "P3"].iloc[0]
        assert unassigned["reason_code"] == "UNRESOLVED"
        assert unassigned["assigned_utr"] is None

    def test_run_id_is_stable_across_rows(self, result_df, bank_df):
        audit = export_audit_trail(result_df, {"UTR_1": [0, 1]}, None)
        assert audit["run_id"].nunique() == 1

    def test_explicit_run_id_is_used(self, result_df, bank_df):
        audit = export_audit_trail(result_df, {"UTR_1": [0, 1]}, None, run_id="RUN_X")
        assert set(audit["run_id"]) == {"RUN_X"}

    def test_tier_name_populated_for_assigned(self, result_df, bank_df):
        audit = export_audit_trail(result_df, {"UTR_1": [0, 1]}, None)
        assigned = audit[audit["payment_id"] == "P1"].iloc[0]
        assert assigned["tier_name"] == "T2 · Subset-sum DP"
        assert assigned["assigned_tier"] == 2

    def test_defaults_to_run_id_stamped_by_reconcile(self, result_df):
        result_df.attrs["run_id"] = "RUN_FROM_RECONCILE"
        audit = export_audit_trail(result_df, {"UTR_1": [0, 1]}, None)
        assert set(audit["run_id"]) == {"RUN_FROM_RECONCILE"}

    def test_evidence_carries_solution_count_and_excess(self, result_df):
        result_df["solution_count"] = [1.0, 1.0, float("nan")]
        result_df["excess_paise"] = [145.0, 145.0, float("nan")]
        audit = export_audit_trail(result_df, {"UTR_1": [0, 1]}, None)

        evidence = json.loads(audit[audit["payment_id"] == "P1"].iloc[0]["evidence"])
        assert evidence["solution_count"] == 1
        assert evidence["excess_paise"] == 145


class TestReconcileStampsRunId:
    def test_reconcile_stamps_a_run_id(self):
        from generator.config import GeneratorConfig, Difficulty
        from generator.truth import build_truth
        from generator.views import to_orders, to_settlements, to_bank
        from recon.engine import reconcile

        config = GeneratorConfig(
            seed=42, num_payments=50, difficulty=Difficulty.MEDIUM,
            batch_size_min=8, batch_size_max=15,
        )
        truth = build_truth(config)
        result, _cleared = reconcile(
            to_orders(truth), to_settlements(truth), to_bank(truth), max_tier=2
        )
        assert result.attrs.get("run_id")

    def test_explicit_run_id_is_honoured(self):
        from generator.config import GeneratorConfig, Difficulty
        from generator.truth import build_truth
        from generator.views import to_orders, to_settlements, to_bank
        from recon.engine import reconcile

        config = GeneratorConfig(
            seed=42, num_payments=50, difficulty=Difficulty.MEDIUM,
            batch_size_min=8, batch_size_max=15,
        )
        truth = build_truth(config)
        result, _cleared = reconcile(
            to_orders(truth), to_settlements(truth), to_bank(truth),
            max_tier=2, run_id="RUN_FIXED",
        )
        assert result.attrs["run_id"] == "RUN_FIXED"
