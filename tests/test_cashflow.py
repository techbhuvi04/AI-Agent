import pandas as pd
import pytest

from recon.cashflow import build_cash_position


@pytest.fixture
def result_df():
    return pd.DataFrame({
        "payment_id": ["P1", "P2", "P3", "P4"],
        "net": [100.0, 200.0, 300.0, 400.0],
        "settled_at": ["2025-01-10", "2025-01-10", "2025-01-11", "2025-01-12"],
        "assigned_utr": ["UTR_1", "UTR_1", None, None],
        "assigned_tier": pd.array([2, 2, pd.NA, pd.NA], dtype="Int64"),
        "assigned_confidence": [1.0, 1.0, pd.NA, pd.NA],
    })


@pytest.fixture
def bank_df():
    return pd.DataFrame({
        "utr": ["UTR_1", "UTR_2"],
        "credit": [300.0, 700.0],
        "value_date": ["2025-01-10", "2025-01-12"],
    })


@pytest.fixture
def exception_df():
    return pd.DataFrame({
        "credit_utr": ["UTR_2"],
        "credit_amount": [700.0],
        "value_date": [pd.to_datetime("2025-01-12").date()],
        "age_days": [5],
        "materiality": ["MEDIUM"],
        "break_code": ["UNRESOLVED"],
        "delta_inr": [0.0],
        "hypothesis": [""],
        "suggested_action": ["Escalate to finance ops"],
        "evidence": ["{}"],
    })


class TestBuildCashPosition:
    def test_verified_settled_sums_assigned_payments(self, result_df, bank_df, exception_df):
        cp = build_cash_position(result_df, bank_df, exception_df, run_date="2025-01-12")
        assert cp["verified_settled"] == 300.0

    def test_at_risk_sums_unresolved_exceptions(self, result_df, bank_df, exception_df):
        cp = build_cash_position(result_df, bank_df, exception_df, run_date="2025-01-12")
        assert cp["at_risk"] == 700.0

    def test_expected_inflow_has_seven_days(self, result_df, bank_df, exception_df):
        cp = build_cash_position(result_df, bank_df, exception_df, run_date="2025-01-12")
        assert len(cp["expected_inflow"]) == 7

    def test_unassigned_projected_by_settlement_cycle(self, result_df, bank_df, exception_df):
        # P3 settles 01-11 → credits 01-13; P4 settles 01-12 → credits 01-14
        cp = build_cash_position(
            result_df, bank_df, exception_df, run_date="2025-01-12", settlement_cycle=2
        )
        assert cp["expected_inflow"]["2025-01-13"] == 300.0
        assert cp["expected_inflow"]["2025-01-14"] == 400.0
        assert cp["expected_inflow_total"] == 700.0

    def test_confidence_interval_brackets_at_risk(self, result_df, bank_df, exception_df):
        cp = build_cash_position(result_df, bank_df, exception_df, run_date="2025-01-12")
        low, high = cp["confidence_interval"]
        assert low == cp["expected_inflow_total"]
        assert high == cp["expected_inflow_total"] + cp["at_risk"]

    def test_cash_at_risk_bucketed_by_age(self, result_df, bank_df, exception_df):
        cp = build_cash_position(result_df, bank_df, exception_df, run_date="2025-01-12")
        # exception is 5 days old → falls in the 3-7d bucket
        assert cp["cash_at_risk_by_age"]["3-7d"] == 700.0
        assert cp["cash_at_risk_by_age"]["<3d"] == 0.0
        assert cp["cash_at_risk_by_age"][">7d"] == 0.0

    def test_handles_empty_exception_report(self, result_df, bank_df):
        empty = pd.DataFrame(columns=["credit_utr", "credit_amount", "age_days", "break_code"])
        cp = build_cash_position(result_df, bank_df, empty, run_date="2025-01-12")
        assert cp["at_risk"] == 0.0
        assert cp["cash_at_risk_by_age"] == {"<3d": 0.0, "3-7d": 0.0, ">7d": 0.0}
