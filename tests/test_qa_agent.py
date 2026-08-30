from unittest.mock import patch

import pandas as pd
import pytest

from recon.qa_agent import answer_question, build_context, is_available


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
        "credit": [300.0, 300.0],
        "value_date": ["2025-01-10", "2025-01-12"],
    })


@pytest.fixture
def exception_df():
    return pd.DataFrame({
        "credit_utr": ["UTR_2"],
        "credit_amount": [300.0],
        "value_date": [pd.to_datetime("2025-01-12").date()],
        "age_days": [5],
        "materiality": ["MEDIUM"],
        "break_code": ["UNRESOLVED"],
        "delta_inr": [0.0],
        "hypothesis": [""],
        "suggested_action": ["Escalate to finance ops"],
        "evidence": ["{}"],
    })


@pytest.fixture
def cash_position():
    return {
        "verified_settled": 300.0,
        "at_risk": 300.0,
        "expected_inflow": {"2025-01-13": 300.0},
        "expected_inflow_total": 300.0,
        "confidence_interval": (300.0, 600.0),
        "cash_at_risk_by_age": {"<3d": 0.0, "3-7d": 300.0, ">7d": 0.0},
    }


class TestBuildContext:
    def test_includes_summary_figures(self, result_df, bank_df, exception_df, cash_position):
        ctx = build_context(result_df, bank_df, exception_df, cash_position)
        assert "Total payments: 3" in ctx
        assert "UTR_2" in ctx
        assert "UNRESOLVED" in ctx

    def test_stays_compact(self, result_df, bank_df, exception_df, cash_position):
        ctx = build_context(result_df, bank_df, exception_df, cash_position)
        # Rough token proxy — the context must stay well under 2000 tokens.
        assert len(ctx) // 4 < 2000

    def test_handles_empty_exception_report(self, result_df, bank_df, cash_position):
        empty = pd.DataFrame(columns=[
            "credit_utr", "credit_amount", "value_date", "age_days",
            "materiality", "break_code", "delta_inr", "hypothesis",
            "suggested_action", "evidence",
        ])
        ctx = build_context(result_df, bank_df, empty, cash_position)
        assert "No open exceptions" in ctx

    def test_does_not_dump_raw_rows(self, result_df, bank_df, exception_df, cash_position):
        ctx = build_context(result_df, bank_df, exception_df, cash_position)
        # Individual payment_ids must not leak into the context — only
        # aggregates and credit-level exception rows.
        assert "P1" not in ctx
        assert "P3" not in ctx


class TestGracefulDegradation:
    def test_returns_message_without_api_key(self, result_df, bank_df, exception_df, cash_position):
        with patch.dict("os.environ", {}, clear=True):
            answer = answer_question(
                "What is at risk?", result_df, bank_df, exception_df, cash_position
            )
        assert "GROQ_API_KEY" in answer

    def test_is_available_reflects_env(self):
        with patch.dict("os.environ", {}, clear=True):
            assert is_available() is False
        with patch.dict("os.environ", {"GROQ_API_KEY": "x"}, clear=True):
            assert is_available() is True
