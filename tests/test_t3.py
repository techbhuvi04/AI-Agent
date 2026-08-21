import json
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from generator.config import GeneratorConfig, Difficulty
from generator.truth import build_truth
from generator.views import to_orders, to_settlements, to_bank
from recon.engine import reconcile
from recon.t3_agent import _build_prompt, _parse_response, apply_claims


SMALL_CONFIG = GeneratorConfig(
    seed=42,
    num_payments=50,
    difficulty=Difficulty.MEDIUM,
    batch_size_min=8,
    batch_size_max=15,
)


class TestParseResponse:
    def test_parses_clean_json(self):
        raw = json.dumps({
            "credit_utr": "UTR001",
            "proposed_entry_ids": [1, 2, 3],
            "reasoning": "test",
            "confidence": 0.9,
        })
        result = _parse_response(raw)
        assert result["proposed_entry_ids"] == [1, 2, 3]

    def test_parses_markdown_wrapped_json(self):
        raw = '```json\n{"credit_utr": "UTR001", "proposed_entry_ids": [4, 5]}\n```'
        result = _parse_response(raw)
        assert result["proposed_entry_ids"] == [4, 5]

    def test_parses_json_with_surrounding_text(self):
        raw = 'Here is the result:\n{"credit_utr": "X", "proposed_entry_ids": [10]}\nDone.'
        result = _parse_response(raw)
        assert result["proposed_entry_ids"] == [10]

    def test_returns_none_for_garbage(self):
        assert _parse_response("not json at all") is None

    def test_returns_none_for_none_input(self):
        assert _parse_response(None) is None


class TestBuildPrompt:
    def test_contains_credit_info(self):
        candidates = pd.DataFrame({
            "payment_id": ["PAY_001"],
            "net": [100.00],
            "settled_at": ["2025-01-10"],
        })
        prompt = _build_prompt("UTR_TEST", 100.00, "2025-01-10", candidates)
        assert "UTR_TEST" in prompt
        assert "100.00" in prompt
        assert "PAY_001" in prompt

    def test_includes_all_candidates(self):
        candidates = pd.DataFrame({
            "payment_id": [f"PAY_{i}" for i in range(5)],
            "net": [10.0] * 5,
            "settled_at": ["2025-01-10"] * 5,
        })
        prompt = _build_prompt("UTR_X", 50.00, "2025-01-10", candidates)
        for i in range(5):
            assert f"PAY_{i}" in prompt


class TestApplyClaims:
    def test_applies_valid_claim(self):
        df = pd.DataFrame({
            "payment_id": ["PAY_A", "PAY_B", "PAY_C"],
            "net": [10.0, 20.0, 30.0],
            "assigned_utr": [None, None, None],
            "assigned_tier": pd.array([pd.NA, pd.NA, pd.NA], dtype="Int64"),
        })
        claims = [{
            "credit_utr": "UTR_1",
            "proposed_entry_ids": [0, 1],
        }]
        result, cleared = apply_claims(df, claims)
        assert result.at[0, "assigned_utr"] == "UTR_1"
        assert result.at[1, "assigned_utr"] == "UTR_1"
        assert pd.isna(result.at[2, "assigned_utr"])
        assert "UTR_1" in cleared

    def test_skips_already_assigned(self):
        df = pd.DataFrame({
            "payment_id": ["PAY_A", "PAY_B"],
            "net": [10.0, 20.0],
            "assigned_utr": ["UTR_OLD", None],
            "assigned_tier": pd.array([2, pd.NA], dtype="Int64"),
        })
        claims = [{
            "credit_utr": "UTR_NEW",
            "proposed_entry_ids": [0, 1],
        }]
        result, _ = apply_claims(df, claims)
        assert result.at[0, "assigned_utr"] == "UTR_OLD"
        assert result.at[1, "assigned_utr"] == "UTR_NEW"

    def test_skips_invalid_entry_ids(self):
        df = pd.DataFrame({
            "payment_id": ["PAY_A"],
            "net": [10.0],
            "assigned_utr": [None],
            "assigned_tier": pd.array([pd.NA], dtype="Int64"),
        })
        claims = [{
            "credit_utr": "UTR_1",
            "proposed_entry_ids": [0, 999],
        }]
        result, cleared = apply_claims(df, claims)
        assert result.at[0, "assigned_utr"] == "UTR_1"
        assert len(cleared["UTR_1"]) == 1


class TestT3GracefulSkip:
    def test_no_api_key_returns_empty_claims(self):
        truth = build_truth(SMALL_CONFIG)
        orders = to_orders(truth)
        settlements = to_settlements(truth)
        bank = to_bank(truth)

        with patch.dict("os.environ", {}, clear=True):
            r2, c2 = reconcile(orders, settlements, bank, max_tier=2)
            r3, c3 = reconcile(orders, settlements, bank, max_tier=3)
            assert len(c3) >= len(c2)
