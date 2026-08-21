import pandas as pd
import pytest

from generator.config import GeneratorConfig, Difficulty
from generator.truth import build_truth
from generator.views import to_orders, to_settlements, to_bank
from recon import t0_keys, t1_arith
from recon.engine import reconcile, score


SMALL_CONFIG = GeneratorConfig(
    seed=42,
    num_payments=100,
    difficulty=Difficulty.HARD,
    batch_size_min=10,
    batch_size_max=25,
)


@pytest.fixture
def generated_data():
    truth = build_truth(SMALL_CONFIG)
    orders = to_orders(truth)
    settlements = to_settlements(truth)
    bank = to_bank(truth)
    return truth, orders, settlements, bank


class TestT0Keys:
    def test_preserves_all_settlement_rows(self, generated_data):
        truth, orders, settlements, bank = generated_data
        enriched = t0_keys.run(orders, settlements)
        assert len(enriched) == len(settlements)

    def test_marks_missing_orders(self, generated_data):
        truth, orders, settlements, bank = generated_data
        enriched = t0_keys.run(orders, settlements)

        missing_truth_count = (
            truth["is_original"] & (truth["break_type"] == "missing_order")
        ).sum()
        unmatched_count = (~enriched["order_matched"]).sum()

        assert unmatched_count >= missing_truth_count

    def test_matched_orders_have_captured_at(self, generated_data):
        truth, orders, settlements, bank = generated_data
        enriched = t0_keys.run(orders, settlements)
        matched = enriched[enriched["order_matched"]]
        assert matched["captured_at"].notna().all()


class TestT1Arith:
    def test_flags_rounding_drift(self, generated_data):
        truth, orders, settlements, bank = generated_data
        enriched = t0_keys.run(orders, settlements)
        result, _ = t1_arith.run(enriched, bank)

        drift_mask = truth["break_type"] == "rounding_drift"
        if drift_mask.sum() == 0:
            pytest.skip("no rounding_drift in this seed")

        drift_arith = result.loc[drift_mask, "arith_ok"]
        assert not drift_arith.all(), "rounding_drift should fail arithmetic check"

    def test_clean_payments_pass_arithmetic(self, generated_data):
        truth, orders, settlements, bank = generated_data
        enriched = t0_keys.run(orders, settlements)
        result, _ = t1_arith.run(enriched, bank)

        clean_original = truth["is_original"] & (truth["break_type"] == "clean")
        clean_arith = result.loc[clean_original, "arith_ok"]
        assert clean_arith.all()

    def test_some_credits_auto_cleared(self, generated_data):
        truth, orders, settlements, bank = generated_data
        enriched = t0_keys.run(orders, settlements)
        _, cleared = t1_arith.run(enriched, bank)
        assert len(cleared) > 0, "at least one credit should auto-clear"


class TestEndToEnd:
    def test_recall_is_measured(self, generated_data):
        truth, orders, settlements, bank = generated_data
        result, cleared = reconcile(orders, settlements, bank)
        scores = score(result, truth)

        assert "OVERALL" in scores["break_type"].values
        overall = scores[scores["break_type"] == "OVERALL"].iloc[0]
        assert overall["total"] == len(truth)
        assert overall["recall"] >= 0.0

    def test_no_fabricated_assignments(self, generated_data):
        truth, orders, settlements, bank = generated_data
        result, _ = reconcile(orders, settlements, bank)

        assigned = result[result["assigned_utr"].notna()]
        valid_utrs = set(bank["utr"])
        for utr in assigned["assigned_utr"]:
            assert utr in valid_utrs
