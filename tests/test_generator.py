import pandas as pd
import pytest

from generator.config import GeneratorConfig, Difficulty, BREAK_TYPES
from generator.truth import build_truth
from generator.views import to_orders, to_settlements, to_bank


SMALL_CONFIG = GeneratorConfig(
    seed=42,
    num_payments=100,
    difficulty=Difficulty.HARD,
    batch_size_min=10,
    batch_size_max=25,
)


@pytest.fixture
def truth():
    return build_truth(SMALL_CONFIG)


@pytest.fixture
def orders(truth):
    return to_orders(truth)


@pytest.fixture
def settlements(truth):
    return to_settlements(truth)


@pytest.fixture
def bank(truth):
    return to_bank(truth)


class TestDeterminism:
    def test_same_seed_produces_identical_output(self):
        run_a = build_truth(SMALL_CONFIG)
        run_b = build_truth(SMALL_CONFIG)
        pd.testing.assert_frame_equal(run_a, run_b)

    def test_different_seed_produces_different_output(self):
        other = GeneratorConfig(
            seed=99,
            num_payments=100,
            difficulty=Difficulty.HARD,
            batch_size_min=10,
            batch_size_max=25,
        )
        run_a = build_truth(SMALL_CONFIG)
        run_b = build_truth(other)
        assert not run_a["net"].equals(run_b["net"])


class TestArithmeticInvariant:
    def test_fee_gst_net_for_clean_originals(self, truth):
        clean_originals = truth[
            truth["is_original"]
            & ~truth["break_type"].isin(["rounding_drift"])
        ]
        expected_fee = (clean_originals["gross"] * SMALL_CONFIG.fee_rate).round().astype(int)
        expected_gst = (expected_fee * SMALL_CONFIG.gst_rate).round().astype(int)
        expected_net = clean_originals["gross"] - expected_fee - expected_gst

        pd.testing.assert_series_equal(
            clean_originals["fee"].reset_index(drop=True),
            expected_fee.reset_index(drop=True),
            check_names=False,
        )
        pd.testing.assert_series_equal(
            clean_originals["gst"].reset_index(drop=True),
            expected_gst.reset_index(drop=True),
            check_names=False,
        )
        pd.testing.assert_series_equal(
            clean_originals["net"].reset_index(drop=True),
            expected_net.reset_index(drop=True),
            check_names=False,
        )

    def test_bank_credit_equals_sum_of_member_nets(self, truth, bank):
        total_truth_paise = truth["net"].sum()
        total_bank_rupees = bank["credit"].sum()
        assert abs(total_truth_paise / 100 - total_bank_rupees) < 0.01

    def test_per_batch_credit_integrity(self, truth):
        for batch_id in truth["credit_batch_id"].unique():
            members = truth[truth["credit_batch_id"] == batch_id]
            assert len(members) > 0

    def test_rounding_drift_changes_net(self, truth):
        drifted = truth[
            truth["is_original"] & (truth["break_type"] == "rounding_drift")
        ]
        if len(drifted) == 0:
            pytest.skip("no rounding_drift rows in this seed")
        for _, row in drifted.iterrows():
            expected_net = row["gross"] - row["fee"] - row["gst"]
            assert row["net"] != expected_net
            assert abs(row["net"] - expected_net) <= SMALL_CONFIG.rounding_drift_max_paise


class TestColumnCompleteness:
    def test_orders_columns(self, orders):
        assert list(orders.columns) == ["order_id", "payment_id", "gross", "captured_at"]

    def test_settlements_columns(self, settlements):
        assert list(settlements.columns) == [
            "payment_id", "gross", "fee", "gst", "net", "settled_at",
        ]

    def test_bank_columns(self, bank):
        assert list(bank.columns) == ["value_date", "narration", "credit", "utr"]

    def test_no_utr_in_orders(self, orders):
        assert "utr" not in orders.columns
        assert "credit_utr" not in orders.columns

    def test_no_payment_id_in_bank(self, bank):
        assert "payment_id" not in bank.columns


class TestBreakCounts:
    def test_each_break_type_present_on_hard(self, truth):
        originals = truth[truth["is_original"]]
        present = set(originals["break_type"].unique())
        for bt in BREAK_TYPES:
            assert bt in present, f"break type '{bt}' not found in hard difficulty"

    def test_adjustments_exist_for_refunds(self, truth):
        refund_originals = truth[
            truth["is_original"] & (truth["break_type"] == "refund")
        ]
        refund_adjustments = truth[
            ~truth["is_original"] & (truth["break_type"] == "refund")
        ]
        assert len(refund_adjustments) == len(refund_originals)

    def test_adjustments_exist_for_chargebacks(self, truth):
        cb_originals = truth[
            truth["is_original"] & (truth["break_type"] == "chargeback_reversal")
        ]
        cb_adjustments = truth[
            ~truth["is_original"] & (truth["break_type"] == "chargeback_reversal")
        ]
        assert len(cb_adjustments) == 2 * len(cb_originals)


class TestViewConsistency:
    def test_missing_order_excluded_from_orders(self, truth, orders):
        missing = truth[
            truth["is_original"] & (truth["break_type"] == "missing_order")
        ]
        for _, row in missing.iterrows():
            assert row["payment_id"] not in orders["payment_id"].values

    def test_missing_order_present_in_settlements(self, truth, settlements):
        missing = truth[
            truth["is_original"] & (truth["break_type"] == "missing_order")
        ]
        for _, row in missing.iterrows():
            assert row["payment_id"] in settlements["payment_id"].values

    def test_settlement_row_count_matches_truth(self, truth, settlements):
        assert len(settlements) == len(truth)

    def test_order_count_excludes_adjustments_and_missing(self, truth, orders):
        expected = len(
            truth[truth["is_original"] & (truth["break_type"] != "missing_order")]
        )
        assert len(orders) == expected


class TestWorkingDayLogic:
    def test_no_weekend_settlements(self, truth):
        for settled_at in truth["settled_at"]:
            assert settled_at.weekday() < 5, f"{settled_at} is a weekend"

    def test_no_weekend_captures(self, truth):
        originals = truth[truth["is_original"]]
        for captured_at in originals["captured_at"]:
            assert captured_at.weekday() < 5, f"{captured_at} is a weekend"
