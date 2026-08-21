import pandas as pd
import pytest

from generator.config import GeneratorConfig, Difficulty
from generator.truth import build_truth
from generator.views import to_orders, to_settlements, to_bank
from recon.engine import reconcile
from eval.metrics import compute_metrics


SMALL_CONFIG = GeneratorConfig(
    seed=42,
    num_payments=50,
    difficulty=Difficulty.MEDIUM,
    batch_size_min=8,
    batch_size_max=15,
)


@pytest.fixture
def eval_data():
    truth = build_truth(SMALL_CONFIG)
    orders = to_orders(truth)
    settlements = to_settlements(truth)
    bank = to_bank(truth)
    return truth, orders, settlements, bank


class TestMetricsZeroAssignments:
    def test_recall_is_zero(self, eval_data):
        truth, orders, settlements, bank = eval_data
        result, cleared = reconcile(orders, settlements, bank, max_tier=0)
        metrics = compute_metrics(result, truth, cleared, bank)
        assert metrics["overall"]["recall"] == 0.0

    def test_precision_is_none(self, eval_data):
        truth, orders, settlements, bank = eval_data
        result, cleared = reconcile(orders, settlements, bank, max_tier=0)
        metrics = compute_metrics(result, truth, cleared, bank)
        assert metrics["overall"]["precision"] is None

    def test_auto_clear_rate_is_zero(self, eval_data):
        truth, orders, settlements, bank = eval_data
        result, cleared = reconcile(orders, settlements, bank, max_tier=0)
        metrics = compute_metrics(result, truth, cleared, bank)
        assert metrics["overall"]["auto_clear_rate"] == 0.0


class TestMetricsPerfectAssignment:
    def test_perfect_scores(self, eval_data):
        truth, orders, settlements, bank = eval_data
        result, cleared = reconcile(orders, settlements, bank, max_tier=0)
        result["assigned_utr"] = truth["credit_utr"].values
        fake_cleared = {utr: [] for utr in bank["utr"]}
        metrics = compute_metrics(result, truth, fake_cleared, bank)
        assert metrics["overall"]["precision"] == 1.0
        assert metrics["overall"]["recall"] == 1.0
        assert metrics["overall"]["auto_clear_rate"] == 1.0
        assert metrics["overall"]["f1"] == 1.0

    def test_all_breaks_have_perfect_recall(self, eval_data):
        truth, orders, settlements, bank = eval_data
        result, cleared = reconcile(orders, settlements, bank, max_tier=0)
        result["assigned_utr"] = truth["credit_utr"].values
        fake_cleared = {utr: [] for utr in bank["utr"]}
        metrics = compute_metrics(result, truth, fake_cleared, bank)
        for bt, m in metrics["per_break"].items():
            assert m["recall"] == 1.0, f"{bt} recall != 1.0"


class TestMetricsStructure:
    def test_required_overall_fields(self, eval_data):
        truth, orders, settlements, bank = eval_data
        result, cleared = reconcile(orders, settlements, bank, max_tier=1)
        metrics = compute_metrics(result, truth, cleared, bank)
        required = [
            "auto_clear_rate", "precision", "recall",
            "f1", "credits_cleared", "total_credits",
        ]
        for field in required:
            assert field in metrics["overall"], f"missing {field}"

    def test_per_break_has_entries(self, eval_data):
        truth, orders, settlements, bank = eval_data
        result, cleared = reconcile(orders, settlements, bank, max_tier=1)
        metrics = compute_metrics(result, truth, cleared, bank)
        assert len(metrics["per_break"]) > 0

    def test_per_break_fields(self, eval_data):
        truth, orders, settlements, bank = eval_data
        result, cleared = reconcile(orders, settlements, bank, max_tier=1)
        metrics = compute_metrics(result, truth, cleared, bank)
        for bt, m in metrics["per_break"].items():
            for field in ["total", "assigned", "correct", "precision", "recall"]:
                assert field in m, f"{bt} missing {field}"


class TestAblation:
    def test_t0_only_has_zero_recall(self, eval_data):
        truth, orders, settlements, bank = eval_data
        result, cleared = reconcile(orders, settlements, bank, max_tier=0)
        metrics = compute_metrics(result, truth, cleared, bank)
        assert metrics["overall"]["recall"] == 0.0
        assert metrics["overall"]["credits_cleared"] == 0

    def test_higher_tier_geq_lower_tier_recall(self, eval_data):
        truth, orders, settlements, bank = eval_data

        r0, c0 = reconcile(orders, settlements, bank, max_tier=0)
        r1, c1 = reconcile(orders, settlements, bank, max_tier=1)

        m0 = compute_metrics(r0, truth, c0, bank)
        m1 = compute_metrics(r1, truth, c1, bank)

        assert m1["overall"]["recall"] >= m0["overall"]["recall"]
