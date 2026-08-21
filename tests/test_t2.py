import pytest

from generator.config import GeneratorConfig, Difficulty
from generator.truth import build_truth
from generator.views import to_orders, to_settlements, to_bank
from recon.engine import reconcile
from recon.t2_subset import _dp_find_subset, _to_paise
from eval.metrics import compute_metrics


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


class TestDPFindSubset:
    def test_finds_single_element(self):
        assert _dp_find_subset([100, 200, 300], 200) == {1}

    def test_finds_pair(self):
        result = _dp_find_subset([100, 200, 300], 500)
        assert result == {1, 2}

    def test_returns_none_for_impossible_target(self):
        assert _dp_find_subset([100, 200, 300], 150) is None

    def test_handles_zero_target(self):
        assert _dp_find_subset([100, 200], 0) == set()

    def test_respects_iteration_cap(self):
        values = list(range(1, 51))
        result = _dp_find_subset(values, 99999, max_iter=100)
        assert result is None

    def test_skips_negative_values(self):
        result = _dp_find_subset([-50, 100, 200], 200)
        assert result == {2}


class TestT2Integration:
    def test_t2_clears_more_than_t1(self, generated_data):
        truth, orders, settlements, bank = generated_data

        _, c1 = reconcile(orders, settlements, bank, max_tier=1)
        _, c2 = reconcile(orders, settlements, bank, max_tier=2)

        assert len(c2) >= len(c1)

    def test_t2_improves_recall(self, generated_data):
        truth, orders, settlements, bank = generated_data

        r1, c1 = reconcile(orders, settlements, bank, max_tier=1)
        r2, c2 = reconcile(orders, settlements, bank, max_tier=2)

        m1 = compute_metrics(r1, truth, c1, bank)
        m2 = compute_metrics(r2, truth, c2, bank)

        assert m2["overall"]["recall"] >= m1["overall"]["recall"]

    def test_assigned_utrs_are_valid(self, generated_data):
        truth, orders, settlements, bank = generated_data
        result, _ = reconcile(orders, settlements, bank, max_tier=2)

        valid_utrs = set(bank["utr"])
        assigned = result[result["assigned_utr"].notna()]
        for utr in assigned["assigned_utr"]:
            assert utr in valid_utrs

    def test_paise_conversion_roundtrip(self):
        assert _to_paise(123.45) == 12345
        assert _to_paise(0.01) == 1
        assert _to_paise(-500.00) == -50000
