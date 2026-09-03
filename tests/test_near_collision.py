"""Adversarial test: two distinct payment subsets sum to the same credit
within the ±₹0.50 tolerance. A naive matcher that stops at the first
sum-equal subset would auto-clear this — and be wrong half the time.

The uniqueness gate (T4) must instead enumerate both subsets, see the
count exceed 1, and refuse to auto-clear, surfacing the credit as a
SUM_COLLISION in the exception queue with solution_count attached.

This is the concrete proof that "sum-equality is not proof" is enforced,
not just asserted in the design doc.
"""

import pandas as pd

from recon.t4_verifier import verify_claims
from recon.engine import build_exception_report


def _pool(nets):
    n = len(nets)
    return pd.DataFrame({
        "payment_id": [f"P{i}" for i in range(n)],
        "net": nets,
        "settled_at": ["2025-03-01"] * n,
        "assigned_utr": [None] * n,
        "assigned_tier": pd.array([pd.NA] * n, dtype="Int64"),
        "assigned_confidence": [pd.NA] * n,
    })


class TestNearCollision:
    def setup_method(self):
        # Target ₹100.00 is reachable two ways within this pool:
        #   {P0=30, P1=70}  and  {P2=45, P3=55}
        # P4/P5 are decoys too large to participate.
        self.result = _pool([30.0, 70.0, 45.0, 55.0, 200.0, 210.0])
        self.bank = pd.DataFrame({
            "utr": ["UTR_COLLIDE"],
            "credit": [100.0],
            "value_date": ["2025-03-03"],
        })

    def test_two_valid_subsets_are_not_auto_cleared(self):
        # T3 proposes one of the two valid subsets.
        claims = [{"credit_utr": "UTR_COLLIDE", "proposed_entry_ids": [0, 1]}]
        valid, rejected = verify_claims(self.result, self.bank, claims)

        assert len(valid) == 0, "a non-unique match must not auto-clear"
        assert len(rejected) == 1
        assert rejected[0]["reason_code"] == "NON_UNIQUE"
        assert rejected[0]["solution_count"] == 2

    def test_the_other_subset_is_also_rejected(self):
        # Whichever of the two subsets T3 happens to pick, the verdict
        # is the same — the ambiguity is a property of the pool.
        claims = [{"credit_utr": "UTR_COLLIDE", "proposed_entry_ids": [2, 3]}]
        valid, rejected = verify_claims(self.result, self.bank, claims)

        assert len(valid) == 0
        assert rejected[0]["reason_code"] == "NON_UNIQUE"

    def test_within_tolerance_collision_still_rejected(self):
        # Second subset is ₹0.40 off the target — inside the ±₹0.50 sum
        # tolerance, so a tolerance-only check would accept it. The
        # uniqueness sweep across the tolerance band must still catch it.
        result = _pool([30.0, 70.0, 45.20, 55.20, 200.0, 210.0])
        bank = pd.DataFrame({
            "utr": ["UTR_COLLIDE"],
            "credit": [100.0],
            "value_date": ["2025-03-03"],
        })
        claims = [{"credit_utr": "UTR_COLLIDE", "proposed_entry_ids": [0, 1]}]
        valid, rejected = verify_claims(result, bank, claims)

        assert len(valid) == 0
        assert rejected[0]["reason_code"] == "NON_UNIQUE"

    def test_collision_surfaces_as_sum_collision_exception(self):
        # End to end: a rejected NON_UNIQUE claim must appear in the
        # controller's exception queue as SUM_COLLISION, with a
        # solution-count-aware suggested action.
        claims = [{"credit_utr": "UTR_COLLIDE", "proposed_entry_ids": [0, 1]}]
        _valid, rejected = verify_claims(self.result, self.bank, claims)

        bank = self.bank.copy()
        bank.attrs["t3_claims"] = []
        bank.attrs["t4_rejected"] = rejected

        report = build_exception_report(
            self.result, bank, cleared={}, run_date="2025-03-10"
        )
        row = report.loc[report["credit_utr"] == "UTR_COLLIDE"].iloc[0]

        assert row["break_code"] == "SUM_COLLISION"
        assert "valid subsets" in row["suggested_action"]

    def test_unique_match_in_same_shaped_pool_is_cleared(self):
        # Control: the same pool shape but with only ONE subset hitting
        # the target must clear as UNIQUE_MATCH — the gate is selective,
        # not just conservative.
        result = _pool([30.0, 70.0, 45.0, 60.0, 200.0, 210.0])  # {P2,P3}=105, no 2nd path to 100
        bank = pd.DataFrame({
            "utr": ["UTR_UNIQ"], "credit": [100.0], "value_date": ["2025-03-03"],
        })
        claims = [{"credit_utr": "UTR_UNIQ", "proposed_entry_ids": [0, 1]}]
        valid, rejected = verify_claims(result, bank, claims)

        assert len(valid) == 1
        assert len(rejected) == 0
        assert valid[0]["reason_code"] == "UNIQUE_MATCH"
