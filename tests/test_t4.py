import pandas as pd

from recon.t4_verifier import verify_claims, _count_solutions


class TestT4Verifier:
    def setup_method(self):
        self.bank = pd.DataFrame({
            "utr": ["UTR_1", "UTR_2", "UTR_3"],
            "credit": [100.0, 200.0, 300.0]
        })
        
        self.result = pd.DataFrame({
            "payment_id": ["P1", "P2", "P3", "P4", "P5"],
            "net": [40.0, 60.0, 200.0, 150.0, 150.0],
            "assigned_utr": [None, None, "UTR_OLD", None, None],
            "assigned_tier": pd.array([pd.NA, pd.NA, 2, pd.NA, pd.NA], dtype="Int64"),
            "assigned_confidence": [pd.NA, pd.NA, 1.0, pd.NA, pd.NA],
        })

    def test_accepts_valid_claim(self):
        claims = [{"credit_utr": "UTR_1", "proposed_entry_ids": [0, 1]}]
        valid, rejected = verify_claims(self.result, self.bank, claims)
        
        assert len(valid) == 1
        assert len(rejected) == 0
        assert valid[0]["credit_utr"] == "UTR_1"

    def test_rejects_sum_mismatch(self):
        claims = [{"credit_utr": "UTR_1", "proposed_entry_ids": [0]}]
        valid, rejected = verify_claims(self.result, self.bank, claims)
        
        assert len(valid) == 0
        assert len(rejected) == 1
        assert "Sum mismatch" in rejected[0]["rejection_reason"]

    def test_rejects_unknown_utr(self):
        claims = [{"credit_utr": "UTR_UNKNOWN", "proposed_entry_ids": [0, 1]}]
        valid, rejected = verify_claims(self.result, self.bank, claims)
        
        assert len(valid) == 0
        assert len(rejected) == 1
        assert "Unknown UTR" in rejected[0]["rejection_reason"]

    def test_rejects_already_assigned_entry(self):
        claims = [{"credit_utr": "UTR_2", "proposed_entry_ids": [2]}]
        valid, rejected = verify_claims(self.result, self.bank, claims)
        
        assert len(valid) == 0
        assert len(rejected) == 1
        assert "Invalid or already assigned" in rejected[0]["rejection_reason"]

    def test_rejects_empty_entry_ids(self):
        claims = [{"credit_utr": "UTR_1", "proposed_entry_ids": []}]
        valid, rejected = verify_claims(self.result, self.bank, claims)
        
        assert len(valid) == 0
        assert len(rejected) == 1
        assert "No valid entry IDs" in rejected[0]["rejection_reason"]

    def test_accepts_within_tolerance(self):
        # Result net is 40 + 60 = 100.0, bank is 100.49
        bank_tol = pd.DataFrame({"utr": ["UTR_1"], "credit": [100.49]})
        claims = [{"credit_utr": "UTR_1", "proposed_entry_ids": [0, 1]}]
        valid, rejected = verify_claims(self.result, bank_tol, claims)
        
        assert len(valid) == 1
        
    def test_rejects_outside_tolerance(self):
        # Result net is 40 + 60 = 100.0, bank is 100.51
        bank_tol = pd.DataFrame({"utr": ["UTR_1"], "credit": [100.51]})
        claims = [{"credit_utr": "UTR_1", "proposed_entry_ids": [0, 1]}]
        valid, rejected = verify_claims(self.result, bank_tol, claims)

        assert len(valid) == 0

    def test_valid_claim_sets_unique_match_reason_code(self):
        # Only one subset of the eligible pool (P1=40, P2=60, P4=150,
        # P5=150) sums to 100: {P1, P2}. Should be accepted as unique.
        claims = [{"credit_utr": "UTR_1", "proposed_entry_ids": [0, 1]}]
        valid, rejected = verify_claims(self.result, self.bank, claims)

        assert len(valid) == 1
        assert valid[0]["reason_code"] == "UNIQUE_MATCH"
        assert valid[0]["confidence"] == 1.0


class TestCountSolutions:
    def test_single_solution(self):
        # 40 + 60 = 100, and no other subset of these four values does.
        solutions, exhaustive = _count_solutions([4000, 6000, 15000, 15000], 10000, cap=51)
        assert len(solutions) == 1
        assert exhaustive is True

    def test_no_solution(self):
        solutions, exhaustive = _count_solutions([100, 200, 300], 999, cap=51)
        assert len(solutions) == 0
        assert exhaustive is True

    def test_many_solutions_capped(self):
        # 60 entries of ₹10 each; any 30 of them sum to the same ₹300
        # target, giving C(60,30) >> 51 distinct solutions.
        values = [1000] * 60
        solutions, exhaustive = _count_solutions(values, 30000, cap=51)
        assert len(solutions) == 51
        # Hitting the cap means the search stopped early, not that the
        # space was fully explored.
        assert exhaustive is False


class TestUniquenessGate:
    def test_single_solution_credit_is_accepted(self):
        bank = pd.DataFrame({"utr": ["UTR_UNIQ"], "credit": [100.0]})
        result = pd.DataFrame({
            "payment_id": ["P1", "P2", "P3", "P4"],
            "net": [40.0, 60.0, 150.0, 150.0],
            "assigned_utr": [None, None, None, None],
            "assigned_tier": pd.array([pd.NA] * 4, dtype="Int64"),
            "assigned_confidence": [pd.NA] * 4,
        })
        claims = [{"credit_utr": "UTR_UNIQ", "proposed_entry_ids": [0, 1]}]
        valid, rejected = verify_claims(result, bank, claims)

        assert len(valid) == 1
        assert len(rejected) == 0
        assert valid[0]["reason_code"] == "UNIQUE_MATCH"
        assert valid[0]["confidence"] == 1.0

    def test_unprovable_uniqueness_is_rejected_not_cleared(self):
        # A pool whose enumeration is cut short must never be read as
        # "unique" — the engine refuses rather than clearing on evidence
        # it did not actually establish.
        n = 60
        bank = pd.DataFrame({"utr": ["UTR_BIG"], "credit": [300.0]})
        result = pd.DataFrame({
            "payment_id": [f"P{i}" for i in range(n)],
            "net": [10.0] * n,
            "assigned_utr": [None] * n,
            "assigned_tier": pd.array([pd.NA] * n, dtype="Int64"),
            "assigned_confidence": [pd.NA] * n,
        })
        claims = [{
            "credit_utr": "UTR_BIG",
            "proposed_entry_ids": list(range(30)),
        }]
        valid, rejected = verify_claims(result, bank, claims)

        assert len(valid) == 0
        assert rejected[0]["reason_code"] == "NON_UNIQUE"

    def test_many_solution_credit_is_rejected_non_unique(self):
        # 60 entries of ₹10 each; the target ₹300 is reachable by
        # C(60,30) distinct 30-entry subsets — far more than the
        # ambiguous-partial cap of 10, so the claim must be rejected
        # outright rather than auto-cleared.
        n = 60
        bank = pd.DataFrame({"utr": ["UTR_AMBIG"], "credit": [300.0]})
        result = pd.DataFrame({
            "payment_id": [f"P{i}" for i in range(n)],
            "net": [10.0] * n,
            "assigned_utr": [None] * n,
            "assigned_tier": pd.array([pd.NA] * n, dtype="Int64"),
            "assigned_confidence": [pd.NA] * n,
        })
        claims = [{
            "credit_utr": "UTR_AMBIG",
            "proposed_entry_ids": list(range(30)),
        }]
        valid, rejected = verify_claims(result, bank, claims)

        assert len(valid) == 0
        assert len(rejected) == 1
        assert "NON_UNIQUE" in rejected[0]["rejection_reason"]
