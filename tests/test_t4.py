import pandas as pd

from recon.t4_verifier import verify_claims


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
            "assigned_tier": pd.array([pd.NA, pd.NA, 2, pd.NA, pd.NA], dtype="Int64")
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
