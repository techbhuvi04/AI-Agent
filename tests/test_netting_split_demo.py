"""The DP's limit, and the LLM crossing it — concrete, not asserted.

One settlement batch of six payments (₹1,000 total) is paid out by the
bank as TWO credits: ₹600 and ₹400. Taken one credit at a time the
candidate pool is ambiguous:

  ₹600  ->  {300,200,100}  or  {250,200,90,60}
  ₹400  ->  {300,100}      or  {250,90,60}

The deterministic subset-sum (T2) plus the T4 uniqueness gate will not
guess which subset is real. The gate clears only the payments that
appear in *every* valid subset (here just ₹200, shared by both ₹600
subsets) and routes the rest as AMBIGUOUS_PARTIAL — it never auto-clears
the full ambiguous split.

The LLM's job is structural: recognise this as a netting_split and
propose `merge_with_utr`. The deterministic resolver then solves the
*combined* target ₹1,000 — which IS unique, exactly the six payments —
recovering the batch the per-credit DP could not. (The final per-credit
re-split still inherits the ₹600/₹400 ambiguity; the win is that the
batch itself is identified.)
"""

import pandas as pd

from recon.t4_verifier import verify_claims, _solutions_as_entry_sets
from recon.t3_agent import resolve_hypothesis, _window_candidates
from recon.t2_subset import _to_paise


def _ledger():
    nets = [300.0, 200.0, 100.0, 250.0, 90.0, 60.0]  # one batch, ₹1,000
    n = len(nets)
    df = pd.DataFrame({
        "payment_id": [f"P{i}" for i in range(n)],
        "net": nets,
        "settled_at": ["2025-04-10"] * n,
        "assigned_utr": [None] * n,
        "assigned_tier": pd.array([pd.NA] * n, dtype="Int64"),
        "assigned_confidence": [pd.NA] * n,
    })
    bank = pd.DataFrame({
        "utr": ["UTR_A", "UTR_B"],
        "credit": [600.0, 400.0],
        "value_date": ["2025-04-12", "2025-04-12"],
    })
    return df, bank


class TestDeterministicWillNotGuessTheSplit:
    def test_per_credit_pool_is_ambiguous_two_subsets_hit_600(self):
        df, bank = _ledger()
        pool = _window_candidates(
            df, set(),
            pd.to_datetime("2025-04-05").date(),
            pd.to_datetime("2025-04-19").date(),
        )
        ids = list(pool.index)
        vals = [_to_paise(v) for v in pool["net"]]
        sols_600, exhaustive = _solutions_as_entry_sets(ids, vals, _to_paise(600.0), cap=51)

        assert exhaustive
        assert len(sols_600) == 2, "the ₹600 credit is reachable two ways"

    def test_gate_clears_only_the_shared_member_not_the_whole_split(self):
        df, bank = _ledger()
        # T2/T3 proposes one of the two ₹600 subsets.
        claims = [{"credit_utr": "UTR_A", "proposed_entry_ids": [0, 1, 2]}]
        valid, rejected = verify_claims(df, bank, claims)

        assert len(valid) == 1
        assert valid[0]["reason_code"] == "AMBIGUOUS_PARTIAL"
        assert valid[0]["solution_count"] == 2
        # Only ₹200 (P1) is in both valid subsets — the rest stays ambiguous.
        assert valid[0]["proposed_entry_ids"] == [1]
        assert set(valid[0]["ambiguous_entry_ids"]) == {0, 2}

    def test_the_decoy_subset_yields_the_same_partial_verdict(self):
        df, bank = _ledger()
        claims = [{"credit_utr": "UTR_A", "proposed_entry_ids": [3, 1, 4, 5]}]  # {250,200,90,60}
        valid, rejected = verify_claims(df, bank, claims)

        assert len(valid) == 1
        assert valid[0]["reason_code"] == "AMBIGUOUS_PARTIAL"
        assert valid[0]["proposed_entry_ids"] == [1]


class TestLLMMergeRecoversTheBatch:
    def test_combined_target_is_unique_though_each_half_is_not(self):
        df, _bank = _ledger()
        pool = _window_candidates(
            df, set(),
            pd.to_datetime("2025-04-05").date(),
            pd.to_datetime("2025-04-19").date(),
        )
        ids = list(pool.index)
        vals = [_to_paise(v) for v in pool["net"]]

        sols_600, _ = _solutions_as_entry_sets(ids, vals, _to_paise(600.0), cap=51)
        sols_1000, exhaustive = _solutions_as_entry_sets(ids, vals, _to_paise(1000.0), cap=51)

        assert len(sols_600) == 2          # ambiguous per-credit
        assert exhaustive
        assert len(sols_1000) == 1         # unique combined
        assert sols_1000[0] == frozenset(ids)

    def test_merge_with_utr_recovers_the_full_batch(self):
        df, bank = _ledger()
        bank_lookup = bank.set_index("utr")["credit"].to_dict()
        vd = pd.to_datetime("2025-04-12").date()
        diag = {
            "credit_utr": "UTR_A",
            "credit_amount": 600.0,
            "window_start": str(vd - pd.Timedelta(days=7)),
            "window_end": str(vd + pd.Timedelta(days=7)),
        }

        members = resolve_hypothesis(
            df, used_indices=set(), bank_lookup=bank_lookup, diagnostics=diag,
            break_classification="netting_split",
            hypothesis={"action": "merge_with_utr", "merge_utr": "UTR_B"},
        )

        assert members is not None, "merge_with_utr should resolve"
        # It recovers a valid ₹600 subset from the unique combined batch —
        # the arithmetic still decides which, the LLM only pointed at the
        # partner credit.
        assert abs(df.loc[list(members), "net"].sum() - 600.0) < 0.01
        assert members.issubset(set(df.index))

    def test_merge_fails_cleanly_when_partner_utr_is_unknown(self):
        df, bank = _ledger()
        bank_lookup = bank.set_index("utr")["credit"].to_dict()
        vd = pd.to_datetime("2025-04-12").date()
        diag = {
            "credit_utr": "UTR_A", "credit_amount": 600.0,
            "window_start": str(vd - pd.Timedelta(days=7)),
            "window_end": str(vd + pd.Timedelta(days=7)),
        }
        members = resolve_hypothesis(
            df, set(), bank_lookup, diag, "netting_split",
            {"action": "merge_with_utr", "merge_utr": "UTR_DOES_NOT_EXIST"},
        )
        assert members is None
