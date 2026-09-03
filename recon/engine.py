import json
import os
import uuid
from collections import Counter
from datetime import date, datetime, timedelta, timezone

import pandas as pd

from recon import t0_keys, t1_arith, t2_subset, t3_agent, t4_verifier
from generator.config import BREAK_TYPES

SUM_TOLERANCE_PAISE = 50  # 0.50 INR

TIER_NAMES = {
    0: "T0 · Key enrichment",
    1: "T1 · Date arithmetic",
    2: "T2 · Subset-sum DP",
    3: "T3 · LLM agent / break classifier",
    4: "T4 · Arithmetic verifier",
}


def load_data(data_dir):
    orders = pd.read_csv(os.path.join(data_dir, "orders.csv"))
    settlements = pd.read_csv(os.path.join(data_dir, "settlements.csv"))
    bank = pd.read_csv(os.path.join(data_dir, "bank.csv"))
    return orders, settlements, bank


def load_ground_truth(data_dir):
    return pd.read_csv(os.path.join(data_dir, "ground_truth.csv"))


def reconcile(orders, settlements, bank, max_tier=1, min_confidence=0.0, run_id=None):
    # One run_id per reconcile() call, stamped onto every audit-trail row
    # produced from this run's output.
    if run_id is None:
        run_id = str(uuid.uuid4())

    enriched = t0_keys.run(orders, settlements)

    if max_tier < 1:
        enriched["assigned_utr"] = pd.Series(dtype="object", index=enriched.index)
        enriched["assigned_tier"] = pd.Series(dtype="Int64", index=enriched.index)
        enriched["assigned_confidence"] = pd.Series(dtype="float", index=enriched.index)
        enriched["arith_ok"] = pd.Series(dtype="object", index=enriched.index)
        enriched.attrs["run_id"] = run_id
        return enriched, {}

    result, cleared = t1_arith.run(enriched, bank)

    if max_tier >= 2:
        result, new_cleared = t2_subset.run(result, bank, cleared)
        cleared.update(new_cleared)

    if max_tier >= 3:
        claims = t3_agent.run(result, bank, cleared)
        # Stash T3/T4 diagnostics on the bank_df's .attrs so downstream
        # reporting (build_exception_report, the audit trail) can look up
        # why an uncleared credit failed without needing a wider return
        # signature — bank is the same object the caller holds, so this
        # metadata survives back to their reference.
        bank.attrs["t3_claims"] = claims
        bank.attrs["t4_rejected"] = []
        if claims:
            if max_tier >= 4:
                valid_claims, rejected_claims = t4_verifier.verify_claims(result, bank, claims)
                bank.attrs["t4_rejected"] = rejected_claims
                for rc in rejected_claims:
                    print(f"  T4: Rejected claim for {rc['credit_utr']}: {rc.get('rejection_reason')}")
                claims_to_apply = valid_claims
            else:
                claims_to_apply = claims

            if claims_to_apply:
                result, new_cleared = t3_agent.apply_claims(result, claims_to_apply)
                cleared.update(new_cleared)
                # Reconcile cleared dict: entries may have been
                # reassigned from one credit to another by T3.
                for utr in list(cleared.keys()):
                    valid = [
                        i for i in cleared[utr]
                        if result.at[i, "assigned_utr"] == utr
                    ]
                    if valid:
                        cleared[utr] = valid
                    else:
                        del cleared[utr]

    if min_confidence > 0.0:
        low_conf_mask = result["assigned_confidence"] < min_confidence
        if low_conf_mask.any():
            result.loc[low_conf_mask, ["assigned_utr", "assigned_tier", "assigned_confidence"]] = [pd.NA, pd.NA, pd.NA]
            
            cleared = {}
            valid_assignments = result[result["assigned_utr"].notna()]
            for idx, row in valid_assignments.iterrows():
                utr = row["assigned_utr"]
                if utr not in cleared:
                    cleared[utr] = []
                cleared[utr].append(idx)

    # attrs is not reliably preserved across the .copy() each tier makes,
    # so stamp the run_id onto the final frame the caller receives.
    result.attrs["run_id"] = run_id
    return result, cleared


def score(result, ground_truth):
    assert len(result) == len(ground_truth), "result/ground_truth length mismatch"
    gt = ground_truth.set_index("payment_id").reindex(result["payment_id"].values)
    truth_utr = gt["credit_utr"].reset_index(drop=True)
    truth_break = gt["break_type"].reset_index(drop=True)
    assigned_utr = result["assigned_utr"].reset_index(drop=True)

    all_break_types = ["clean"] + BREAK_TYPES
    rows = []

    for bt in all_break_types:
        mask = truth_break == bt
        total = int(mask.sum())
        if total == 0:
            continue
        correct = int(
            ((assigned_utr[mask] == truth_utr[mask]) & assigned_utr[mask].notna()).sum()
        )
        rows.append({
            "break_type": bt,
            "total": total,
            "correct": correct,
            "recall": round(correct / total, 3) if total > 0 else 0.0,
        })

    total_all = len(ground_truth)
    correct_all = int(
        ((assigned_utr == truth_utr) & assigned_utr.notna()).sum()
    )
    rows.append({
        "break_type": "OVERALL",
        "total": total_all,
        "correct": correct_all,
        "recall": round(correct_all / total_all, 3),
    })

    return pd.DataFrame(rows)


def print_report(scores, cleared, total_credits):
    print(f"\n{'='*50}")
    print(f"  T0 + T1 Reconciliation Report")
    print(f"{'='*50}")
    print(f"\nAuto-cleared credits: {len(cleared)} / {total_credits}")
    print(f"\nPer-break-type recall (members_correct):")
    print(f"  {'break_type':<25} {'correct':>8} {'total':>8} {'recall':>8}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8}")
    for _, row in scores.iterrows():
        marker = " <<<" if row["break_type"] == "OVERALL" else ""
        print(
            f"  {row['break_type']:<25} {row['correct']:>8} "
            f"{row['total']:>8} {row['recall']:>8.3f}{marker}"
        )


def _materiality(amount):
    if amount > 50000:
        return "HIGH"
    if amount > 5000:
        return "MEDIUM"
    return "LOW"


def _to_paise(v):
    return int(round(v * 100))


def build_exception_report(result_df, bank_df, cleared, run_date=None):
    """One row per uncleared bank credit: age, materiality, an inferred or
    T3-classified break code, the closest candidate subset found, and a
    suggested next action. Reads T3's break classification / T4's
    rejection reason from bank_df.attrs (see reconcile()) when available,
    and falls back to structural inference otherwise.
    """
    if run_date is None:
        run_date = date.today()
    elif isinstance(run_date, str):
        run_date = pd.to_datetime(run_date).date()
    elif isinstance(run_date, datetime):
        run_date = run_date.date()

    claims_by_utr = {c["credit_utr"]: c for c in (bank_df.attrs.get("t3_claims") or [])}
    rejected_by_utr = {c["credit_utr"]: c for c in (bank_df.attrs.get("t4_rejected") or [])}

    cleared_utrs = set(cleared.keys())
    unassigned = result_df.loc[result_df["assigned_utr"].isna()]
    settled_dates = pd.to_datetime(unassigned["settled_at"]).dt.date if len(unassigned) else pd.Series(dtype="object")

    rows = []
    for _, credit in bank_df.iterrows():
        utr = credit["utr"]
        if utr in cleared_utrs:
            continue

        amount = float(credit["credit"])
        amount_paise = _to_paise(amount)
        value_date = pd.to_datetime(credit["value_date"]).date()
        age_days = (run_date - value_date).days

        window_start = value_date - timedelta(days=7)
        window_end = value_date + timedelta(days=7)
        if len(unassigned):
            candidates = unassigned.loc[settled_dates.between(window_start, window_end)]
        else:
            candidates = unassigned

        candidate_paise = [_to_paise(v) for v in candidates["net"]] if len(candidates) else []
        total_candidates_paise = sum(candidate_paise)

        claim = claims_by_utr.get(utr)
        rejected = rejected_by_utr.get(utr)

        best_ids = list(candidates.index) if len(candidates) else []
        best_sum_paise = total_candidates_paise

        if rejected is not None and rejected.get("reason_code") == "NON_UNIQUE":
            break_code = "SUM_COLLISION"
        elif claim is not None and claim.get("break_classification") not in (None, "unknown"):
            break_code = claim["break_classification"].upper()
        elif len(candidates) == 0:
            break_code = "NO_CANDIDATES"
        elif total_candidates_paise < amount_paise - SUM_TOLERANCE_PAISE:
            break_code = "WINDOW_DEFICIT"
        else:
            break_code = "UNRESOLVED"

        if break_code == "WINDOW_DEFICIT":
            suggested_action = "Expand settlement window to T+4"
        elif break_code == "SUM_COLLISION":
            n = rejected.get("solution_count") if rejected else None
            suggested_action = (
                f"Manual review: {n}+ valid subsets found" if n else "Manual review: multiple valid subsets found"
            )
        elif break_code == "NO_CANDIDATES":
            suggested_action = "Check for missing settlements data"
        else:
            suggested_action = "Escalate to finance ops"

        delta_inr = (amount_paise - best_sum_paise) / 100 if best_ids else 0.0
        hypothesis = claim.get("reasoning", "") if claim else ""

        rows.append({
            "credit_utr": utr,
            "credit_amount": amount,
            "value_date": value_date,
            "age_days": age_days,
            "materiality": _materiality(amount),
            "break_code": break_code,
            "delta_inr": delta_inr,
            "hypothesis": hypothesis,
            "suggested_action": suggested_action,
            "evidence": json.dumps({
                "closest_entry_ids": best_ids[:25],
                "closest_sum": best_sum_paise / 100,
                "num_candidates": int(len(candidates)),
            }),
        })

    return pd.DataFrame(rows, columns=[
        "credit_utr", "credit_amount", "value_date", "age_days", "materiality",
        "break_code", "delta_inr", "hypothesis", "suggested_action", "evidence",
    ])


def export_audit_trail(result_df, cleared, exception_report_df, run_id=None):
    """One row per payment: what it was assigned to (or not), why, and
    with what confidence — the full paper trail for a reconciliation run.

    Defaults `run_id` to the uuid4 stamped by reconcile(), so every row
    ties back to the run that produced it.
    """
    if run_id is None:
        run_id = result_df.attrs.get("run_id") or str(uuid.uuid4())

    has_break_col = "break_classification" in result_df.columns
    has_reason_col = "reason_code" in result_df.columns
    reconciled_at = datetime.now(timezone.utc).isoformat()
    rows = []

    for idx, row in result_df.iterrows():
        assigned_utr = row.get("assigned_utr")
        assigned_tier = row.get("assigned_tier")
        confidence = row.get("assigned_confidence")
        has_assignment = pd.notna(assigned_utr)

        solution_count = row.get("solution_count")
        excess_paise = row.get("excess_paise")

        if has_assignment:
            tier_num = int(assigned_tier) if pd.notna(assigned_tier) else None
            tier_name = TIER_NAMES.get(tier_num, f"T{tier_num}" if tier_num is not None else "—")

            # Prefer the reason code T4 actually assigned; fall back to the
            # break classification, then to tier-implied defaults.
            stored_reason = row.get("reason_code") if has_reason_col else None
            break_classification = row.get("break_classification") if has_break_col else None

            if pd.notna(stored_reason) and stored_reason:
                reason_code = str(stored_reason)
            elif pd.notna(break_classification) and break_classification not in (None, "unknown"):
                reason_code = str(break_classification).upper()
            elif tier_num in (3, 4):
                reason_code = "UNIQUE_MATCH" if (pd.notna(confidence) and confidence >= 0.95) else "AMBIGUOUS_PARTIAL"
            else:
                reason_code = "SUM_MATCH"

            evidence_payload = {"tier": tier_num}
            if pd.notna(solution_count):
                evidence_payload["solution_count"] = int(solution_count)
            if pd.notna(excess_paise):
                evidence_payload["excess_paise"] = int(excess_paise)
            evidence = json.dumps(evidence_payload)
        else:
            tier_name = None
            reason_code = "UNRESOLVED"
            confidence = None
            evidence = json.dumps({})

        rows.append({
            "run_id": run_id,
            "entry_id": idx,
            "payment_id": row.get("payment_id"),
            "assigned_utr": assigned_utr if has_assignment else None,
            "assigned_tier": int(assigned_tier) if has_assignment and pd.notna(assigned_tier) else None,
            "tier_name": tier_name,
            "reason_code": reason_code,
            "confidence": float(confidence) if pd.notna(confidence) else None,
            "evidence": evidence,
            "reconciled_at": reconciled_at,
        })

    return pd.DataFrame(rows)


def run_full(data_dir="data"):
    orders, settlements, bank = load_data(data_dir)
    ground_truth = load_ground_truth(data_dir)
    result, cleared = reconcile(orders, settlements, bank)
    scores = score(result, ground_truth)
    print_report(scores, cleared, len(bank))
    return result, cleared, scores


if __name__ == "__main__":
    run_full()
