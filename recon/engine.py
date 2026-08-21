import os
from collections import Counter

import pandas as pd

from recon import t0_keys, t1_arith, t2_subset, t3_agent, t4_verifier
from generator.config import BREAK_TYPES


def load_data(data_dir):
    orders = pd.read_csv(os.path.join(data_dir, "orders.csv"))
    settlements = pd.read_csv(os.path.join(data_dir, "settlements.csv"))
    bank = pd.read_csv(os.path.join(data_dir, "bank.csv"))
    return orders, settlements, bank


def load_ground_truth(data_dir):
    return pd.read_csv(os.path.join(data_dir, "ground_truth.csv"))


def reconcile(orders, settlements, bank, max_tier=1, min_confidence=0.0):
    enriched = t0_keys.run(orders, settlements)

    if max_tier < 1:
        enriched["assigned_utr"] = pd.Series(dtype="object", index=enriched.index)
        enriched["assigned_tier"] = pd.Series(dtype="Int64", index=enriched.index)
        enriched["assigned_confidence"] = pd.Series(dtype="float", index=enriched.index)
        enriched["arith_ok"] = pd.Series(dtype="object", index=enriched.index)
        return enriched, {}

    result, cleared = t1_arith.run(enriched, bank)

    if max_tier >= 2:
        result, new_cleared = t2_subset.run(result, bank, cleared)
        cleared.update(new_cleared)

    if max_tier >= 3:
        claims = t3_agent.run(result, bank, cleared)
        if claims:
            if max_tier >= 4:
                valid_claims, rejected_claims = t4_verifier.verify_claims(result, bank, claims)
                for rc in rejected_claims:
                    print(f"  T4: Rejected claim for {rc['credit_utr']}: {rc.get('rejection_reason')}")
                claims_to_apply = valid_claims
            else:
                claims_to_apply = claims
                
            if claims_to_apply:
                result, new_cleared = t3_agent.apply_claims(result, claims_to_apply)
                cleared.update(new_cleared)

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

    return result, cleared


def score(result, ground_truth):
    truth_utr = ground_truth["credit_utr"]
    assigned_utr = result["assigned_utr"]

    all_break_types = ["clean"] + BREAK_TYPES
    rows = []

    for bt in all_break_types:
        mask = ground_truth["break_type"] == bt
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


def run_full(data_dir="data"):
    orders, settlements, bank = load_data(data_dir)
    ground_truth = load_ground_truth(data_dir)
    result, cleared = reconcile(orders, settlements, bank)
    scores = score(result, ground_truth)
    print_report(scores, cleared, len(bank))
    return result, cleared, scores


if __name__ == "__main__":
    run_full()
