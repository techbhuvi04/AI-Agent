import pandas as pd

from generator.config import BREAK_TYPES


ALL_BREAK_TYPES = ["clean"] + BREAK_TYPES


def compute_metrics(result_df, ground_truth_df, cleared_credits, bank_df):
    assigned_mask = result_df["assigned_utr"].notna()
    correct_mask = assigned_mask & (
        result_df["assigned_utr"].values == ground_truth_df["credit_utr"].values
    )

    total = len(result_df)
    assigned = int(assigned_mask.sum())
    correct = int(correct_mask.sum())

    auto_clear_rate = assigned / total if total > 0 else 0.0
    precision = correct / assigned if assigned > 0 else None
    recall = correct / total if total > 0 else 0.0

    if precision is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = None

    overall = {
        "auto_clear_rate": auto_clear_rate,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "credits_cleared": len(cleared_credits),
        "total_credits": len(bank_df),
    }

    per_break = {}
    for bt in ALL_BREAK_TYPES:
        mask = ground_truth_df["break_type"] == bt
        bt_total = int(mask.sum())
        if bt_total == 0:
            continue
        bt_assigned = int((assigned_mask & mask).sum())
        bt_correct = int((correct_mask & mask).sum())
        per_break[bt] = {
            "total": bt_total,
            "assigned": bt_assigned,
            "correct": bt_correct,
            "precision": bt_correct / bt_assigned if bt_assigned > 0 else None,
            "recall": bt_correct / bt_total,
        }

    return {"overall": overall, "per_break": per_break}
