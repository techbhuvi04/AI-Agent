import pandas as pd

from generator.config import BREAK_TYPES


ALL_BREAK_TYPES = ["clean"] + BREAK_TYPES


def compute_metrics(result_df, ground_truth_df, cleared_credits, bank_df):
    assert len(result_df) == len(ground_truth_df), "result/ground_truth length mismatch"
    gt = ground_truth_df.set_index("payment_id").reindex(result_df["payment_id"].values)
    truth_utr = gt["credit_utr"].values
    truth_break = gt["break_type"].values

    assigned_mask = result_df["assigned_utr"].notna().values
    correct_mask = assigned_mask & (
        result_df["assigned_utr"].fillna("").values == truth_utr
    )

    total = len(result_df)
    assigned = int(assigned_mask.sum())
    correct = int(correct_mask.sum())

    auto_clear_rate = assigned / total if total > 0 else 0.0
    precision = correct / assigned if assigned > 0 else None
    recall = correct / total if total > 0 else 0.0

    accuracy = correct / total if total > 0 else 0.0

    overall = {
        "auto_clear_rate": auto_clear_rate,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "credits_cleared": len(cleared_credits),
        "total_credits": len(bank_df),
        "payments_assigned": assigned,
        "total_payments": total,
    }

    per_break = {}
    for bt in ALL_BREAK_TYPES:
        mask = truth_break == bt
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
