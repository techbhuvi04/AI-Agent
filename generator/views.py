import pandas as pd


AMOUNT_COLUMNS = ["gross", "fee", "gst", "net", "credit"]


def _to_rupees(df, columns=None):
    if columns is None:
        columns = [c for c in AMOUNT_COLUMNS if c in df.columns]
    result = df.copy()
    for col in columns:
        result[col] = (result[col] / 100).round(2)
    return result


def to_orders(truth):
    originals = truth[truth["is_original"] & (truth["break_type"] != "missing_order")]
    result = originals[["order_id", "payment_id", "gross", "captured_at"]].copy()
    return _to_rupees(result).reset_index(drop=True)


def to_settlements(truth):
    result = truth[["payment_id", "gross", "fee", "gst", "net", "settled_at"]].copy()
    return _to_rupees(result).reset_index(drop=True)


def to_bank(truth):
    grouped = truth.groupby("credit_batch_id").agg(
        value_date=("settled_at", "max"),
        credit=("net", "sum"),
        utr=("credit_utr", "first"),
    ).reset_index(drop=True)

    grouped["narration"] = grouped["utr"].apply(
        lambda u: f"NEFT-{u}-RAZORPAY SOFTWARE PVT LTD-SETTLEMENT"
    )

    result = grouped[["value_date", "narration", "credit", "utr"]]
    return _to_rupees(result).reset_index(drop=True)


def truth_to_rupees(truth):
    return _to_rupees(truth)
