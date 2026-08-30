import pandas as pd
from datetime import date, timedelta


SUM_TOLERANCE = 0.50


def _build_date_windows(bank):
    sorted_bank = bank.sort_values("value_date").reset_index(drop=True)
    dates = [pd.to_datetime(v).date() for v in sorted_bank["value_date"]]

    # Anchor each window to the previous *distinct* value_date — several
    # credits can share a value_date, and keying off the immediately
    # preceding row would give every credit after the first an inverted,
    # empty window.
    distinct = sorted(set(dates))
    prev_of = {d: (distinct[i - 1] if i > 0 else None) for i, d in enumerate(distinct)}

    windows = []
    for (_, row), vd in zip(sorted_bank.iterrows(), dates):
        prev_date = prev_of[vd]
        window_start = prev_date + timedelta(days=1) if prev_date else date(2000, 1, 1)
        windows.append({
            "utr": row["utr"],
            "credit": float(row["credit"]),
            "window_start": window_start,
            "window_end": vd,
        })

    return windows


def _check_arithmetic(df):
    result = df.copy()
    computed_net = result["gross"] - result["fee"] - result["gst"]
    result["arith_ok"] = (result["net"] - computed_net).abs() < 0.005
    return result


def _try_greedy_date_split(settlements_in_window, target_credit, window_utr):
    if len(settlements_in_window) == 0:
        return None

    total = settlements_in_window["net"].sum()
    if abs(total - target_credit) < SUM_TOLERANCE:
        return settlements_in_window.index.tolist()

    return None


def _try_boundary_expansion(df, window, assigned_indices, all_windows):
    window_idx = all_windows.index(window)
    candidates = df.loc[
        ~df.index.isin(assigned_indices)
        & df["_settled_date"].between(window["window_start"], window["window_end"])
    ]

    if _try_greedy_date_split(candidates, window["credit"], window["utr"]) is not None:
        return candidates.index.tolist()

    if window_idx > 0:
        prev_window = all_windows[window_idx - 1]
        shared_date = window["window_start"] - timedelta(days=1)
        expanded_start = shared_date
        expanded = df.loc[
            ~df.index.isin(assigned_indices)
            & df["_settled_date"].between(expanded_start, window["window_end"])
        ]
        total = expanded["net"].sum()
        if abs(total - window["credit"]) < SUM_TOLERANCE:
            return expanded.index.tolist()

    return None


def run(enriched, bank):
    df = _check_arithmetic(enriched)
    df["_settled_date"] = pd.to_datetime(df["settled_at"]).dt.date
    df["assigned_utr"] = pd.Series(dtype="object")
    df["assigned_tier"] = pd.Series(dtype="Int64")
    df["assigned_confidence"] = pd.Series(dtype="float")

    windows = _build_date_windows(bank)
    assigned_indices = set()
    cleared_credits = {}

    for window in windows:
        candidate_mask = (
            df["_settled_date"].between(window["window_start"], window["window_end"])
            & ~df.index.isin(assigned_indices)
        )
        candidates = df.loc[candidate_mask]
        match_indices = _try_greedy_date_split(candidates, window["credit"], window["utr"])

        if match_indices is not None:
            for idx in match_indices:
                df.at[idx, "assigned_utr"] = window["utr"]
                df.at[idx, "assigned_tier"] = 1
                df.at[idx, "assigned_confidence"] = 1.0
                assigned_indices.add(idx)
            cleared_credits[window["utr"]] = match_indices

    uncleared_windows = [w for w in windows if w["utr"] not in cleared_credits]
    for window in uncleared_windows:
        result = _try_boundary_expansion(df, window, assigned_indices, windows)
        if result is not None:
            for idx in result:
                df.at[idx, "assigned_utr"] = window["utr"]
                df.at[idx, "assigned_tier"] = 1
                df.at[idx, "assigned_confidence"] = 1.0
                assigned_indices.add(idx)
            cleared_credits[window["utr"]] = result

    df = df.drop(columns=["_settled_date"])
    return df, cleared_credits
