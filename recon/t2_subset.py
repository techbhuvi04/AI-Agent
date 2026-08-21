import pandas as pd
from datetime import date, timedelta


ITERATION_CAP = 2_000_000
WINDOW_EXPANSION_DAYS = 3


def _to_paise(v):
    return int(round(v * 100))


def _build_windows(bank_df):
    sorted_bank = bank_df.sort_values("value_date").reset_index(drop=True)
    windows = []
    prev_date = None

    for _, row in sorted_bank.iterrows():
        vd = pd.to_datetime(row["value_date"]).date()
        strict_start = (
            prev_date + timedelta(days=1) if prev_date else date(2000, 1, 1)
        )
        windows.append({
            "utr": row["utr"],
            "credit_paise": _to_paise(row["credit"]),
            "strict_start": strict_start,
            "strict_end": vd,
        })
        prev_date = vd

    return windows


def _dp_find_subset(values, target, max_iter=ITERATION_CAP):
    if target <= 0:
        return set() if target == 0 else None

    dp = {0: None}
    iterations = 0

    for i, v in enumerate(values):
        if v <= 0 or v > target:
            continue
        new_entries = {}
        for s in list(dp.keys()):
            iterations += 1
            if iterations > max_iter:
                return None
            new_s = s + v
            if new_s <= target and new_s not in dp and new_s not in new_entries:
                new_entries[new_s] = (i, s)
                if new_s == target:
                    result = {i}
                    prev = s
                    while dp[prev] is not None:
                        idx, ps = dp[prev]
                        result.add(idx)
                        prev = ps
                    return result
        dp.update(new_entries)

    return None


def _solve_credit(candidate_df_indices, candidate_nets_paise, target_paise):
    total = sum(candidate_nets_paise)
    excess = total - target_paise

    if excess == 0:
        return set(candidate_df_indices)

    if excess > 0:
        excludable_positions = []
        excludable_values = []
        for pos, net in enumerate(candidate_nets_paise):
            if net > 0:
                excludable_positions.append(pos)
                excludable_values.append(net)

        excl_result = _dp_find_subset(excludable_values, excess)
        if excl_result is not None:
            exclude_positions = {excludable_positions[i] for i in excl_result}
            return {
                candidate_df_indices[pos]
                for pos in range(len(candidate_df_indices))
                if pos not in exclude_positions
            }

    return None


def run(result_df, bank_df, already_cleared):
    df = result_df.copy()
    df["_settled_date"] = pd.to_datetime(df["settled_at"]).dt.date

    windows = _build_windows(bank_df)
    new_cleared = {}
    assigned_indices = set()
    for indices in already_cleared.values():
        assigned_indices.update(indices)

    for window in windows:
        if window["utr"] in already_cleared:
            continue

        for expansion in [0, WINDOW_EXPANSION_DAYS, WINDOW_EXPANSION_DAYS * 2]:
            exp_start = window["strict_start"] - timedelta(days=expansion)
            exp_end = window["strict_end"] + timedelta(days=expansion)

            candidate_mask = (
                ~df.index.isin(assigned_indices)
                & df["_settled_date"].between(exp_start, exp_end)
            )
            candidates = df.loc[candidate_mask]

            if len(candidates) == 0:
                continue

            c_indices = list(candidates.index)
            c_nets = [_to_paise(row["net"]) for _, row in candidates.iterrows()]

            member_indices = _solve_credit(c_indices, c_nets, window["credit_paise"])

            if member_indices is not None:
                for idx in member_indices:
                    df.at[idx, "assigned_utr"] = window["utr"]
                    df.at[idx, "assigned_tier"] = 2
                    df.at[idx, "assigned_confidence"] = 1.0
                    assigned_indices.add(idx)
                new_cleared[window["utr"]] = list(member_indices)
                break

    df = df.drop(columns=["_settled_date"])
    return df, new_cleared
