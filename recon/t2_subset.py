import pandas as pd
from datetime import date, timedelta


ITERATION_CAP = 500_000
WINDOW_EXPANSION_DAYS = 3

# The bitset DP trades iterations for memory: budget bits rather than
# loop steps. Tuned so the old iteration caps map onto a few hundred MB
# worst case.
BITSET_BUDGET_FACTOR = 64


def _to_paise(v):
    return int(round(v * 100))


def _build_windows(bank_df):
    sorted_bank = bank_df.sort_values("value_date").reset_index(drop=True)
    dates = [pd.to_datetime(v).date() for v in sorted_bank["value_date"]]

    # Anchor each window to the previous *distinct* value_date. Several
    # credits can share a value_date (multiple batches settling the same
    # day); keying off the immediately preceding row instead would hand
    # every credit after the first an inverted, empty window.
    distinct = sorted(set(dates))
    prev_of = {d: (distinct[i - 1] if i > 0 else None) for i, d in enumerate(distinct)}

    windows = []
    for (_, row), vd in zip(sorted_bank.iterrows(), dates):
        prev_date = prev_of[vd]
        strict_start = (
            prev_date + timedelta(days=1) if prev_date else date(2000, 1, 1)
        )
        windows.append({
            "utr": row["utr"],
            "credit_paise": _to_paise(row["credit"]),
            "strict_start": strict_start,
            "strict_end": vd,
        })

    return windows


def _dp_find_subset(values, target, max_iter=ITERATION_CAP):
    """Find any subset of `values` summing exactly to `target`.

    The reachable-sum set is carried as a bitmask in a single Python
    integer — bit *s* set means "sum s is reachable" — so one relaxation
    step is `reach |= reach << v`, a C-speed big-integer shift instead of
    a Python loop over every reachable sum. A per-prefix snapshot of the
    mask lets the chosen items be recovered afterwards.

    Returns a set of indices into `values`, or None if no subset sums to
    target (or the problem is too large for the memory budget).
    """
    if target <= 0:
        return set() if target == 0 else None

    usable = [i for i, v in enumerate(values) if 0 < v <= target]
    if not usable:
        return None

    # Snapshots cost (target+1) bits each; bail rather than blow up memory
    # on a pathologically large target. `max_iter` keeps the old
    # give-up-instead-of-hang contract, reinterpreted as a bit budget.
    if len(usable) * (target + 1) > max_iter * BITSET_BUDGET_FACTOR:
        return None

    mask = (1 << (target + 1)) - 1
    reach = 1  # only sum 0 reachable to start
    prefix = [reach]

    for i in usable:
        reach = (reach | (reach << values[i])) & mask
        prefix.append(reach)
        if (reach >> target) & 1:
            break

    if not ((reach >> target) & 1):
        return None

    # Walk the snapshots back: if the target sum was already reachable
    # without item k, it wasn't needed; otherwise it must have been used.
    result = set()
    s = target
    for k in range(len(prefix) - 1, 0, -1):
        if s == 0:
            break
        if (prefix[k - 1] >> s) & 1:
            continue
        i = usable[k - 1]
        result.add(i)
        s -= values[i]

    return result if s == 0 else None


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

        excl_result = _dp_find_subset(excludable_values, excess, max_iter=3_000_000)
        if excl_result is not None:
            exclude_positions = {excludable_positions[i] for i in excl_result}
            return {
                candidate_df_indices[pos]
                for pos in range(len(candidate_df_indices))
                if pos not in exclude_positions
            }

    # Refund-aware fallback: separate negatives (refunds/chargebacks)
    # from positives, include all negatives, DP on positives only.
    neg_positions = [p for p, v in enumerate(candidate_nets_paise) if v < 0]
    if neg_positions:
        pos_positions = [p for p, v in enumerate(candidate_nets_paise) if v >= 0]
        pos_values = [candidate_nets_paise[p] for p in pos_positions]
        neg_sum = sum(candidate_nets_paise[p] for p in neg_positions)

        pos_target = target_paise - neg_sum
        if pos_target >= 0:
            total_pos = sum(pos_values)

            if abs(total_pos - pos_target) <= 50:  # 0.50 INR tolerance
                return {candidate_df_indices[p]
                        for p in neg_positions + pos_positions}

            if total_pos > pos_target:
                pos_excess = total_pos - pos_target
                excl = _dp_find_subset(
                    pos_values, pos_excess, max_iter=3_000_000
                )
                if excl is not None:
                    exclude_pos = {pos_positions[i] for i in excl}
                    result = {candidate_df_indices[p] for p in neg_positions}
                    result.update(
                        candidate_df_indices[p] for p in pos_positions
                        if p not in exclude_pos
                    )
                    return result

    return None


def run(result_df, bank_df, already_cleared):
    df = result_df.copy()
    df["_settled_date"] = pd.to_datetime(df["settled_at"]).dt.date
    if "excess_paise" not in df.columns:
        df["excess_paise"] = pd.Series(dtype="float", index=df.index)

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

            # Sort: entries farthest from value_date first, so the DP
            # preferentially excludes foreign entries from adjacent credits
            vd = window["strict_end"]
            candidates = candidates.copy()
            candidates["_vd_dist"] = candidates["_settled_date"].apply(
                lambda d: abs((d - vd).days)
            )
            candidates = candidates.sort_values("_vd_dist", ascending=False)

            c_indices = list(candidates.index)
            c_nets = [_to_paise(row["net"]) for _, row in candidates.iterrows()]
            total_paise = sum(c_nets)

            member_indices = _solve_credit(c_indices, c_nets, window["credit_paise"])

            if member_indices is not None:
                # Confidence based on excess ratio: how many foreign entries
                # are in the candidate pool.  Small excess → few foreign
                # entries → high confidence.
                if len(member_indices) == len(candidates):
                    conf = 1.0
                else:
                    excess_ratio = (
                        abs(total_paise - window["credit_paise"])
                        / window["credit_paise"]
                    )
                    conf = max(0.50, 1.0 - excess_ratio)
                    if expansion > 0:
                        conf = max(0.50, conf - 0.10)

                excess_paise = total_paise - window["credit_paise"]
                for idx in member_indices:
                    df.at[idx, "assigned_utr"] = window["utr"]
                    df.at[idx, "assigned_tier"] = 2
                    df.at[idx, "assigned_confidence"] = conf
                    df.at[idx, "excess_paise"] = excess_paise
                    assigned_indices.add(idx)
                new_cleared[window["utr"]] = list(member_indices)
                break

    df = df.drop(columns=["_settled_date"])
    return df, new_cleared
