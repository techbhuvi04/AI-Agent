import pandas as pd

SUM_TOLERANCE = 0.50
SUM_TOLERANCE_PAISE = 50

AMBIGUOUS_MAX_SOLUTIONS = 10
WINDOW_DAYS = 14  # matches T3's widest heuristic window

# Ceiling on distinct partial sums held in the enumeration table, and on
# total relaxation steps. The DP is exponential in the worst case; past
# these points we stop and report the enumeration as non-exhaustive
# rather than burning memory or wall-clock.
MAX_DP_STATES = 20_000
MAX_DP_ITERATIONS = 2_000_000


def _to_paise(v):
    return int(round(v * 100))


def _count_solutions(values, target, cap=51):
    """Enumerate distinct subsets of `values` (positive ints) summing to `target`.

    A sum-indexed DP table built by iterating candidates one at a time.
    Rather than stopping at the first path that reaches `target`, every
    entry in dp[target] is tracked as a separate solution (as a frozenset
    of the positions used), and exploration is capped once `cap` distinct
    solutions have been found — beyond that point the exact count no
    longer matters, only that the claim is non-unique.

    Each round collects its additions into a separate `updates` map read
    from the pre-round table, so item i is never folded into the same
    subset twice (true 0/1 knapsack) without copying the whole table per
    item — that copy is what made this quadratic in table size.

    Returns `(solutions, exhaustive)`. `solutions` is a list of at most
    `cap` frozensets of positions. `exhaustive` is False when the search
    was cut short (solution cap, state ceiling, or iteration budget),
    meaning the absence of further solutions was never actually proven —
    callers must not read a short list as proof of uniqueness.
    """
    if target < 0:
        return [], True
    if target == 0:
        return [frozenset()], True

    # dp[s] = list of frozensets of positions summing to s
    dp = {0: [frozenset()]}
    exhaustive = True
    iterations = 0

    for i, v in enumerate(values):
        if v <= 0 or v > target:
            continue
        if len(dp) > MAX_DP_STATES or iterations > MAX_DP_ITERATIONS:
            exhaustive = False
            break

        updates = {}
        for s, subsets in dp.items():
            iterations += 1
            new_s = s + v
            if new_s > target:
                continue
            existing = dp.get(new_s, ())
            room = cap - len(existing)
            if room <= 0:
                continue
            bucket = updates.setdefault(new_s, [])
            room -= len(bucket)
            if room <= 0:
                continue
            for base in subsets[:room]:
                bucket.append(base | {i})

        for s, bucket in updates.items():
            if s in dp:
                dp[s].extend(bucket)
                del dp[s][cap:]
            else:
                dp[s] = bucket[:cap]

        if len(dp.get(target, ())) >= cap:
            exhaustive = False
            break

    return dp.get(target, [])[:cap], exhaustive


def _count_solutions_by_exclusion(values, target, cap=51):
    """Count subsets summing to `target` by counting what to *exclude*.

    Mirrors T2's central optimisation: choosing which entries sum to the
    credit is equivalent to choosing which to leave out, and the excess
    (pool total − target) is normally far smaller than the target itself,
    so the DP table is correspondingly smaller. The two formulations are
    in bijection, so the solution count is identical either way — this
    just gets there cheaply.

    Returns `(solutions, exhaustive)` with solutions expressed as
    *inclusion* sets of positions, same as `_count_solutions`.
    """
    total = sum(v for v in values if v > 0)
    excess = total - target

    if excess < 0:
        # The pool cannot reach the target at all.
        return [], True

    positive_positions = [i for i, v in enumerate(values) if v > 0]

    # Pick whichever side yields the smaller DP target.
    if excess <= target:
        excl_values = [values[i] for i in positive_positions]
        exclusions, exhaustive = _count_solutions(excl_values, excess, cap=cap)
        all_positive = set(positive_positions)
        solutions = [
            frozenset(all_positive - {positive_positions[p] for p in excl})
            for excl in exclusions
        ]
        return solutions, exhaustive

    return _count_solutions(values, target, cap=cap)


def _solutions_as_entry_sets(entry_ids, values, target, cap=51):
    solutions, exhaustive = _count_solutions_by_exclusion(values, target, cap=cap)
    return (
        [frozenset(entry_ids[pos] for pos in sol) for sol in solutions],
        exhaustive,
    )


def _candidate_pool(result_df, bank_df, utr, extra_indices=(), reassign_threshold=0.80):
    """The set of entries eligible to be members of `utr`'s credit: every
    currently-unassigned or low-confidence entry within a date window
    around the credit's value_date, plus any entries already proposed in
    the claim (extra_indices) that may sit outside that window.

    Scoping to a window (instead of the entire result_df) keeps the
    uniqueness-count DP tractable at real data scale — mirrors the
    windowing T2 and T3 already use.
    """
    conf = pd.to_numeric(result_df["assigned_confidence"], errors="coerce").fillna(0)
    eligible_mask = (
        result_df["assigned_utr"].isna()
        | (conf < reassign_threshold)
    )
    eligible = result_df.loc[eligible_mask]

    bank_row = bank_df.loc[bank_df["utr"] == utr]
    if len(bank_row) == 0 or "value_date" not in bank_df.columns or "settled_at" not in eligible.columns:
        # Not enough date info to window (e.g. minimal test fixtures) —
        # fall back to the full eligible pool.
        return eligible

    value_date = pd.to_datetime(bank_row.iloc[0]["value_date"]).date()
    settled = pd.to_datetime(eligible["settled_at"]).dt.date
    window_start = value_date - pd.Timedelta(days=WINDOW_DAYS)
    window_end = value_date + pd.Timedelta(days=WINDOW_DAYS)
    window_mask = settled.between(window_start, window_end)
    pool = eligible.loc[window_mask]

    missing_extra = [i for i in extra_indices if i in result_df.index and i not in pool.index]
    if missing_extra:
        pool = pd.concat([pool, result_df.loc[missing_extra]])

    return pool


def verify_claims(result_df, bank_df, claims):
    valid_claims = []
    rejected_claims = []

    bank_lookup = bank_df.set_index("utr")["credit"].to_dict()

    for claim in claims:
        utr = claim["credit_utr"]
        if utr not in bank_lookup:
            claim["rejection_reason"] = f"Unknown UTR: {utr}"
            rejected_claims.append(claim)
            continue

        target_credit = bank_lookup[utr]
        entry_ids = claim["proposed_entry_ids"]

        valid_indices = []
        invalid_indices = []
        REASSIGN_THRESHOLD = 0.80
        for eid in entry_ids:
            if eid in result_df.index:
                is_unassigned = pd.isna(result_df.at[eid, "assigned_utr"])
                current_conf = result_df.at[eid, "assigned_confidence"]
                is_low_conf = (
                    not pd.isna(current_conf)
                    and current_conf < REASSIGN_THRESHOLD
                )
                if is_unassigned or is_low_conf:
                    valid_indices.append(eid)
                else:
                    invalid_indices.append(eid)
            else:
                invalid_indices.append(eid)

        if invalid_indices:
            claim["rejection_reason"] = f"Invalid or already assigned entry IDs: {invalid_indices}"
            rejected_claims.append(claim)
            continue

        if not valid_indices:
            claim["rejection_reason"] = "No valid entry IDs provided"
            rejected_claims.append(claim)
            continue

        subset_sum = result_df.loc[valid_indices, "net"].sum()

        if abs(subset_sum - target_credit) > SUM_TOLERANCE:
            claim["rejection_reason"] = f"Sum mismatch: target={target_credit:.2f}, subset_sum={subset_sum:.2f}"
            rejected_claims.append(claim)
            continue

        # --- Uniqueness gate ---------------------------------------------
        # A sum match alone is not evidence: many distinct subsets of the
        # candidate pool can hit the same target sum. We re-derive the
        # pool of entries eligible to be members of this credit and count
        # how many distinct subsets of it sum to the target (within the
        # rounding tolerance). Only a unique (or near-unique) match counts
        # as real evidence.
        pool = _candidate_pool(result_df, bank_df, utr, extra_indices=valid_indices)
        pool_entry_ids = list(pool.index)
        pool_values_paise = [_to_paise(v) for v in pool["net"]]

        cap = AMBIGUOUS_MAX_SOLUTIONS + 1
        target_paise = _to_paise(target_credit)

        sols, exhaustive = _solutions_as_entry_sets(
            pool_entry_ids, pool_values_paise, target_paise, cap=cap
        )
        all_solutions = set(sols)
        if not all_solutions and exhaustive:
            # Rounding-drift fallback: the claim already passed the ±0.50
            # sum check, so if the exact paise target provably has no
            # subset, the true match must sit a few paise off. Search
            # outward and stop at the first offset that yields anything.
            #
            # Only worth doing when the exact-target search was
            # exhaustive. If the DP already ran out of budget there, it
            # will run out at every offset too — uniqueness is
            # unprovable either way, so sweeping the band is ~100 wasted
            # DP runs per claim that cannot change the verdict.
            for delta in range(1, SUM_TOLERANCE_PAISE + 1):
                for t in (target_paise - delta, target_paise + delta):
                    band_sols, band_exhaustive = _solutions_as_entry_sets(
                        pool_entry_ids, pool_values_paise, t, cap=cap
                    )
                    all_solutions.update(band_sols)
                    exhaustive = exhaustive and band_exhaustive
                if all_solutions or not exhaustive:
                    break

        count = len(all_solutions)
        claim["solution_count"] = count

        if not exhaustive and count <= AMBIGUOUS_MAX_SOLUTIONS:
            # The search was cut short, so "few solutions found" is not the
            # same as "few solutions exist" — uniqueness was never proven.
            # Refuse rather than clear on unproven evidence.
            claim["reason_code"] = "NON_UNIQUE"
            claim["rejection_reason"] = (
                "NON_UNIQUE: solution space too large to enumerate — "
                "match is not evidenced"
            )
            rejected_claims.append(claim)
            continue

        if count <= 1:
            claim["confidence"] = 1.0
            claim["reason_code"] = "UNIQUE_MATCH"
            valid_claims.append(claim)
            continue

        if count <= AMBIGUOUS_MAX_SOLUTIONS:
            # Intersection of all valid solutions = entries present in
            # every valid subset — these are "proven" regardless of which
            # solution is the true one. Everything else is ambiguous.
            intersection = set.intersection(*[set(s) for s in all_solutions]) if all_solutions else set()
            ambiguous_members = set(valid_indices) - intersection

            claim["proposed_entry_ids"] = sorted(intersection)
            claim["ambiguous_entry_ids"] = sorted(ambiguous_members)
            claim["confidence"] = 0.75
            claim["reason_code"] = "AMBIGUOUS_PARTIAL"

            if not intersection:
                claim["reason_code"] = "NON_UNIQUE"
                claim["rejection_reason"] = (
                    f"NON_UNIQUE: {count} valid subsets exist and share no common "
                    f"members — match is not evidenced"
                )
                rejected_claims.append(claim)
                continue

            valid_claims.append(claim)
            continue

        claim["reason_code"] = "NON_UNIQUE"
        claim["rejection_reason"] = (
            f"NON_UNIQUE: {count}+ valid subsets exist — match is not evidenced"
        )
        rejected_claims.append(claim)

    return valid_claims, rejected_claims
