import json
import os
from datetime import date, timedelta

import pandas as pd

from recon.t2_subset import _to_paise, _solve_credit
from recon import llm_client

BREAK_CLASSIFICATIONS = [
    "late_settlement",
    "netting_split",
    "refund_batch",
    "duplicate_utr",
    "rounding_drift",
    "missing_order",
    "unknown",
]

HYPOTHESIS_ACTIONS = [
    "expand_window",
    "merge_with_utr",
    "accept_with_tolerance",
    "manual_review",
]

REASSIGN_THRESHOLD = 0.80
SUM_TOLERANCE_PAISE = 50
DEFAULT_EXPAND_DAYS = 7
MERGE_WINDOW_DAYS = 7


def _get_client():
    return llm_client.get_client()


# ---------------------------------------------------------------------------
# Legacy prompt (subset-proposal). Kept as a standalone utility — the main
# flow no longer asks the LLM to propose entry_ids (see the break-
# classification flow below), but this remains available/tested as a
# building block.
# ---------------------------------------------------------------------------

def _build_prompt(credit_utr, credit_amount, value_date, candidates_df):
    lines = []
    for idx, row in candidates_df.iterrows():
        lines.append(
            f"{idx:>6} | {row['payment_id']:<14} | {row['net']:>10.2f} | {row['settled_at']}"
        )
    table = "\n".join(lines)

    return f"""You are a payment reconciliation expert. Match settlement entries to a bank credit.

## Bank Credit
- UTR: {credit_utr}
- Amount: {credit_amount:.2f}
- Value Date: {value_date}

## Candidate Settlement Entries (unassigned)
entry_id | payment_id     |        net | settled_at
---------|----------------|------------|------------
{table}

## Rules
1. The sum of selected entries' net values MUST equal {credit_amount:.2f} (tolerance ±0.50)
2. Typical batch size is 60–140 entries
3. Entries with settled_at near the value_date are more likely members
4. Negative net values are refunds or chargeback adjustments — they reduce the credit total
5. entry_id is the unique row identifier — use it in your response

Respond with ONLY this JSON (no markdown, no explanation outside the JSON):
{{
  "credit_utr": "{credit_utr}",
  "proposed_entry_ids": [integer entry_ids whose net sums to {credit_amount:.2f}],
  "reasoning": "one-sentence explanation",
  "confidence": 0.0
}}"""


def _parse_response(text):
    if text is None:
        return None
    cleaned = text.strip()
    if "```" in cleaned:
        segments = cleaned.split("```")
        for segment in segments:
            segment = segment.strip()
            if segment.startswith("json"):
                segment = segment[4:].strip()
            if segment.startswith("{"):
                try:
                    return json.loads(segment)
                except json.JSONDecodeError:
                    continue

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(cleaned[start:end])
            except json.JSONDecodeError:
                return None
    return None


def _call_llm(client, prompt):
    try:
        response = client.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"  T3: LLM call failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Break-classification flow.
#
# The LLM is bad at subset-sum (T2's DP already solves that exactly). Its
# useful job is structural: look at the *shape* of a break — not the raw
# candidate rows — and classify what kind of break it is, then propose a
# STRUCTURED hypothesis (expand the window? merge with a neighbouring
# credit? accept a rounding-drift tolerance? route to a human?). A
# deterministic resolver then tests that hypothesis; nothing the LLM says
# is trusted without the DP re-verifying it, and T4 still re-checks the
# arithmetic (and uniqueness) of whatever comes out.
# ---------------------------------------------------------------------------

def _eligible_pool(df, used_indices):
    conf = pd.to_numeric(df["assigned_confidence"], errors="coerce").fillna(0)
    mask = (
        ~df.index.isin(used_indices)
        & (df["assigned_utr"].isna() | (conf < REASSIGN_THRESHOLD))
    )
    return df.loc[mask]


def _window_candidates(df, used_indices, start, end):
    pool = _eligible_pool(df, used_indices)
    dates = pd.to_datetime(pool["settled_at"]).dt.date
    return pool.loc[dates.between(start, end)]


def compute_diagnostics(df, used_indices, credit_row, bank_sorted, pos):
    """Structural diagnostics for one uncleared credit — the input handed
    to the LLM classifier instead of the raw candidate rows."""
    utr = credit_row["utr"]
    value_date = pd.to_datetime(credit_row["value_date"]).date()
    credit_paise = _to_paise(float(credit_row["credit"]))

    if pos == 0:
        window_start = date(2000, 1, 1)
    else:
        prev_vd = pd.to_datetime(bank_sorted.iloc[pos - 1]["value_date"]).date()
        window_start = prev_vd + timedelta(days=1)
    window_end = value_date

    candidates = _window_candidates(df, used_indices, window_start, window_end)
    total_candidates_paise = sum(_to_paise(v) for v in candidates["net"]) if len(candidates) else 0

    nearby_credits = []
    if pos > 0:
        prow = bank_sorted.iloc[pos - 1]
        nearby_credits.append({
            "utr": prow["utr"], "amount": float(prow["credit"]), "position": "prev",
        })
    if pos < len(bank_sorted) - 1:
        nrow = bank_sorted.iloc[pos + 1]
        nearby_credits.append({
            "utr": nrow["utr"], "amount": float(nrow["credit"]), "position": "next",
        })

    return {
        "credit_utr": utr,
        "credit_amount": float(credit_row["credit"]),
        "value_date": str(value_date),
        "window_start": str(window_start),
        "window_end": str(window_end),
        "window_deficit": total_candidates_paise < credit_paise,
        "excess_paise": total_candidates_paise - credit_paise,
        "num_candidates": int(len(candidates)),
        "nearby_credits": nearby_credits,
        "negative_entries_present": bool(len(candidates) and (candidates["net"] < 0).any()),
    }


def _build_classification_prompt(diagnostics):
    nearby_lines = "\n".join(
        f"  - {n['position']}: {n['utr']} (₹{n['amount']:.2f})"
        for n in diagnostics["nearby_credits"]
    ) or "  (none)"

    return f"""You are a payment reconciliation expert. A bank credit could not be matched \
to a set of settlement entries by deterministic subset-sum. Classify the break and \
propose a STRUCTURED hypothesis for how to resolve it — do NOT propose specific \
entry_ids; a deterministic solver will test your hypothesis and a separate \
arithmetic verifier will check the result.

## Bank Credit
- UTR: {diagnostics['credit_utr']}
- Amount: {diagnostics['credit_amount']:.2f}
- Value Date: {diagnostics['value_date']}
- Strict window: {diagnostics['window_start']} to {diagnostics['window_end']}

## Diagnostics
- window_deficit (candidates in window can't reach the credit amount): {diagnostics['window_deficit']}
- excess_paise (candidate total minus credit target, in paise): {diagnostics['excess_paise']}
- num_candidates in window: {diagnostics['num_candidates']}
- negative_entries_present (refunds/chargebacks in window): {diagnostics['negative_entries_present']}
- nearby_credits:
{nearby_lines}

## Break classifications
- late_settlement: credit exceeds window, members settled T+3/T+4
- netting_split: this credit is part of a split across 2 UTRs
- refund_batch: batch contains refunds/chargebacks reducing total
- duplicate_utr: two bank rows reference the same underlying batch
- rounding_drift: <₹1 discrepancy from fee rounding
- missing_order: some payment_ids have no order record
- unknown: none of the above clearly applies

## Hypothesis actions
- expand_window: widen the settlement window by expand_days on each side
- merge_with_utr: this credit's members are split with an adjacent UTR — set merge_utr
- accept_with_tolerance: accept the window as-is within rounding tolerance (rounding_drift only)
- manual_review: route to the exception queue for a human

Respond with ONLY this JSON (no markdown, no explanation outside the JSON):
{{
  "credit_utr": "{diagnostics['credit_utr']}",
  "break_classification": "one of {'|'.join(BREAK_CLASSIFICATIONS)}",
  "hypothesis": {{
    "action": "one of {'|'.join(HYPOTHESIS_ACTIONS)}",
    "expand_days": 0,
    "merge_utr": ""
  }},
  "reasoning": "one sentence",
  "confidence": 0.0
}}"""


def resolve_hypothesis(df, used_indices, bank_lookup, diagnostics, break_classification, hypothesis):
    """Deterministic resolver: takes the LLM's structured hypothesis and
    tests it with the actual subset-sum solver. Returns a set of entry_ids
    if the hypothesis resolves to a candidate match, else None. Nothing
    here is trusted output from the LLM — every branch re-runs real
    arithmetic against the ledger."""
    action = hypothesis.get("action")
    utr = diagnostics["credit_utr"]
    credit_paise = _to_paise(diagnostics["credit_amount"])
    window_start = pd.to_datetime(diagnostics["window_start"]).date()
    window_end = pd.to_datetime(diagnostics["window_end"]).date()

    if action == "expand_window":
        try:
            expand_days = int(hypothesis.get("expand_days") or DEFAULT_EXPAND_DAYS)
        except (TypeError, ValueError):
            expand_days = DEFAULT_EXPAND_DAYS
        expand_days = max(0, min(expand_days, 30))

        start = window_start - timedelta(days=expand_days)
        end = window_end + timedelta(days=expand_days)
        candidates = _window_candidates(df, used_indices, start, end)
        if len(candidates) == 0:
            return None
        members = _solve_credit(
            list(candidates.index),
            [_to_paise(v) for v in candidates["net"]],
            credit_paise,
        )
        return members

    if action == "merge_with_utr":
        merge_utr = hypothesis.get("merge_utr")
        if not merge_utr or merge_utr not in bank_lookup:
            return None

        # Widen the window to plausibly cover both credits' true members,
        # solve for the COMBINED target (this credit + the merge partner),
        # then re-solve within that combined member pool for just this
        # credit's own amount — this is what recovers a netting_split
        # (one settlement batch paid out across two bank credits).
        start = window_start - timedelta(days=MERGE_WINDOW_DAYS)
        end = window_end + timedelta(days=MERGE_WINDOW_DAYS)
        candidates = _window_candidates(df, used_indices, start, end)
        if len(candidates) == 0:
            return None

        combined_target = credit_paise + _to_paise(bank_lookup[merge_utr])
        c_indices = list(candidates.index)
        c_nets = [_to_paise(v) for v in candidates["net"]]
        combined_members = _solve_credit(c_indices, c_nets, combined_target)
        if not combined_members:
            return None

        member_list = list(combined_members)
        member_nets = [_to_paise(df.at[i, "net"]) for i in member_list]
        this_members = _solve_credit(member_list, member_nets, credit_paise)
        return this_members

    if action == "accept_with_tolerance":
        if break_classification != "rounding_drift":
            return None
        candidates = _window_candidates(df, used_indices, window_start, window_end)
        if len(candidates) == 0:
            return None
        total = sum(_to_paise(v) for v in candidates["net"])
        if abs(total - credit_paise) <= SUM_TOLERANCE_PAISE:
            return set(candidates.index)
        return None

    # manual_review, unknown action, or anything unrecognised → no
    # automatic resolution; the credit stays in the exception queue.
    return None


def _classification_solve(result_df, bank_df, already_cleared, client):
    used_indices = set()
    for indices in already_cleared.values():
        for idx in indices:
            if result_df.at[idx, "assigned_confidence"] >= REASSIGN_THRESHOLD:
                used_indices.add(idx)

    bank_sorted = bank_df.sort_values("value_date").reset_index(drop=True)
    bank_lookup = bank_df.set_index("utr")["credit"].to_dict()

    claims = []
    for pos, credit_row in bank_sorted.iterrows():
        utr = credit_row["utr"]
        if utr in already_cleared:
            continue

        diagnostics = compute_diagnostics(result_df, used_indices, credit_row, bank_sorted, pos)
        prompt = _build_classification_prompt(diagnostics)

        print(f"  T3: classifying break for {utr}...")
        raw = _call_llm(client, prompt)
        classification = _parse_response(raw)

        if classification is None or "hypothesis" not in classification:
            print(f"  T3: {utr} — failed to parse classification response")
            continue

        break_classification = classification.get("break_classification", "unknown")
        hypothesis = classification.get("hypothesis") or {}

        members = resolve_hypothesis(
            result_df, used_indices, bank_lookup, diagnostics, break_classification, hypothesis
        )

        if not members:
            print(
                f"  T3: {utr} — classified {break_classification} / "
                f"action={hypothesis.get('action')} — no automatic resolution"
            )
            continue

        conf = float(classification.get("confidence", 0.6))
        if hypothesis.get("action") == "accept_with_tolerance":
            conf = min(conf, 0.85)

        claim = {
            "credit_utr": utr,
            "proposed_entry_ids": sorted(members),
            "reasoning": classification.get("reasoning", ""),
            "confidence": conf,
            "break_classification": break_classification,
        }
        claims.append(claim)
        used_indices.update(members)
        print(
            f"  T3: {utr} — {break_classification} / {hypothesis.get('action')} "
            f"→ {len(members)} members"
        )

    return claims


# ---------------------------------------------------------------------------
# Deterministic heuristic fallback (no LLM available). Unchanged behaviour
# from before the break-classification rewrite — this must keep working
# with GROQ_API_KEY unset.
# ---------------------------------------------------------------------------

def _try_heuristic_match(df, credit_paise, value_date, used_indices):
    """Try to match a credit using wider windows and refund-aware logic.

    T2 fails on credits containing refunds because its DP treats all
    candidates together and skips negative values.  Here we split:
      • negatives (refunds/chargebacks) → always include
      • positives → DP to find the subset summing to the adjusted target

    Also considers low-confidence entries for reassignment.
    """
    from recon.t2_subset import _dp_find_subset

    TOLERANCE_PAISE = 50  # 0.50 INR

    for expansion in [7, 14]:
        window_start = value_date - timedelta(days=expansion)
        window_end = value_date + timedelta(days=expansion)

        candidate_mask = (
            ~df.index.isin(used_indices)
            & (
                df["assigned_utr"].isna()
                | (df["assigned_confidence"] < REASSIGN_THRESHOLD)
            )
            & df["_settled_date"].between(window_start, window_end)
        )
        candidates = df.loc[candidate_mask]
        if len(candidates) == 0:
            continue

        # Cap pool size to keep DP tractable; prefer entries closest
        # to the credit's value_date (most likely to truly belong).
        MAX_HEURISTIC_CANDIDATES = 150
        if len(candidates) > MAX_HEURISTIC_CANDIDATES:
            candidates = candidates.copy()
            candidates["_dist"] = candidates["_settled_date"].apply(
                lambda d: abs((d - value_date).days)
            )
            candidates = candidates.nsmallest(MAX_HEURISTIC_CANDIDATES, "_dist")


        neg_mask = candidates["net"] < 0
        negatives = candidates[neg_mask]
        positives = candidates[~neg_mask]

        neg_indices = list(negatives.index)
        neg_sum_paise = sum(_to_paise(v) for v in negatives["net"])

        pos_target = credit_paise - neg_sum_paise
        if pos_target < 0:
            continue

        pos_indices = list(positives.index)
        pos_values = [_to_paise(v) for v in positives["net"]]
        total_pos = sum(pos_values)

        # All positives in the window match exactly
        if abs(total_pos - pos_target) <= TOLERANCE_PAISE:
            return set(neg_indices + pos_indices)

        # More positives than needed → exclude a subset summing to the excess
        if total_pos > pos_target:
            excess = total_pos - pos_target
            excl = _dp_find_subset(pos_values, excess, max_iter=3_000_000)
            if excl is not None:
                exclude_set = {pos_indices[i] for i in excl}
                result = set(neg_indices)
                result.update(i for i in pos_indices if i not in exclude_set)
                return result

    return None


def _heuristic_solve(result_df, bank_df, already_cleared):
    """Deterministic heuristic fallback when no LLM is available.

    Uses wider date windows and refund-aware subset matching to clear
    credits that T2 missed.
    """
    df = result_df.copy()
    df["_settled_date"] = pd.to_datetime(df["settled_at"]).dt.date
    bank_sorted = bank_df.sort_values("value_date")

    claims = []
    used_indices = set()
    for indices in already_cleared.values():
        for idx in indices:
            # Only treat high-confidence assignments as truly used;
            # low-confidence entries can be reassigned by the heuristic.
            if df.at[idx, "assigned_confidence"] >= REASSIGN_THRESHOLD:
                used_indices.add(idx)

    for _, credit_row in bank_sorted.iterrows():
        utr = credit_row["utr"]
        if utr in already_cleared:
            continue

        credit_paise = _to_paise(float(credit_row["credit"]))
        value_date = pd.to_datetime(credit_row["value_date"]).date()

        matched = _try_heuristic_match(df, credit_paise, value_date, used_indices)
        if matched is not None:
            used_indices.update(matched)
            claims.append({
                "credit_utr": utr,
                "proposed_entry_ids": list(matched),
                "reasoning": "heuristic: refund-aware wide-window match",
                "confidence": 0.70,
                "break_classification": "unknown",
            })
            print(f"  T3: {utr} — heuristic matched {len(matched)} entries")

    return claims


def run(result_df, bank_df, already_cleared):
    client = _get_client()
    if client is None:
        print("  T3: using deterministic heuristic (set GROQ_API_KEY for LLM)")
        return _heuristic_solve(result_df, bank_df, already_cleared)

    unassigned_mask = result_df["assigned_utr"].isna()
    if unassigned_mask.sum() == 0:
        return []

    return _classification_solve(result_df, bank_df, already_cleared, client)


def apply_claims(result_df, claims):
    df = result_df.copy()
    new_cleared = {}

    if "break_classification" not in df.columns:
        df["break_classification"] = pd.Series(dtype="object", index=df.index)
    if "solution_count" not in df.columns:
        df["solution_count"] = pd.Series(dtype="float", index=df.index)
    if "reason_code" not in df.columns:
        df["reason_code"] = pd.Series(dtype="object", index=df.index)

    for claim in claims:
        utr = claim["credit_utr"]
        indices = []

        conf = float(claim.get("confidence", 0.0))
        break_classification = claim.get("break_classification")
        solution_count = claim.get("solution_count")
        reason_code = claim.get("reason_code")

        for entry_id in claim["proposed_entry_ids"]:
            if entry_id not in df.index:
                continue
            is_unassigned = pd.isna(df.at[entry_id, "assigned_utr"])
            current_conf = df.at[entry_id, "assigned_confidence"]
            is_low_conf = (
                not pd.isna(current_conf)
                and current_conf < REASSIGN_THRESHOLD
            )
            if is_unassigned or is_low_conf:
                df.at[entry_id, "assigned_utr"] = utr
                df.at[entry_id, "assigned_tier"] = 3
                df.at[entry_id, "assigned_confidence"] = conf
                if break_classification:
                    df.at[entry_id, "break_classification"] = break_classification
                if solution_count is not None:
                    df.at[entry_id, "solution_count"] = solution_count
                if reason_code:
                    df.at[entry_id, "reason_code"] = reason_code
                indices.append(entry_id)

        if indices:
            new_cleared[utr] = indices

    return df, new_cleared
