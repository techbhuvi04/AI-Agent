import json
import os

import pandas as pd


LLM_MODEL = os.environ.get("LLM_MODEL", "gemini-2.0-flash")


def _get_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None

    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        return genai.GenerativeModel(LLM_MODEL)
    except ImportError:
        return None


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


def run(result_df, bank_df, already_cleared):
    client = _get_client()
    if client is None:
        print("  T3: skipped (set GEMINI_API_KEY to enable)")
        return []

    unassigned_mask = result_df["assigned_utr"].isna()
    unassigned = result_df[unassigned_mask]

    if len(unassigned) == 0:
        return []

    claims = []
    bank_sorted = bank_df.sort_values("value_date")

    for _, credit_row in bank_sorted.iterrows():
        utr = credit_row["utr"]
        if utr in already_cleared:
            continue

        prompt = _build_prompt(
            utr,
            float(credit_row["credit"]),
            credit_row["value_date"],
            unassigned,
        )

        print(f"  T3: querying LLM for {utr} ({len(unassigned)} candidates)...")
        raw = _call_llm(client, prompt)
        claim = _parse_response(raw)

        if claim is None or "proposed_entry_ids" not in claim:
            print(f"  T3: {utr} — failed to parse LLM response")
            continue

        claim["credit_utr"] = utr
        valid_ids = [
            int(eid)
            for eid in claim["proposed_entry_ids"]
            if int(eid) in unassigned.index
        ]
        claim["proposed_entry_ids"] = valid_ids

        if valid_ids:
            claims.append(claim)
            print(f"  T3: {utr} — {len(valid_ids)} proposed members")
        else:
            print(f"  T3: {utr} — no valid entry IDs")

    return claims


def apply_claims(result_df, claims):
    df = result_df.copy()
    new_cleared = {}

    for claim in claims:
        utr = claim["credit_utr"]
        indices = []

        conf = float(claim.get("confidence", 0.0))

        for entry_id in claim["proposed_entry_ids"]:
            if entry_id in df.index and pd.isna(df.at[entry_id, "assigned_utr"]):
                df.at[entry_id, "assigned_utr"] = utr
                df.at[entry_id, "assigned_tier"] = 3
                df.at[entry_id, "assigned_confidence"] = conf
                indices.append(entry_id)

        if indices:
            new_cleared[utr] = indices

    return df, new_cleared
