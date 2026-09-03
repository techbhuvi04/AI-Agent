"""Natural-language Q&A over the reconciliation output.

The LLM never sees the raw ledger — it sees a compact structured summary
(<2000 tokens) built from the exception report, the cash position, and the
cleared-credit totals. That keeps the context small, keeps costs bounded,
and means the model is answering from figures the deterministic engine
already verified rather than from rows it might misread.
"""

import re
import time

import pandas as pd

from recon import llm_client

SYSTEM_PROMPT = """You are a finance controller assistant.
Answer the question using only the structured data provided.
Never invent numbers. If the data doesn't support a confident answer, say so.
Be concise — a few sentences. Quote figures exactly as given."""

MAX_EXCEPTION_ROWS = 25


def is_available():
    return llm_client.is_available()


def _get_client():
    return llm_client.get_client()


def _fmt_inr(v):
    return f"₹{v:,.2f}"


def build_context(result_df, bank_df, exception_report_df, cash_position_dict):
    """Serialise the structured data into a compact context string.

    Deliberately summarises rather than dumping rows — a 50k-row ledger
    becomes aggregate counts plus the top exceptions by value.
    """
    lines = []

    # --- Reconciliation summary -----------------------------------------
    total_payments = len(result_df)
    assigned = int(result_df["assigned_utr"].notna().sum())
    cleared_value = float(result_df.loc[result_df["assigned_utr"].notna(), "net"].sum())
    total_credits = len(bank_df)
    cleared_credits = int(result_df["assigned_utr"].dropna().nunique())

    lines.append("## Reconciliation summary")
    lines.append(f"- Total payments: {total_payments:,}")
    lines.append(f"- Payments auto-assigned: {assigned:,} ({assigned / total_payments:.1%})" if total_payments else "- Payments auto-assigned: 0")
    lines.append(f"- Bank credits: {total_credits}, auto-cleared: {cleared_credits}")
    lines.append(f"- Value in auto-cleared credits: {_fmt_inr(cleared_value)}")

    if "assigned_tier" in result_df.columns:
        tier_counts = result_df["assigned_tier"].dropna().astype(int).value_counts().sort_index()
        if len(tier_counts):
            tiers = ", ".join(f"T{t}: {c:,}" for t, c in tier_counts.items())
            lines.append(f"- Payments cleared per tier — {tiers}")

    # --- Cash position ---------------------------------------------------
    if cash_position_dict:
        cp = cash_position_dict
        lines.append("")
        lines.append("## Cash position")
        lines.append(f"- Verified settled: {_fmt_inr(cp.get('verified_settled', 0))}")
        lines.append(f"- At risk: {_fmt_inr(cp.get('at_risk', 0))}")
        lines.append(f"- Expected inflow next 7 days: {_fmt_inr(cp.get('expected_inflow_total', 0))}")
        ci = cp.get("confidence_interval")
        if ci:
            lines.append(f"- Forecast range: {_fmt_inr(ci[0])} to {_fmt_inr(ci[1])}")
        inflow = cp.get("expected_inflow") or {}
        if inflow:
            daily = ", ".join(f"{d}: {_fmt_inr(v)}" for d, v in inflow.items())
            lines.append(f"- Daily expected inflow — {daily}")
        by_age = cp.get("cash_at_risk_by_age") or {}
        if by_age:
            aged = ", ".join(f"{k}: {_fmt_inr(v)}" for k, v in by_age.items())
            lines.append(f"- Cash at risk by age — {aged}")

    # --- Exceptions ------------------------------------------------------
    lines.append("")
    lines.append("## Exception queue")
    if exception_report_df is None or len(exception_report_df) == 0:
        lines.append("- No open exceptions.")
    else:
        exc = exception_report_df
        lines.append(f"- Open exceptions: {len(exc)}")
        lines.append(f"- Total value at risk: {_fmt_inr(float(exc['credit_amount'].sum()))}")
        lines.append(f"- Oldest exception age: {int(exc['age_days'].max())} days")

        code_counts = exc["break_code"].value_counts()
        codes = ", ".join(f"{c}: {n}" for c, n in code_counts.items())
        lines.append(f"- Break codes — {codes}")

        mat_counts = exc["materiality"].value_counts()
        mats = ", ".join(f"{m}: {n}" for m, n in mat_counts.items())
        lines.append(f"- Materiality — {mats}")

        lines.append("")
        lines.append(f"### Top {min(MAX_EXCEPTION_ROWS, len(exc))} exceptions by value")
        lines.append("utr | amount | value_date | age_days | materiality | break_code | delta | suggested_action")
        top = exc.nlargest(min(MAX_EXCEPTION_ROWS, len(exc)), "credit_amount")
        for _, r in top.iterrows():
            lines.append(
                f"{r['credit_utr']} | {_fmt_inr(r['credit_amount'])} | {r['value_date']} | "
                f"{r['age_days']} | {r['materiality']} | {r['break_code']} | "
                f"{_fmt_inr(r['delta_inr'])} | {r['suggested_action']}"
            )

    return "\n".join(lines)


def answer_question(question, result_df, bank_df, exception_report_df, cash_position_dict):
    """Answer a natural-language question about the reconciliation state.

    Returns an answer string. Degrades gracefully when GROQ_API_KEY is
    unset rather than raising.
    """
    client = _get_client()
    if client is None:
        return "Set GROQ_API_KEY to enable natural language queries."

    context = build_context(result_df, bank_df, exception_report_df, cash_position_dict)

    prompt = f"""{SYSTEM_PROMPT}

# Structured reconciliation data
{context}

# Question
{question}

# Answer"""

    # One retry on a transient per-minute rate limit — the free-tier token
    # window refills every ~60s and the 429 body tells us how long to wait.
    for attempt in (1, 2):
        try:
            response = client.generate_content(prompt, max_tokens=600)
            return (response.text or "").strip() or "The model returned an empty response."
        except Exception as e:
            msg = str(e)
            is_rate_limit = "rate_limit_exceeded" in msg or "429" in msg
            if is_rate_limit and attempt == 1:
                m = re.search(r"try again in ([\d.]+)\s*s", msg)
                wait = min(float(m.group(1)) + 0.5, 20.0) if m else 5.0
                time.sleep(wait)
                continue
            if is_rate_limit:
                return (
                    "The LLM provider's usage limit was hit for this model. "
                    "Try again in a minute, or set LLM_MODEL to a different "
                    "model in your .env file."
                )
            print(f"  QA agent: LLM call failed: {e}")
            return "The assistant hit an error answering that — please try again."
