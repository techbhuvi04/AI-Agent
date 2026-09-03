"""Shared utilities for the multi-page dashboard.

Centralises data loading, caching, and formatting so every page
imports the same cached results instead of re-running the engine.
"""

import os
import sys

import pandas as pd
import streamlit as st

# Ensure we can import from the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from recon.engine import load_data, reconcile, build_exception_report, export_audit_trail
from eval.metrics import compute_metrics
from eval.harness import load_ground_truth, run_ablation, run_curve


# ---------------------------------------------------------------------------
# Cached pipeline
# ---------------------------------------------------------------------------

@st.cache_resource(ttl="30m")
def run_pipeline(data_dir, max_tier, min_confidence):
    """Run the full reconciliation pipeline and return results + metrics.

    Wall-clock timing rides along inside `metrics["timing"]` so pages can
    report throughput without triggering a second reconciliation.
    """
    import time

    orders, settlements, bank = load_data(data_dir)
    truth = load_ground_truth(data_dir)

    start = time.perf_counter()
    result_df, cleared = reconcile(
        orders, settlements, bank,
        max_tier=max_tier,
        min_confidence=min_confidence,
    )
    elapsed = time.perf_counter() - start

    metrics = compute_metrics(result_df, truth, cleared, bank)
    metrics["timing"] = {
        "elapsed_seconds": elapsed,
        "num_payments": len(result_df),
        "throughput": len(result_df) / elapsed if elapsed > 0 else 0.0,
    }
    return result_df, bank, cleared, metrics, truth


@st.cache_resource(ttl="30m")
def run_tier_by_tier(data_dir, min_confidence):
    """Run each tier individually to capture incremental clearance counts."""
    orders, settlements, bank = load_data(data_dir)
    truth = load_ground_truth(data_dir)

    tier_results = []
    for tier in range(5):  # T0 through T4
        result_df, cleared = reconcile(
            orders, settlements, bank,
            max_tier=tier,
            min_confidence=min_confidence,
        )
        metrics = compute_metrics(result_df, truth, cleared, bank)
        tier_results.append({
            "tier": f"T{tier}",
            "credits_cleared": metrics["overall"]["credits_cleared"],
            "total_credits": metrics["overall"]["total_credits"],
            "precision": metrics["overall"]["precision"],
            "recall": metrics["overall"]["recall"],
            "auto_clear_rate": metrics["overall"]["auto_clear_rate"],
        })

    return tier_results


@st.cache_resource(ttl="30m")
def run_headline_metrics(data_dir, max_tier, min_confidence):
    """Value-at-risk figures for the overview page.

    Reuses the cached pipeline rather than reconciling again — throughput
    is carried on metrics["timing"] from that same run.
    """
    from recon.cashflow import build_cash_position

    result_df, bank, cleared, metrics, _truth = run_pipeline(
        data_dir, max_tier, min_confidence
    )

    run_date = pd.to_datetime(bank["value_date"]).max().date()
    report = build_exception_report(result_df, bank, cleared, run_date=run_date)
    position = build_cash_position(result_df, bank, report, run_date=run_date)

    timing = metrics.get("timing", {})
    return {
        "verified_inr": position["verified_settled"],
        "at_risk_inr": position["at_risk"],
        "elapsed_seconds": timing.get("elapsed_seconds", 0.0),
        "num_payments": timing.get("num_payments", len(result_df)),
        "throughput": timing.get("throughput", 0.0),
        "open_exceptions": len(report),
    }


@st.cache_resource(ttl="30m")
def run_exception_report(data_dir, max_tier, min_confidence):
    """Cached structured exception report for the exception queue page."""
    result_df, bank, cleared, _metrics, _truth = run_pipeline(
        data_dir, max_tier, min_confidence
    )
    run_date = pd.to_datetime(bank["value_date"]).max().date()
    return build_exception_report(result_df, bank, cleared, run_date=run_date)


@st.cache_resource(ttl="30m")
def run_cash_position(data_dir, max_tier, min_confidence, run_date=None, settlement_cycle=2):
    """Cached forward cash position for the cash flow page.

    Defaults run_date to the latest settlement date in the ledger so the
    7-day forecast lands inside the dataset's date range rather than
    today's real-world date.
    """
    from recon.cashflow import build_cash_position

    result_df, bank, cleared, _metrics, _truth = run_pipeline(
        data_dir, max_tier, min_confidence
    )
    if run_date is None:
        run_date = pd.to_datetime(bank["value_date"]).max().date()

    report = build_exception_report(result_df, bank, cleared, run_date=run_date)
    position = build_cash_position(
        result_df, bank, report, run_date=run_date, settlement_cycle=settlement_cycle
    )
    return position, report, run_date


@st.cache_resource(ttl="30m")
def run_audit_trail(data_dir, max_tier, min_confidence):
    """Cached audit trail export for the reconciliation page."""
    result_df, bank, cleared, _metrics, _truth = run_pipeline(
        data_dir, max_tier, min_confidence
    )
    run_date = pd.to_datetime(bank["value_date"]).max().date()
    report = build_exception_report(result_df, bank, cleared, run_date=run_date)
    return export_audit_trail(result_df, cleared, report)


@st.cache_resource(ttl="30m")
def run_ablation_cached(data_dir, max_tier_limit):
    """Cached wrapper around the evaluation harness ablation study."""
    return run_ablation(data_dir, max_tier_limit=max_tier_limit)


@st.cache_resource(ttl="30m")
def run_curve_cached(data_dir, max_tier):
    """Cached wrapper around the precision/auto-clear curve."""
    return run_curve(data_dir, max_tier=max_tier)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def fmt_inr(value):
    """Format a numeric value as Indian Rupees."""
    if value is None:
        return "—"
    return f"₹{value:,.2f}"


def fmt_pct(value):
    """Format a float (0–1) as a percentage string."""
    if value is None:
        return "N/A"
    return f"{value * 100:.1f}%"


def tier_label(tier_num):
    """Return a human-readable label for a tier number."""
    labels = {
        0: "T0 · Key enrichment",
        1: "T1 · Date arithmetic",
        2: "T2 · Subset-sum DP",
        3: "T3 · Break classifier",
        4: "T4 · Verifier + uniqueness gate",
    }
    return labels.get(tier_num, f"T{tier_num}")


def tier_description(tier_num):
    """Return a short description for a tier."""
    descriptions = {
        0: "Joins orders and settlements using payment_id keys.",
        1: "Finds perfect date-boundary batches where totals match exactly.",
        2: "Uses Dynamic Programming to find the optimal subset of payments.",
        3: "Classifies the break type and proposes a structured hypothesis a deterministic solver then tests.",
        4: "Verifies the arithmetic and rejects matches that many different subsets could satisfy.",
    }
    return descriptions.get(tier_num, "")


def tier_icon(tier_num):
    """Return a Material Symbols icon name for a tier."""
    icons = {
        0: ":material/key:",
        1: ":material/calendar_today:",
        2: ":material/calculate:",
        3: ":material/smart_toy:",
        4: ":material/verified:",
    }
    return icons.get(tier_num, ":material/circle:")


def inject_theme_css():
    """Visual polish layered on top of the .streamlit/config.toml theme:
    gradient hero banners, card depth/hover, tighter metric typography,
    and a colored left-rail on exception cards for at-a-glance severity.
    Pure CSS against stable data-testid hooks — no functional change."""
    st.html("""
    <style>
      [data-testid="stMetric"] {
        background: linear-gradient(160deg, rgba(255,255,255,0.06), rgba(20,20,20,0.4));
        border-radius: 12px;
        padding: 14px 16px 12px;
        transition: transform 120ms ease, border-color 120ms ease;
      }
      [data-testid="stMetric"]:hover {
        transform: translateY(-1px);
        border-color: #FFFFFF55;
      }
      [data-testid="stMetricValue"] {
        font-weight: 700;
        letter-spacing: -0.01em;
      }
      [data-testid="stMetricLabel"] {
        opacity: 0.75;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
      }

      [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 14px !important;
        transition: border-color 150ms ease, box-shadow 150ms ease;
      }
      [data-testid="stVerticalBlockBorderWrapper"]:hover {
        box-shadow: 0 4px 20px rgba(0,0,0,0.5);
      }

      .fc-hero {
        background: linear-gradient(120deg, #262626 0%, #141414 55%, #000000 100%);
        border: 1px solid #3A3A3A;
        border-radius: 18px;
        padding: 36px 40px;
        margin-bottom: 4px;
        position: relative;
        overflow: hidden;
      }
      .fc-hero::after {
        content: "";
        position: absolute;
        top: -60%; right: -10%;
        width: 340px; height: 340px;
        background: radial-gradient(circle, rgba(255,255,255,0.12), transparent 70%);
        pointer-events: none;
      }
      .fc-hero h1 {
        font-size: 2.1rem;
        font-weight: 700;
        margin: 0 0 6px 0;
        letter-spacing: -0.02em;
      }
      .fc-hero p {
        color: #D4D4D4;
        font-size: 1.02rem;
        margin: 0;
        max-width: 60ch;
      }
      .fc-hero-badges { margin-top: 16px; display: flex; gap: 8px; flex-wrap: wrap; }
      .fc-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        border: 1px solid;
      }
      .fc-pill-live {
        background: rgba(255,255,255,0.10);
        border-color: rgba(255,255,255,0.35);
        color: #F5F5F5;
      }
      .fc-pill-live::before {
        content: "";
        width: 6px; height: 6px; border-radius: 50%;
        background: #F5F5F5;
        box-shadow: 0 0 0 3px rgba(255,255,255,0.2);
      }
      .fc-pill-neutral {
        background: rgba(115,115,115,0.15);
        border-color: rgba(115,115,115,0.4);
        color: #A3A3A3;
      }

      .fc-exc-card {
        border-radius: 12px;
        border-left: 4px solid var(--fc-accent, #737373);
        background: rgba(20,20,20,0.5);
        padding: 2px 4px;
        margin-bottom: 2px;
      }

      [data-testid="stExpander"] summary {
        font-weight: 500;
      }

      /* ---------------------------------------------------------------
         Ask the controller — chat UI
         --------------------------------------------------------------- */

      /* Message rows: breathing room + a hairline divider between turns. */
      [data-testid="stChatMessage"] {
        background: transparent;
        padding: 16px 0 8px;
        gap: 12px;
        align-items: flex-start;
      }
      [data-testid="stChatMessage"] + [data-testid="stChatMessage"] {
        border-top: 1px solid rgba(255,255,255,0.06);
      }

      /* Avatars: monochrome. User = outline, assistant = filled (it speaks). */
      [data-testid="stChatMessageAvatarUser"] {
        background: #141414 !important;
        border: 1px solid #3A3A3A !important;
        color: #F5F5F5 !important;
      }
      [data-testid="stChatMessageAvatarAssistant"] {
        background: #F5F5F5 !important;
        border: 1px solid #F5F5F5 !important;
        color: #000000 !important;
      }

      /* Assistant reply reads as a card; the user's question stays plain
         so the eye lands on the answer. :has() picks the row by its avatar. */
      [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"])
        > [data-testid="stChatMessageContent"] {
        background: linear-gradient(160deg, rgba(255,255,255,0.05), rgba(20,20,20,0.35));
        border: 1px solid #2A2A2A;
        border-radius: 3px 12px 12px 12px;
        padding: 12px 16px;
      }
      [data-testid="stChatMessageContent"] { line-height: 1.55; }
      [data-testid="stChatMessageContent"] p:last-child { margin-bottom: 0; }

      /* Input bar: lift it off the page, match the 8px radius system. */
      [data-testid="stChatInput"] {
        border: 1px solid #3A3A3A;
        border-radius: 12px;
        background: #0C0C0C;
        box-shadow: 0 -2px 24px rgba(0,0,0,0.6);
      }
      [data-testid="stChatInput"]:focus-within { border-color: #6E6E6E; }
      [data-testid="stChatInput"] textarea::placeholder { color: #6E6E6E; }

      /* Empty state */
      .fc-chat-empty {
        text-align: center;
        max-width: 44ch;
        margin: 30px auto 4px;
        color: #A3A3A3;
      }
      .fc-chat-empty-icon {
        width: 44px; height: 44px;
        margin: 0 auto 14px;
        border-radius: 12px;
        border: 1px solid #3A3A3A;
        background: #141414;
        display: flex; align-items: center; justify-content: center;
        font-size: 22px; color: #F5F5F5;
      }
      .fc-chat-empty p { margin: 0; font-size: 0.92rem; line-height: 1.5; }

      .fc-chip-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #737373;
        margin: 22px 0 8px;
      }
      /* Suggestion chips: quiet until hovered, left-aligned like real prompts. */
      .fc-chip-label + [data-testid="stHorizontalBlock"] [data-testid="stButton"] button {
        background: #101010;
        border: 1px solid #2A2A2A;
        color: #D4D4D4;
        font-weight: 400;
        text-align: left;
        justify-content: flex-start;
        transition: border-color 120ms ease, background 120ms ease;
      }
      .fc-chip-label + [data-testid="stHorizontalBlock"] [data-testid="stButton"] button:hover:not(:disabled) {
        border-color: #6E6E6E;
        background: #171717;
        color: #FFFFFF;
      }
    </style>
    """)


def tier_type_badge(tier_num):
    """Return whether the tier is deterministic or probabilistic."""
    if tier_num <= 2 or tier_num == 4:
        return "Deterministic"
    return "Probabilistic"
