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

from recon.engine import load_data, reconcile
from eval.metrics import compute_metrics
from eval.harness import load_ground_truth, run_ablation, run_curve


# ---------------------------------------------------------------------------
# Cached pipeline
# ---------------------------------------------------------------------------

@st.cache_data(ttl="30m")
def run_pipeline(data_dir, max_tier, min_confidence):
    """Run the full reconciliation pipeline and return results + metrics."""
    orders, settlements, bank = load_data(data_dir)
    truth = load_ground_truth(data_dir)
    result_df, cleared = reconcile(
        orders, settlements, bank,
        max_tier=max_tier,
        min_confidence=min_confidence,
    )
    metrics = compute_metrics(result_df, truth, cleared, bank)
    return result_df, bank, cleared, metrics, truth


@st.cache_data(ttl="30m")
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


@st.cache_data(ttl="30m")
def run_ablation_cached(data_dir, max_tier_limit):
    """Cached wrapper around the evaluation harness ablation study."""
    return run_ablation(data_dir, max_tier_limit=max_tier_limit)


@st.cache_data(ttl="30m")
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
        3: "T3 · LLM agent",
        4: "T4 · Arithmetic verifier",
    }
    return labels.get(tier_num, f"T{tier_num}")


def tier_description(tier_num):
    """Return a short description for a tier."""
    descriptions = {
        0: "Joins orders and settlements using payment_id keys.",
        1: "Finds perfect date-boundary batches where totals match exactly.",
        2: "Uses Dynamic Programming to find the optimal subset of payments.",
        3: "Sends remaining unmatched credits to Gemini for structural reasoning.",
        4: "Verifies every LLM claim with deterministic arithmetic checks.",
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


def tier_type_badge(tier_num):
    """Return whether the tier is deterministic or probabilistic."""
    if tier_num <= 2 or tier_num == 4:
        return "Deterministic"
    return "Probabilistic"
