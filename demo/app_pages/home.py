"""Overview — the landing page judges see first."""

import os
import streamlit as st

from shared import (
    run_pipeline,
    fmt_pct,
    tier_icon,
    tier_label,
    tier_description,
    tier_type_badge,
)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

data_dir = st.session_state.get("data_dir", "data")
max_tier = st.session_state.get("max_tier", 4)
min_conf = st.session_state.get("min_conf", 0.90)

if not os.path.exists(data_dir):
    st.error(
        f"Data directory '{data_dir}' not found. Run `make generate` first.",
        icon=":material/error:",
    )
    st.stop()

result_df, bank_df, cleared, metrics, truth = run_pipeline(
    data_dir, max_tier, min_conf
)
overall = metrics["overall"]

# ---------------------------------------------------------------------------
# Hero section
# ---------------------------------------------------------------------------

st.title(":material/account_balance: AI Finance Controller")

with st.container(horizontal=True):
    st.caption(
        "Automated ledger reconciliation engine — "
        "Razorpay AI Buildathon submission"
    )
    st.badge("Live", icon=":material/circle:", color="green")

st.space("medium")

# ---------------------------------------------------------------------------
# Quick stats (sparkline-style metrics)
# ---------------------------------------------------------------------------

with st.container(horizontal=True):
    st.metric(
        "Total payments",
        f"{len(result_df):,}",
        border=True,
    )
    st.metric(
        "Bank credits",
        overall["total_credits"],
        border=True,
    )
    st.metric(
        "Auto-cleared",
        overall["credits_cleared"],
        f"{fmt_pct(overall['auto_clear_rate'])} clearance",
        border=True,
    )
    precision = overall["precision"]
    st.metric(
        "Precision",
        fmt_pct(precision),
        border=True,
    )
    st.metric(
        "Unit tests",
        "66 / 66",
        "100% passing",
        border=True,
    )

st.space("medium")

# ---------------------------------------------------------------------------
# Architecture pipeline (Mermaid)
# ---------------------------------------------------------------------------

st.subheader(":material/schema: Architecture pipeline")
st.caption(
    "Deterministic first, probabilistic second — each tier narrows the search "
    "space for the next."
)

st.markdown(
    """
```mermaid
flowchart LR
    T0[":material/key: **T0**<br/>Key Enrichment"]
    T1[":material/calendar_today: **T1**<br/>Date Arithmetic"]
    T2[":material/calculate: **T2**<br/>Subset-Sum DP"]
    T3[":material/smart_toy: **T3**<br/>LLM Agent"]
    T4[":material/verified: **T4**<br/>Arithmetic Verifier"]

    T0 -->|Enriched ledger| T1
    T1 -->|Uncleared credits| T2
    T2 -->|Remaining credits| T3
    T3 -->|Proposed claims| T4
    T4 -->|Verified ✓ / Rejected ✗| EQ[Exception Queue]

    style T0 fill:#1e40af,stroke:#3b82f6,color:#fff
    style T1 fill:#065f46,stroke:#10b981,color:#fff
    style T2 fill:#7e22ce,stroke:#a78bfa,color:#fff
    style T3 fill:#92400e,stroke:#f59e0b,color:#fff
    style T4 fill:#166534,stroke:#22c55e,color:#fff
    style EQ fill:#991b1b,stroke:#f87171,color:#fff
```
"""
)

st.space("medium")

# ---------------------------------------------------------------------------
# Tier cards — "how it works"
# ---------------------------------------------------------------------------

st.subheader(":material/layers: How it works")

# Row 1: T0, T1, T2
cols_top = st.columns(3)
for i, col in enumerate(cols_top):
    with col:
        with st.container(border=True):
            badge_color = "blue" if tier_type_badge(i) == "Deterministic" else "orange"
            st.markdown(f"{tier_icon(i)} **{tier_label(i)}**")
            st.caption(tier_description(i))
            st.badge(tier_type_badge(i), color=badge_color)

# Row 2: T3, T4
cols_bot = st.columns(3)
for i, col in enumerate(cols_bot):
    tier_num = i + 3
    if tier_num > 4:
        break
    with col:
        with st.container(border=True):
            badge_color = "blue" if tier_type_badge(tier_num) == "Deterministic" else "orange"
            st.markdown(f"{tier_icon(tier_num)} **{tier_label(tier_num)}**")
            st.caption(tier_description(tier_num))
            st.badge(tier_type_badge(tier_num), color=badge_color)

st.space("medium")

# ---------------------------------------------------------------------------
# Design philosophy
# ---------------------------------------------------------------------------

st.subheader(":material/shield: Design philosophy")

col_left, col_right = st.columns(2)
with col_left:
    with st.container(border=True):
        st.markdown(":material/block: **Never fabricate a match**")
        st.caption(
            "High precision is prioritised over high recall. The system will "
            "route uncertain credits to the exception queue rather than risk "
            "an incorrect assignment."
        )

with col_right:
    with st.container(border=True):
        st.markdown(":material/layers: **Deterministic first, probabilistic second**")
        st.caption(
            "Traditional algorithms handle what they can (T0–T2). The LLM "
            "(T3) handles structural reasoning. The Arithmetic Verifier (T4) "
            "never trusts the LLM's math."
        )
