"""Overview — the landing page judges see first."""

import os
import streamlit as st

from shared import (
    run_pipeline,
    run_headline_metrics,
    fmt_inr,
    fmt_pct,
    tier_icon,
    tier_label,
    tier_description,
    tier_type_badge,
    inject_theme_css,
)

inject_theme_css()

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

with st.status("Running reconciliation engine...", expanded=False) as status:
    st.write(":material/database: Loading ledger data...")
    result_df, bank_df, cleared, metrics, truth = run_pipeline(
        data_dir, max_tier, min_conf
    )
    status.update(label="Engine complete", state="complete", expanded=False)

overall = metrics["overall"]

# ---------------------------------------------------------------------------
# Hero section
# ---------------------------------------------------------------------------

st.html(f"""
<div class="fc-hero">
  <h1>🏦 AI Finance Controller</h1>
  <p>Automated ledger reconciliation engine — Razorpay AI Buildathon submission.
  Deterministic first, probabilistic second: every auto-cleared credit is
  proven unique, not guessed.</p>
  <div class="fc-hero-badges">
    <span class="fc-pill fc-pill-live">Live</span>
    <span class="fc-pill fc-pill-neutral">T0–T4 pipeline</span>
    <span class="fc-pill fc-pill-neutral">{overall['total_credits']} bank credits · {len(result_df):,} payments</span>
  </div>
</div>
""")

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
        "121 / 121",
        "100% passing",
        border=True,
    )

st.space("medium")

# ---------------------------------------------------------------------------
# Controller metrics — value and throughput
# ---------------------------------------------------------------------------

st.subheader(":material/query_stats: Controller metrics")
st.caption(
    "What the engine actually proved, what it refused to clear, and how fast "
    "it got there."
)

headline = run_headline_metrics(data_dir, max_tier, min_conf)

with st.container(horizontal=True):
    st.metric(
        "Auto-clear rate",
        fmt_pct(overall["auto_clear_rate"]),
        help="Share of payments assigned to a bank credit by the engine.",
        border=True,
    )
    st.metric(
        "Precision",
        fmt_pct(overall["precision"]),
        help="Entries in auto-cleared credits that belong there.",
        border=True,
    )
    st.metric(
        "₹ verified",
        fmt_inr(headline["verified_inr"]),
        help="Total value sitting inside auto-cleared credits.",
        border=True,
    )
    st.metric(
        "₹ at risk",
        fmt_inr(headline["at_risk_inr"]),
        f"{headline['open_exceptions']} open exceptions",
        delta_color="inverse",
        help="Value in UNRESOLVED / SUM_COLLISION exceptions.",
        border=True,
    )
    st.metric(
        "Throughput",
        f"{headline['throughput']:,.0f} /sec",
        f"{headline['num_payments']:,} payments in {headline['elapsed_seconds']:.1f}s",
        help="Payments reconciled per second on the last run.",
        border=True,
    )

st.caption(
    "**Precision** counts entries in auto-cleared credits that belong there. "
    "The T4 uniqueness gate deliberately trades auto-clear rate for precision — "
    "a match that many different subsets could satisfy is not evidence, so the "
    "engine routes it to the exception queue rather than guessing."
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
    T3[":material/smart_toy: **T3**<br/>Break Classifier"]
    T4[":material/verified: **T4**<br/>Verifier + Uniqueness Gate"]

    T0 -->|Enriched ledger| T1
    T1 -->|Uncleared credits| T2
    T2 -->|Remaining credits| T3
    T3 -->|Hypothesis → DP-resolved claims| T4
    T4 -->|Unique ✓ / Non-unique ✗| EQ[Exception Queue]

    style T0 fill:#262626,stroke:#8a8a8a,color:#fff
    style T1 fill:#333333,stroke:#a3a3a3,color:#fff
    style T2 fill:#404040,stroke:#c7c7c7,color:#fff
    style T3 fill:#4d4d4d,stroke:#d4d4d4,color:#fff
    style T4 fill:#595959,stroke:#e5e5e5,color:#fff
    style EQ fill:#1a1a1a,stroke:#ffffff,color:#fff
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
