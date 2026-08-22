"""Analytics — ablation heatmap, precision/recall curve, per-break charts."""

import os

import altair as alt
import pandas as pd
import streamlit as st

from shared import (
    run_pipeline,
    run_ablation_cached,
    run_curve_cached,
    fmt_pct,
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

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title(":material/analytics: Analytics")
st.caption(
    "Quantitative evidence that the engine works — ablation study, "
    "precision/recall curves, and per-break-type analysis."
)

# Get current metrics for the summary row
result_df, bank_df, cleared, metrics, truth = run_pipeline(
    data_dir, max_tier, min_conf
)
overall = metrics["overall"]

# Summary metrics row
with st.container(horizontal=True):
    st.metric("Precision", fmt_pct(overall["precision"]), border=True)
    st.metric("Recall", fmt_pct(overall["recall"]), border=True)
    st.metric("F1 score", fmt_pct(overall["f1"]), border=True)
    st.metric(
        "Auto-clear rate",
        fmt_pct(overall["auto_clear_rate"]),
        border=True,
    )

st.space("medium")

# ---------------------------------------------------------------------------
# Tier ablation heatmap
# ---------------------------------------------------------------------------

st.subheader(":material/grid_on: Tier ablation heatmap")
st.caption(
    "Per-break-type recall at each tier combination. Darker green = higher recall. "
    "Each column adds one more tier to the pipeline."
)

ablation_results = run_ablation_cached(data_dir, max_tier)

# Build a matrix: rows = break types, columns = tier combos
heat_rows = []
for label, result in ablation_results.items():
    for bt, bt_metrics in result["per_break"].items():
        heat_rows.append({
            "Tier combination": label,
            "Break type": bt,
            "Recall": bt_metrics["recall"],
        })

if heat_rows:
    heat_df = pd.DataFrame(heat_rows)

    # Order the tier combinations correctly
    tier_order = list(ablation_results.keys())

    heatmap = (
        alt.Chart(heat_df)
        .mark_rect(cornerRadius=4)
        .encode(
            x=alt.X(
                "Tier combination:N",
                sort=tier_order,
                title=None,
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y("Break type:N", title=None),
            color=alt.Color(
                "Recall:Q",
                scale=alt.Scale(scheme="greens", domain=[0, 1]),
                legend=alt.Legend(title="Recall"),
            ),
            tooltip=[
                alt.Tooltip("Tier combination:N", title="Tiers"),
                alt.Tooltip("Break type:N", title="Break type"),
                alt.Tooltip("Recall:Q", title="Recall", format=".3f"),
            ],
        )
    )

    text_layer = (
        alt.Chart(heat_df)
        .mark_text(fontSize=12, fontWeight="bold")
        .encode(
            x=alt.X("Tier combination:N", sort=tier_order),
            y=alt.Y("Break type:N"),
            text=alt.Text("Recall:Q", format=".2f"),
            color=alt.condition(
                alt.datum.Recall > 0.5,
                alt.value("#0F172A"),
                alt.value("#F1F5F9"),
            ),
        )
    )

    st.altair_chart(heatmap + text_layer)

    # Also show the overall metrics table beneath
    ablation_summary_rows = []
    for label, result in ablation_results.items():
        o = result["overall"]
        ablation_summary_rows.append({
            "Tiers": label,
            "Credits cleared": f"{o['credits_cleared']} / {o['total_credits']}",
            "Auto-clear rate": fmt_pct(o["auto_clear_rate"]),
            "Precision": fmt_pct(o["precision"]),
            "Recall": fmt_pct(o["recall"]),
            "F1": fmt_pct(o["f1"]),
        })

    st.dataframe(
        pd.DataFrame(ablation_summary_rows),
        hide_index=True,
    )
else:
    st.info("No ablation data available.", icon=":material/info:")

st.space("large")

# ---------------------------------------------------------------------------
# Precision / auto-clear trade-off curve
# ---------------------------------------------------------------------------

st.subheader(":material/show_chart: Precision / auto-clear trade-off")
st.caption(
    "Adjusting the LLM confidence gate lets the business trade off between "
    "automation rate and absolute precision."
)

curve_results = run_curve_cached(data_dir, max_tier)

curve_rows = []
for threshold, result in curve_results.items():
    o = result["overall"]
    curve_rows.append({
        "Confidence threshold": threshold,
        "Auto-clear rate": o["auto_clear_rate"] if o["auto_clear_rate"] is not None else 0,
        "Precision": o["precision"] if o["precision"] is not None else 0,
        "Credits cleared": o["credits_cleared"],
    })

curve_df = pd.DataFrame(curve_rows)

if not curve_df.empty:
    # Melt for dual-line chart
    melted = curve_df.melt(
        id_vars=["Confidence threshold", "Credits cleared"],
        value_vars=["Auto-clear rate", "Precision"],
        var_name="Metric",
        value_name="Value",
    )

    line_chart = (
        alt.Chart(melted)
        .mark_line(point=True, strokeWidth=3)
        .encode(
            x=alt.X(
                "Confidence threshold:Q",
                title="Minimum confidence threshold",
                scale=alt.Scale(domain=[0, 1]),
            ),
            y=alt.Y(
                "Value:Q",
                title="Rate",
                scale=alt.Scale(domain=[0, 1]),
            ),
            color=alt.Color(
                "Metric:N",
                scale=alt.Scale(
                    domain=["Auto-clear rate", "Precision"],
                    range=["#60A5FA", "#34D399"],
                ),
            ),
            tooltip=[
                alt.Tooltip("Confidence threshold:Q", format=".2f"),
                alt.Tooltip("Metric:N"),
                alt.Tooltip("Value:Q", format=".3f"),
            ],
        )
    )

    st.altair_chart(line_chart)
else:
    st.info("No curve data available.", icon=":material/info:")

st.space("large")

# ---------------------------------------------------------------------------
# Per-break-type recall bar chart
# ---------------------------------------------------------------------------

st.subheader(":material/bar_chart: Per-break-type recall")
st.caption(
    f"Recall for each break type at the current engine configuration "
    f"(T0..T{max_tier}, confidence ≥ {min_conf:.0%})."
)

per_break = metrics["per_break"]
break_rows = []
for bt, bm in per_break.items():
    break_rows.append({
        "Break type": bt,
        "Recall": bm["recall"],
        "Total": bm["total"],
        "Correct": bm["correct"],
    })

break_df = pd.DataFrame(break_rows)

if not break_df.empty:
    bar_chart = (
        alt.Chart(break_df)
        .mark_bar(cornerRadiusEnd=6, size=24)
        .encode(
            x=alt.X("Recall:Q", title="Recall", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("Break type:N", sort="-x", title=None),
            color=alt.Color(
                "Recall:Q",
                scale=alt.Scale(scheme="greens", domain=[0, 1]),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("Break type:N"),
                alt.Tooltip("Recall:Q", format=".3f"),
                alt.Tooltip("Correct:Q", title="Correct"),
                alt.Tooltip("Total:Q", title="Total"),
            ],
        )
    )

    text_labels = (
        alt.Chart(break_df)
        .mark_text(dx=20, color="#F1F5F9", fontWeight="bold")
        .encode(
            x="Recall:Q",
            y=alt.Y("Break type:N", sort="-x"),
            text=alt.Text("Recall:Q", format=".2f"),
        )
    )

    st.altair_chart(bar_chart + text_labels)
else:
    st.info("No per-break data available.", icon=":material/info:")
