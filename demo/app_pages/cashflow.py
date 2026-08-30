"""Cash position — 7-day forward inflow forecast with a confidence band."""

import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from shared import run_cash_position, fmt_inr

# Theme-matched colors (see .streamlit/config.toml) — strictly grayscale
INK = "#F5F5F5"
MUTED = "#A3A3A3"
GRID = "#3A3A3A"
SURFACE = "#000000"
ACCENT = "#FFFFFF"
BAND = "#8A8A8A"
RISK = "#D4D4D4"

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

st.title(":material/payments: Cash position")
st.caption(
    "Forward cash position projected from verified reconciliation output."
)

# The forecast is an "as of" view: it projects settlements that have not
# yet been credited. Defaulting to the ledger's last credit date is the
# honest reading of "today", but at the very end of the ledger the
# pipeline is legitimately empty — so the controller can move the date.
_default_position, _report, default_run_date = run_cash_position(
    data_dir, max_tier, min_conf
)

run_date = st.date_input(
    "As-of date",
    value=default_run_date,
    help=(
        "Projects unassigned settlements forward by the settlement cycle. "
        "At the end of the ledger there may be no open pipeline left, in "
        "which case the forecast is legitimately zero."
    ),
)

position, report_df, run_date = run_cash_position(
    data_dir, max_tier, min_conf, run_date=run_date
)

st.space("small")

# ---------------------------------------------------------------------------
# Headline
# ---------------------------------------------------------------------------

total = position["expected_inflow_total"]
low, high = position["confidence_interval"]


def _lakh(v):
    return f"₹{v / 100000:.2f}L"


st.subheader(
    f"Expected inflow next 7 days: {_lakh(total)} "
    f"(range {_lakh(low)}–{_lakh(high)})"
)

if total == 0:
    st.info(
        f"No settlements are pending credit in the 7 days from {run_date}. "
        "Move the as-of date earlier to see the forecast while the pipeline "
        "is still open.",
        icon=":material/info:",
    )

st.space("small")

# ---------------------------------------------------------------------------
# KPI tiles
# ---------------------------------------------------------------------------

with st.container(horizontal=True):
    st.metric("Verified settled", fmt_inr(position["verified_settled"]), border=True)
    st.metric(
        "At risk",
        fmt_inr(position["at_risk"]),
        delta_color="inverse",
        border=True,
    )
    st.metric("7-day forecast", fmt_inr(total), border=True)

st.space("medium")

# ---------------------------------------------------------------------------
# Daily inflow bar chart with confidence band
# ---------------------------------------------------------------------------

st.subheader(":material/bar_chart: Daily expected inflow")
st.caption(
    "Unassigned payments projected to credit at settled_at + settlement cycle. "
    "The shaded band shows the upside if every at-risk credit is recovered."
)

inflow = position["expected_inflow"]
dates = list(inflow.keys())
values = [inflow[d] for d in dates]

# The at-risk upside spread evenly across the forecast horizon — the band
# represents recovery potential, not a per-day prediction.
at_risk = position["at_risk"]
band_per_day = at_risk / len(dates) if dates else 0
upper = [v + band_per_day for v in values]

fig = go.Figure()

fig.add_trace(
    go.Bar(
        x=dates,
        y=values,
        name="Expected inflow",
        marker=dict(color=ACCENT, line=dict(width=0)),
        hovertemplate="<b>%{x}</b><br>Expected: ₹%{y:,.2f}<extra></extra>",
    )
)

fig.add_trace(
    go.Scatter(
        x=dates,
        y=upper,
        name="Upside (at-risk recovered)",
        mode="lines",
        line=dict(color=BAND, width=2, dash="dot"),
        hovertemplate="<b>%{x}</b><br>Upside: ₹%{y:,.2f}<extra></extra>",
    )
)

fig.update_layout(
    paper_bgcolor=SURFACE,
    plot_bgcolor=SURFACE,
    font=dict(color=INK, size=13),
    height=380,
    margin=dict(l=10, r=10, t=30, b=10),
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(color=MUTED)),
    bargap=0.35,
)
fig.update_xaxes(
    showgrid=False,
    linecolor=GRID,
    tickfont=dict(color=MUTED),
    title=None,
)
fig.update_yaxes(
    gridcolor=GRID,
    zerolinecolor=GRID,
    tickfont=dict(color=MUTED),
    tickprefix="₹",
    title=None,
)

st.plotly_chart(fig, width="stretch")

st.space("medium")

# ---------------------------------------------------------------------------
# Cash at risk by age
# ---------------------------------------------------------------------------

st.subheader(":material/hourglass_bottom: Cash at risk by age")

by_age = position["cash_at_risk_by_age"]
with st.container(horizontal=True):
    for bucket in ["<3d", "3-7d", ">7d"]:
        st.metric(bucket, fmt_inr(by_age.get(bucket, 0.0)), border=True)

st.space("small")

# ---------------------------------------------------------------------------
# Aged exception callout
# ---------------------------------------------------------------------------

if len(report_df):
    aged = report_df[report_df["age_days"] > 3]
    aged_count = len(aged)
    aged_value = float(aged["credit_amount"].sum())
    if aged_count > 0:
        st.warning(
            f"**{aged_count}** exception{'s' if aged_count != 1 else ''} aged "
            f">3 days represent **{fmt_inr(aged_value)}** at risk.",
            icon=":material/priority_high:",
        )
    else:
        st.success("No exceptions aged beyond 3 days.", icon=":material/check_circle:")
else:
    st.success("No open exceptions — nothing at risk.", icon=":material/check_circle:")
