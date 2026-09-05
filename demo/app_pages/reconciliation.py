"""Live reconciliation — tier-by-tier execution with visual progression."""

import os
import time

import altair as alt
import pandas as pd
import streamlit as st

from shared import (
    run_pipeline,
    run_tier_by_tier,
    run_audit_trail,
    fmt_inr,
    fmt_pct,
    tier_label,
    tier_icon,
    tier_description,
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

st.title(":material/account_balance: Live reconciliation")
st.caption("Watch the engine clear credits tier by tier.")

st.space("small")

# ---------------------------------------------------------------------------
# Tier-by-tier execution status
# ---------------------------------------------------------------------------

st.subheader(":material/play_circle: Tier execution")

tier_data = run_tier_by_tier(data_dir, min_conf, max_tier)

# Display each tier as an expandable status block
prev_cleared = 0
for i, td in enumerate(tier_data):
    tier_num = i
    current_cleared = td["credits_cleared"]
    new_cleared = current_cleared - prev_cleared

    if tier_num > max_tier:
        break

    icon = "✅" if new_cleared > 0 else "⬜"
    label = f"{tier_label(tier_num)} — cleared {new_cleared} new credit{'s' if new_cleared != 1 else ''}"

    with st.expander(f"{icon}  {label}", expanded=(new_cleared > 0)):
        cols = st.columns(4)
        with cols[0]:
            st.metric("Credits cleared (cumulative)", current_cleared, border=True)
        with cols[1]:
            st.metric("New this tier", new_cleared, border=True)
        with cols[2]:
            st.metric(
                "Auto-clear rate",
                fmt_pct(td["auto_clear_rate"]),
                border=True,
            )
        with cols[3]:
            prec = td["precision"]
            st.metric(
                "Precision",
                fmt_pct(prec),
                border=True,
            )
        st.caption(tier_description(tier_num))

    prev_cleared = current_cleared

st.space("medium")

# ---------------------------------------------------------------------------
# Waterfall chart — cumulative tier clearance
# ---------------------------------------------------------------------------

st.subheader(":material/waterfall_chart: Tier clearance waterfall")
st.caption("Cumulative credits cleared by each tier.")

# Build incremental data
waterfall_rows = []
prev = 0
for td in tier_data:
    tier_num = int(td["tier"][1])
    if tier_num > max_tier:
        break
    incremental = td["credits_cleared"] - prev
    waterfall_rows.append({
        "Tier": td["tier"],
        "Incremental": incremental,
        "Cumulative": td["credits_cleared"],
        "Label": tier_label(tier_num),
    })
    prev = td["credits_cleared"]

wf_df = pd.DataFrame(waterfall_rows)

if not wf_df.empty and wf_df["Incremental"].sum() > 0:
    bars = (
        alt.Chart(wf_df)
        .mark_bar(cornerRadiusEnd=6, size=32)
        .encode(
            x=alt.X("Incremental:Q", title="Credits cleared"),
            y=alt.Y("Tier:N", sort=list(wf_df["Tier"]), title=None),
            color=alt.Color(
                "Tier:N",
                scale=alt.Scale(
                    domain=list(wf_df["Tier"]),
                    range=["#8a8a8a", "#a3a3a3", "#c7c7c7", "#d4d4d4", "#f5f5f5"],
                ),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("Label:N", title="Tier"),
                alt.Tooltip("Incremental:Q", title="New credits"),
                alt.Tooltip("Cumulative:Q", title="Cumulative"),
            ],
        )
    )

    text = (
        alt.Chart(wf_df)
        .mark_text(dx=15, color="#F5F5F5", fontWeight="bold")
        .encode(
            x="Incremental:Q",
            y=alt.Y("Tier:N", sort=list(wf_df["Tier"])),
            text=alt.Text("Incremental:Q", format="d"),
        )
    )

    st.altair_chart(bars + text)
else:
    st.info("No credits cleared at the selected tier level.", icon=":material/info:")

st.space("medium")

# ---------------------------------------------------------------------------
# Full results table
# ---------------------------------------------------------------------------

st.subheader(":material/table_chart: Assignment results")
st.caption("Every settlement and its reconciliation assignment.")

result_df, bank_df, cleared, metrics, truth = run_pipeline(
    data_dir, max_tier, min_conf
)

audit_df = run_audit_trail(data_dir, max_tier, min_conf)
st.download_button(
    "Export audit trail",
    data=audit_df.to_csv(index=False),
    file_name="audit_trail.csv",
    mime="text/csv",
    icon=":material/download:",
    help="Full per-entry paper trail: assignment, tier, reason code, confidence, and evidence.",
)

# Show the settlement assignments
display_cols = [
    "payment_id", "gross", "fee", "gst", "net", "settled_at",
    "assigned_utr", "assigned_tier", "assigned_confidence",
]
available_cols = [c for c in display_cols if c in result_df.columns]
display_df = result_df[available_cols].copy()

st.dataframe(
    display_df,
    column_config={
        "payment_id": st.column_config.TextColumn("Payment ID"),
        "gross": st.column_config.NumberColumn("Gross", format="₹%.2f"),
        "fee": st.column_config.NumberColumn("Fee", format="₹%.2f"),
        "gst": st.column_config.NumberColumn("GST", format="₹%.2f"),
        "net": st.column_config.NumberColumn("Net", format="₹%.2f"),
        "settled_at": st.column_config.DatetimeColumn("Settled", format="DD/MM/YY"),
        "assigned_utr": st.column_config.TextColumn("Assigned UTR"),
        "assigned_tier": st.column_config.NumberColumn("Tier", format="%d"),
        "assigned_confidence": st.column_config.ProgressColumn(
            "Confidence",
            min_value=0,
            max_value=1,
            format="%.2f",
        ),
    },
    hide_index=True,
    height=420,
)

st.space("medium")

# ---------------------------------------------------------------------------
# Credit ledger
# ---------------------------------------------------------------------------

st.subheader(":material/receipt_long: Credit ledger")
st.caption("Every bank credit and its clearance status.")

credit_rows = []
for _, credit in bank_df.iterrows():
    utr = credit["utr"]
    is_cleared = utr in cleared
    member_count = len(cleared[utr]) if is_cleared else 0

    # Find tier that cleared it
    clearing_tier = None
    if is_cleared:
        member_indices = cleared[utr]
        tiers = result_df.loc[member_indices, "assigned_tier"].dropna().unique()
        if len(tiers) > 0:
            clearing_tier = int(max(tiers))

    credit_rows.append({
        "UTR": utr,
        "Amount": credit["credit"],
        "Value date": credit["value_date"],
        "Status": "Cleared" if is_cleared else "Pending",
        "Cleared by tier": f"T{clearing_tier}" if clearing_tier is not None else "—",
        "Member settlements": member_count,
    })

credit_df = pd.DataFrame(credit_rows)

st.dataframe(
    credit_df,
    column_config={
        "UTR": st.column_config.TextColumn("UTR"),
        "Amount": st.column_config.NumberColumn("Amount", format="₹%.2f"),
        "Value date": st.column_config.TextColumn("Value date"),
        "Status": st.column_config.TextColumn("Status"),
        "Cleared by tier": st.column_config.TextColumn("Cleared by"),
        "Member settlements": st.column_config.NumberColumn("Members", format="%d"),
    },
    hide_index=True,
    height=350,
)
