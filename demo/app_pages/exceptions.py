"""Exception queue — redesigned investigation workspace."""

import os

import pandas as pd
import streamlit as st

from shared import run_pipeline, fmt_inr

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

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

cleared_utrs = set(cleared.keys())
investigated = st.session_state.get("investigated_utrs", set())
uncleared_bank = bank_df[
    ~bank_df["utr"].isin(cleared_utrs | investigated)
].sort_values("value_date")
pending_count = len(uncleared_bank)

st.title(":material/warning: Exception queue")

with st.container(horizontal=True):
    st.caption(
        "Credits that failed automatic reconciliation and require manual "
        "controller review."
    )
    if pending_count > 0:
        st.badge(f"{pending_count} pending", icon=":material/schedule:", color="orange")
    else:
        st.badge("Inbox zero", icon=":material/check:", color="green")

st.space("small")

# ---------------------------------------------------------------------------
# Manual investigation counter
# ---------------------------------------------------------------------------

manual_cleared_count = len(
    [u for u in investigated if u not in cleared_utrs]
)

if manual_cleared_count > 0:
    st.info(
        f"You have manually investigated and dismissed "
        f"**{manual_cleared_count}** exception{'s' if manual_cleared_count != 1 else ''} "
        f"during this session.",
        icon=":material/task_alt:",
    )

# ---------------------------------------------------------------------------
# Inbox zero celebration
# ---------------------------------------------------------------------------

if pending_count == 0:
    st.space("large")
    with st.container(horizontal_alignment="center"):
        st.success(
            "**Inbox zero!** All bank credits have been successfully reconciled "
            "or manually investigated.",
            icon=":material/celebration:",
        )
        st.caption("Adjust the engine settings in the sidebar to see the effect on clearance.")
    st.stop()

# ---------------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------------

overall = metrics["overall"]
total = overall["total_credits"]
auto_cleared = overall["credits_cleared"]

with st.container(horizontal=True):
    st.metric(
        "Total credits",
        total,
        border=True,
    )
    st.metric(
        "Auto-cleared",
        auto_cleared,
        f"{auto_cleared}/{total}",
        border=True,
    )
    st.metric(
        "Manually dismissed",
        manual_cleared_count,
        border=True,
    )
    st.metric(
        "Remaining",
        pending_count,
        f"{pending_count} require review",
        delta_color="inverse",
        border=True,
    )

st.space("medium")

# ---------------------------------------------------------------------------
# Exception cards
# ---------------------------------------------------------------------------

unassigned_settlements = result_df[result_df["assigned_utr"].isna()]

for idx, credit in uncleared_bank.iterrows():
    utr = credit["utr"]
    amount = credit["credit"]
    value_date = credit["value_date"]

    with st.container(border=True):
        # Header row
        with st.container(horizontal=True):
            st.markdown(f":material/error_outline: **UTR:** `{utr}`")
            st.badge("Unresolved", icon=":material/warning:", color="red")

        col_left, col_right = st.columns([1, 2])

        with col_left:
            st.markdown("**Target credit**")
            st.metric("Amount", fmt_inr(amount), border=True)
            st.caption(f"Value date: {value_date}")

            st.space("small")

            # Mark as investigated callback
            def mark_investigated(u=utr):
                st.session_state.investigated_utrs.add(u)

            st.button(
                "Mark as investigated",
                key=f"btn_{utr}",
                type="primary",
                icon=":material/check_circle:",
                on_click=mark_investigated,
            )

        with col_right:
            st.markdown("**Suspected candidate settlements**")

            c_date = pd.to_datetime(value_date).date()
            start_date = c_date - pd.Timedelta(days=3)
            end_date = c_date + pd.Timedelta(days=3)

            s_dates = pd.to_datetime(unassigned_settlements["settled_at"]).dt.date
            nearby = unassigned_settlements[
                s_dates.between(start_date, end_date)
            ].copy()

            if len(nearby) > 0:
                nearby_sum = nearby["net"].sum()
                delta = nearby_sum - amount
                delta_sign = "+" if delta >= 0 else ""

                st.caption(
                    f"Found **{len(nearby)}** nearby unassigned settlements "
                    f"summing to **{fmt_inr(nearby_sum)}** "
                    f"(Δ {delta_sign}{fmt_inr(delta)})"
                )

                st.dataframe(
                    nearby[["payment_id", "gross", "fee", "gst", "net", "settled_at"]],
                    column_config={
                        "payment_id": st.column_config.TextColumn("Payment ID"),
                        "gross": st.column_config.NumberColumn("Gross", format="₹%.2f"),
                        "fee": st.column_config.NumberColumn("Fee", format="₹%.2f"),
                        "gst": st.column_config.NumberColumn("GST", format="₹%.2f"),
                        "net": st.column_config.NumberColumn("Net", format="₹%.2f"),
                        "settled_at": st.column_config.DatetimeColumn(
                            "Settled", format="DD/MM/YY"
                        ),
                    },
                    hide_index=True,
                )
            else:
                st.warning(
                    "No nearby unassigned settlements found within ±3 days.",
                    icon=":material/search_off:",
                )
