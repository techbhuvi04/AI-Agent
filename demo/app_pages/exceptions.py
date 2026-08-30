"""Exception queue — structured investigation workspace backed by
build_exception_report()."""

import json
import os

import pandas as pd
import streamlit as st

from shared import run_pipeline, run_exception_report, fmt_inr, inject_theme_css

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

result_df, bank_df, cleared, metrics, truth = run_pipeline(
    data_dir, max_tier, min_conf
)
report_df = run_exception_report(data_dir, max_tier, min_conf)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

cleared_utrs = set(cleared.keys())
investigated = st.session_state.get("investigated_utrs", set())
if len(report_df) and investigated:
    report_df = report_df[~report_df["credit_utr"].isin(investigated)]
pending_count = len(report_df)

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
# Summary banner
# ---------------------------------------------------------------------------

total_at_risk = float(report_df["credit_amount"].sum())
oldest_age = int(report_df["age_days"].max())

with st.container(horizontal=True):
    st.metric("Total unresolved", pending_count, border=True)
    st.metric("₹ at risk", fmt_inr(total_at_risk), border=True)
    st.metric("Oldest exception", f"{oldest_age} days", border=True)

st.space("medium")

# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

filter_cols = st.columns(2)
with filter_cols[0]:
    materiality_options = ["HIGH", "MEDIUM", "LOW"]
    selected_materiality = st.multiselect(
        "Materiality", materiality_options, default=materiality_options,
    )
with filter_cols[1]:
    break_code_options = sorted(report_df["break_code"].unique().tolist())
    selected_break_codes = st.multiselect(
        "Break code", break_code_options, default=break_code_options,
    )

filtered = report_df[
    report_df["materiality"].isin(selected_materiality)
    & report_df["break_code"].isin(selected_break_codes)
].sort_values("credit_amount", ascending=False)

st.caption(f"Showing {len(filtered)} of {pending_count} exceptions.")

# ---------------------------------------------------------------------------
# Export to CSV
# ---------------------------------------------------------------------------

st.download_button(
    "Export to CSV",
    data=report_df.to_csv(index=False),
    file_name="exception_report.csv",
    mime="text/csv",
    icon=":material/download:",
)

st.space("medium")

# ---------------------------------------------------------------------------
# Exception cards
# ---------------------------------------------------------------------------

BREAK_CODE_COLORS = {
    "SUM_COLLISION": "red",
    "WINDOW_DEFICIT": "orange",
    "NO_CANDIDATES": "orange",
    "UNRESOLVED": "gray",
}
BREAK_CODE_ACCENT_HEX = {
    "SUM_COLLISION": "#FFFFFF",
    "WINDOW_DEFICIT": "#C7C7C7",
    "NO_CANDIDATES": "#C7C7C7",
    "UNRESOLVED": "#737373",
}

# The `duplicate_utr` break means two bank rows can legitimately carry
# the same UTR, so widget keys are suffixed with the row position — a
# bare UTR key collides and Streamlit refuses to render the page.
for row_pos, (_, exc) in enumerate(filtered.iterrows()):
    utr = exc["credit_utr"]
    amount = exc["credit_amount"]
    value_date = exc["value_date"]
    break_code = exc["break_code"]
    color = BREAK_CODE_COLORS.get(break_code, "gray")
    accent = BREAK_CODE_ACCENT_HEX.get(break_code, "#737373")

    header = (
        f":material/error_outline: **UTR:** `{utr}` — {fmt_inr(amount)}  "
        f"·  {exc['age_days']}d old"
    )

    st.html(f'<div style="height:3px;border-radius:3px 3px 0 0;background:{accent};margin-bottom:-1px;"></div>')
    with st.expander(header, expanded=False):
        with st.container(horizontal=True):
            st.badge(break_code, color=color)
            st.badge(exc["materiality"], color="blue" if exc["materiality"] == "HIGH" else "gray")

        col_left, col_right = st.columns([1, 2])

        with col_left:
            st.markdown("**Target credit**")
            st.metric("Amount", fmt_inr(amount), border=True)
            st.caption(f"Value date: {value_date}")
            st.caption(f"Δ (delta to closest match): {fmt_inr(exc['delta_inr'])}")

            if exc["hypothesis"]:
                st.markdown("**T3 hypothesis**")
                st.caption(exc["hypothesis"])

            st.markdown(f"**Suggested action:** {exc['suggested_action']}")

            st.space("small")

            def mark_investigated(u=utr):
                st.session_state.investigated_utrs.add(u)

            st.button(
                "Mark as investigated",
                key=f"btn_{row_pos}_{utr}",
                type="primary",
                icon=":material/check_circle:",
                on_click=mark_investigated,
            )

        with col_right:
            st.markdown("**Evidence — closest candidate subset found**")
            try:
                evidence = json.loads(exc["evidence"])
            except (TypeError, ValueError):
                evidence = {}

            closest_ids = evidence.get("closest_entry_ids", [])
            st.caption(
                f"{evidence.get('num_candidates', 0)} candidates in window · "
                f"closest subset sums to {fmt_inr(evidence.get('closest_sum', 0))}"
            )

            if closest_ids:
                evidence_rows = result_df.loc[
                    [i for i in closest_ids if i in result_df.index],
                    ["payment_id", "gross", "fee", "gst", "net", "settled_at"],
                ]
                st.dataframe(
                    evidence_rows,
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
                    "No nearby unassigned settlements found.",
                    icon=":material/search_off:",
                )
