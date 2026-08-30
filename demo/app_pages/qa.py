"""Settlement Q&A — natural-language interface over the reconciliation output."""

import os

import streamlit as st

from shared import run_pipeline, run_cash_position, fmt_inr
from recon.qa_agent import answer_question, build_context, is_available

SUGGESTED_QUESTIONS = [
    "Why was yesterday's payout short?",
    "Which exceptions are most urgent?",
    "What is my expected settlement this week?",
    "Which break type is most common?",
]

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
position, report_df, run_date = run_cash_position(data_dir, max_tier, min_conf)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title(":material/chat: Ask the controller")
st.caption(
    "Natural-language questions answered from the structured reconciliation "
    "output — never from raw ledger rows."
)

if not is_available():
    st.info(
        "Set GROQ_API_KEY to enable natural language queries.",
        icon=":material/key_off:",
    )

st.space("small")

# ---------------------------------------------------------------------------
# Suggested questions
# ---------------------------------------------------------------------------

if "qa_question" not in st.session_state:
    st.session_state.qa_question = ""

st.markdown("**Suggested questions**")
cols = st.columns(2)
for i, q in enumerate(SUGGESTED_QUESTIONS):
    with cols[i % 2]:
        def _set_question(question=q):
            st.session_state.qa_question = question

        st.button(
            q,
            key=f"suggest_{i}",
            on_click=_set_question,
            width="stretch",
        )

st.space("small")

# ---------------------------------------------------------------------------
# Question input
# ---------------------------------------------------------------------------

question = st.text_input(
    "Your question",
    value=st.session_state.qa_question,
    placeholder="e.g. How much cash is at risk right now?",
    key="qa_input",
)

ask = st.button("Ask", type="primary", icon=":material/send:")

# ---------------------------------------------------------------------------
# Answer
# ---------------------------------------------------------------------------

if question and (ask or st.session_state.qa_question == question):
    if not is_available():
        st.warning(
            "Set GROQ_API_KEY to enable natural language queries.",
            icon=":material/key_off:",
        )
    else:
        with st.spinner("Thinking..."):
            answer = answer_question(
                question, result_df, bank_df, report_df, position
            )
        st.markdown("**Answer**")
        with st.container(border=True):
            st.markdown(answer)

    st.space("small")

    with st.expander("Sources used"):
        st.caption(
            "The exact structured context passed to the model — summarised "
            "aggregates and the top exceptions, not raw ledger rows."
        )
        context = build_context(result_df, bank_df, report_df, position)
        st.code(context, language="markdown")
