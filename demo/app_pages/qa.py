"""Settlement Q&A — natural-language interface over the reconciliation output."""

import os

import streamlit as st

from shared import run_pipeline, run_cash_position, fmt_inr, inject_theme_css
from recon.qa_agent import answer_question, build_context, is_available

inject_theme_css()

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

# ---------------------------------------------------------------------------
# Conversation state
# ---------------------------------------------------------------------------

if "qa_history" not in st.session_state:
    st.session_state.qa_history = []  # list of {"role": "user"|"assistant", "content": str}
if "qa_pending" not in st.session_state:
    st.session_state.qa_pending = None  # a question queued from a suggestion chip


def _submit(q):
    q = (q or "").strip()
    if q:
        st.session_state.qa_pending = q


# ---------------------------------------------------------------------------
# Empty state — suggested questions as chips
# ---------------------------------------------------------------------------

if not st.session_state.qa_history and not st.session_state.qa_pending:
    if not is_available():
        st.info(
            "Set `GROQ_API_KEY` to enable natural-language queries. "
            "Everything else in the dashboard works without it.",
            icon=":material/key_off:",
        )

    st.markdown(
        '<div class="fc-chat-empty">'
        '<div class="fc-chat-empty-icon">&#8853;</div>'
        "<p>Ask about payouts, exceptions, cash at risk, or break types. "
        "Answers are grounded in the verified reconciliation output.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="fc-chip-label">Try one of these</div>', unsafe_allow_html=True)
    cols = st.columns(2)
    for i, q in enumerate(SUGGESTED_QUESTIONS):
        with cols[i % 2]:
            st.button(
                q,
                key=f"suggest_{i}",
                on_click=_submit,
                args=(q,),
                width="stretch",
                disabled=not is_available(),
            )

# ---------------------------------------------------------------------------
# Transcript
# ---------------------------------------------------------------------------

for msg in st.session_state.qa_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------------------------------------------------------------------
# Resolve a pending question (from a chip or the input box)
# ---------------------------------------------------------------------------

if st.session_state.qa_pending:
    q = st.session_state.qa_pending
    st.session_state.qa_pending = None
    st.session_state.qa_history.append({"role": "user", "content": q})

    with st.chat_message("user"):
        st.markdown(q)

    with st.chat_message("assistant"):
        with st.spinner("Reading the reconciliation output…"):
            answer = answer_question(q, result_df, bank_df, report_df, position)
        st.markdown(answer)

    st.session_state.qa_history.append({"role": "assistant", "content": answer})

    with st.expander("Sources used — the exact context the model saw"):
        st.caption(
            "Summarised aggregates and the top exceptions by value — not raw "
            "ledger rows. The model answers from figures the engine already verified."
        )
        st.code(
            build_context(result_df, bank_df, report_df, position),
            language="markdown",
        )

# ---------------------------------------------------------------------------
# Input — pinned to the bottom
# ---------------------------------------------------------------------------

prompt = st.chat_input(
    "Ask about payouts, exceptions, or cash at risk…"
    if is_available()
    else "Set GROQ_API_KEY to enable",
    disabled=not is_available(),
)
if prompt:
    _submit(prompt)
    st.rerun()

if st.session_state.qa_history:
    if st.button("Clear conversation", icon=":material/restart_alt:", type="tertiary"):
        st.session_state.qa_history = []
        st.session_state.qa_pending = None
        st.rerun()
