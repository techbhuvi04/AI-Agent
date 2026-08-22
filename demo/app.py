import os
import sys
import streamlit as st

# Ensure we can import from the parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

st.set_page_config(
    page_title="AI Finance Controller",
    page_icon=":material/account_balance:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Shared session state
# ---------------------------------------------------------------------------

if "investigated_utrs" not in st.session_state:
    st.session_state.investigated_utrs = set()

# ---------------------------------------------------------------------------
# Sidebar — engine controls (shared across all pages)
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header(":material/tune: Engine settings")

    data_dir = st.text_input(
        "Data directory",
        value="data",
        key="data_dir",
    )

    st.subheader("Tier control")
    max_tier = st.slider(
        "Max execution tier",
        min_value=0,
        max_value=4,
        value=4,
        help="0: Key enrichment only\n1: Date arithmetic\n2: Subset-sum DP\n3: LLM Agent\n4: Arithmetic Verifier",
        key="max_tier",
    )

    st.subheader("Confidence gate")
    min_conf = st.slider(
        "Minimum confidence",
        min_value=0.0,
        max_value=1.0,
        value=0.90,
        step=0.05,
        help="LLM assignments below this confidence are routed to the exception queue.",
        key="min_conf",
    )

    st.space("large")
    st.caption("Razorpay AI Buildathon submission")

# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

page = st.navigation(
    [
        st.Page("app_pages/home.py", title="Overview", icon=":material/home:"),
        st.Page(
            "app_pages/reconciliation.py",
            title="Reconciliation",
            icon=":material/account_balance:",
        ),
        st.Page(
            "app_pages/analytics.py",
            title="Analytics",
            icon=":material/analytics:",
        ),
        st.Page(
            "app_pages/exceptions.py",
            title="Exception queue",
            icon=":material/warning:",
        ),
    ],
    position="top",
)

page.run()
