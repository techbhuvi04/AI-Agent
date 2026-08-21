import os
import sys
import pandas as pd
import streamlit as st

# Ensure we can import from the parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from recon.engine import load_data, reconcile
from eval.metrics import compute_metrics
from eval.harness import load_ground_truth

st.set_page_config(
    page_title="AI Finance Controller | Exception Queue",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply some custom CSS for a premium feel
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border: 1px solid #e0e0e0;
    }
    .exception-card {
        background-color: #fff5f5;
        border-left: 4px solid #ff4b4b;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def run_pipeline(data_dir, max_tier, min_confidence):
    orders, settlements, bank = load_data(data_dir)
    truth = load_ground_truth(data_dir)
    
    # Run the engine
    result_df, cleared = reconcile(orders, settlements, bank, max_tier=max_tier, min_confidence=min_confidence)
    
    # Compute metrics for the dashboard
    metrics = compute_metrics(result_df, truth, cleared, bank)
    
    return result_df, bank, cleared, metrics


def main():
    st.title("🏦 AI Finance Controller")
    st.subheader("Automated Reconciliation & Exception Queue")
    
    # Sidebar Controls
    st.sidebar.header("Engine Settings")
    data_dir = st.sidebar.text_input("Data Directory", value="data")
    max_tier = st.sidebar.slider("Reconciliation Tiers (Max)", min_value=1, max_value=4, value=4, 
                                 help="1: Arithmetic\n2: Subset-Sum\n3: LLM Agent\n4: Arithmetic Verifier")
    min_conf = st.sidebar.slider("Confidence Gate (Min)", min_value=0.0, max_value=1.0, value=0.90, step=0.05,
                                 help="Assignments below this confidence are rejected to the exception queue.")
    
    if not os.path.exists(data_dir):
        st.error(f"Data directory '{data_dir}' not found. Please run the generator first.")
        return
        
    with st.spinner("Running Reconciliation Engine..."):
        try:
            result_df, bank_df, cleared, metrics = run_pipeline(data_dir, max_tier, min_conf)
        except Exception as e:
            st.error(f"Engine failed: {e}")
            return
            
    # Top Level Metrics
    overall = metrics["overall"]
    
    cols = st.columns(4)
    with cols[0]:
        st.metric("Total Credits", overall["total_credits"])
    with cols[1]:
        st.metric("Auto-Cleared", overall["credits_cleared"], 
                  f"{(overall['credits_cleared'] / overall['total_credits'] * 100):.1f}% rate")
    with cols[2]:
        precision = overall["precision"]
        p_str = f"{(precision * 100):.1f}%" if precision is not None else "N/A"
        st.metric("Precision", p_str)
    with cols[3]:
        st.metric("Exceptions", overall["total_credits"] - overall["credits_cleared"])
        
    st.markdown("---")
    
    # Exception Queue
    st.header("🚨 Exception Queue")
    st.markdown("The following bank credits could not be auto-cleared. They require manual review.")
    
    # Find uncleared credits
    cleared_utrs = set(cleared.keys())
    uncleared_bank = bank_df[~bank_df["utr"].isin(cleared_utrs)].sort_values("value_date")
    
    if len(uncleared_bank) == 0:
        st.success("🎉 Inbox Zero! All bank credits were successfully reconciled.")
    else:
        # Display unassigned candidates
        unassigned_settlements = result_df[result_df["assigned_utr"].isna()]
        
        for _, credit in uncleared_bank.iterrows():
            with st.expander(f"⚠️ UTR: {credit['utr']} | Amount: ₹{credit['credit']:.2f} | Date: {credit['value_date']}"):
                st.markdown(f"**Target Amount:** ₹{credit['credit']:.2f}")
                
                # Show nearby unassigned candidates
                c_date = pd.to_datetime(credit["value_date"]).date()
                start_date = c_date - pd.Timedelta(days=3)
                end_date = c_date + pd.Timedelta(days=3)
                
                # Filter candidates roughly around this date
                s_dates = pd.to_datetime(unassigned_settlements["settled_at"]).dt.date
                nearby_mask = s_dates.between(start_date, end_date)
                nearby = unassigned_settlements[nearby_mask].copy()
                
                if len(nearby) > 0:
                    st.markdown(f"**{len(nearby)} nearby unassigned candidates:** (Sum: ₹{nearby['net'].sum():.2f})")
                    # Display just the relevant columns
                    display_cols = ["payment_id", "gross", "fee", "gst", "net", "settled_at", "assigned_confidence"]
                    st.dataframe(nearby[display_cols], width="stretch")
                else:
                    st.warning("No nearby unassigned settlements found.")
                    
                st.button("Mark as Investigated", key=f"btn_{credit['utr']}")


if __name__ == "__main__":
    main()
