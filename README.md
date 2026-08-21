# AI Finance Controller — Ledger Reconciliation Engine

*A Razorpay AI Buildathon Submission*

This repository contains the **Ledger: Settlement Reconciliation Engine**, a tiered hybrid system designed to reconcile netted bank credits against individual payment ledger entries without shared keys.

## The Problem

A merchant receives netted bank credits from a payment gateway. One credit covers 80–140 individual payments, minus a per-payment fee and 18% GST on that fee, landing at T+2. Refunds and chargebacks appear as negative adjustments inside later credits. 

Reconciliation means proving mathematically that each bank credit equals a specific set of payments minus fees, and flagging what doesn't reconcile.

## Quickstart (For Judges)

1. **Setup Environment**
   ```bash
   pip install -r requirements.txt
   export GEMINI_API_KEY="your_api_key_here" # Required for T3 LLM Agent
   ```

2. **Run the Interactive Demo**
   ```bash
   make demo
   ```
   This opens the **Streamlit Exception Queue Dashboard** on `http://localhost:8501`. 

3. **Run the Quantitative Evaluation (Ablation Study)**
   ```bash
   make eval
   ```
   This generates synthetic ledger data with injected edge cases (late settlements, refunds, chargebacks, missing orders) and runs the multi-tier reconciliation engine against it, outputting a tiered ablation table proving the recall improvement of each tier.

4. **Run the Precision/Recall Confidence Curve**
   ```bash
   make curve
   ```
   This shows how adjusting the LLM Confidence Gate allows a business to trade off between Auto-Clear Rate and Absolute Precision.

## Architecture

The engine uses a tiered "Deterministic First, Probabilistic Second" control loop. 

Please see [ARCHITECTURE.md](./ARCHITECTURE.md) for a deep dive into the design decisions, including the subset-sum Dynamic Programming implementation, the Integer Paise math standard, and the LLM Verification Gate.

## Testing

The engine is heavily unit-tested (66/66 passing). Run the suite via:
```bash
make test
```
