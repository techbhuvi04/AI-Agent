# Architecture & Design Decisions

This document outlines the architectural tradeoffs made in building the Ledger Reconciliation Engine. Because we are dealing with financial ledgers, the system was designed around two non-negotiable principles:
1. **Never fabricate a match.** (High Precision is prioritized over High Recall).
2. **Deterministic first, Probabilistic second.** (Use traditional algorithms where they work, use LLMs where structural reasoning is needed, and never trust the LLM's math).

## The Multi-Tiered Pipeline

The engine executes in progressive tiers. If a credit is solved by an earlier tier, it is removed from the candidate pool, shrinking the search space for the next tier.

### T0: Key-Based Enrichment (Deterministic)
The engine first joins the `orders.csv` and `settlements.csv` using `payment_id`. 
* **Design Decision:** The system expects missing data. If an order is missing from the ledger, it is still retained in the candidate pool for reconciliation, because the bank still credited us for it. 

### T1: Date-Window Arithmetic (Deterministic)
The engine looks for perfect date-boundary batches. If the sum of all unassigned payments on a specific date matches the bank credit exactly, it clears the batch.
* **Failure Mode:** In reality, batches rarely align perfectly with T+2 dates due to late captures. T1 usually fails to clear the majority of credits, but it successfully *narrows the search space* for T2.
* **Design Decision:** Integer paise math. Floating point drift (₹100.00 - ₹2.50 = ₹97.499999) destroys exact matching. The engine converts all floating-point nets into integers (`int(round(v * 100))`) before any arithmetic comparison.

### T2: Constrained Subset-Sum (Deterministic)
To find the exact subset of payments that form a batch across bleeding date boundaries, the engine uses a Dynamic Programming (DP) subset-sum algorithm.
* **Design Decision (Exclusion DP over Inclusion DP):** Standard subset-sum tries to find elements that sum to the target credit (e.g., finding 120 elements out of 150 that sum to ₹10,000). The DP table for this is massive (`O(N * Target)`). Instead, we calculate the `excess` (Sum of Candidates - Target Credit). The excess is usually small (₹50–₹500). We run subset-sum to find the few elements to *exclude* to reach the excess. This reduces the iteration space by orders of magnitude (from ~1.7M iterations to ~10K iterations).
* **Design Decision (Progressive Expansion):** The engine tries to find a subset within the strict date window first. If it fails, it expands the window by 3 days, then 6 days, protecting the candidate pool from unnecessary noise.

### T3: LLM Agent (Probabilistic)
Some credits cannot be solved by subset-sum (e.g. `netting_split` where a batch is split across two bank credits, or `duplicate_utr`). T3 passes the remaining unassigned candidates to an LLM (Gemini 2.0 Flash) with a strictly typed JSON schema.
* **Design Decision (Row-Level Precision):** The LLM is provided the exact `entry_id` (DataFrame index) of each candidate. It does not respond with arbitrary amounts; it responds with an array of `proposed_entry_ids`.
* **Design Decision (Confidence Gate):** The LLM must supply a confidence score. This allows the business to reject low-confidence LLM claims and route them to the manual exception queue.

### T4: The Arithmetic Verifier (Deterministic Gate)
The most important tier in the engine. **LLMs cannot do math reliably.**
* **Design Decision (Trust but Verify):** T3 proposes a claim, but it cannot apply it. T4 intercepts the claim and sums the nets of the `proposed_entry_ids`. If the sum does not equal the target credit (within a ±₹0.50 rounding tolerance), or if the LLM hallucinated an already-assigned `entry_id`, T4 rejects the claim and routes the credit to the Exception Queue.

## Evaluation Harness

Because the engine is stochastic at T3, it requires a formal evaluation harness. The custom harness (`eval/harness.py`) generates synthetic ledgers with injected breaks (late settlements, refunds, chargebacks) and evaluates the engine's Precision, Recall, and Auto-Clear Rate at every tier.
