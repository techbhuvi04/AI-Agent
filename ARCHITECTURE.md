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

### T3: Break Classification Agent (Probabilistic)
Some credits cannot be solved by subset-sum (e.g. `netting_split` where a batch is split across two bank credits, or `duplicate_utr`).

* **Design Decision (Classify the break, don't propose the subset):** T3 originally asked the LLM to return `proposed_entry_ids` — i.e. to *do subset-sum*. That is precisely the task LLMs are worst at and the task T2's DP already solves exactly. The LLM's real comparative advantage is **structural**: recognising the *shape* of a break. So T3 now sends **diagnostics, not rows** — `window_deficit`, `excess_paise`, `num_candidates`, `nearby_credits`, `negative_entries_present` — and asks for two things:
  1. a `break_classification` (`late_settlement`, `netting_split`, `refund_batch`, `duplicate_utr`, `rounding_drift`, `missing_order`, `unknown`), and
  2. a **structured hypothesis** — an `action` (`expand_window`, `merge_with_utr`, `accept_with_tolerance`, `manual_review`) plus its parameters.
* **Design Decision (Deterministic resolver):** The hypothesis is never applied as stated. `resolve_hypothesis()` *tests* it with real arithmetic — `expand_window` re-runs the T2 DP over the widened window; `merge_with_utr` solves for the combined target of both credits and then re-splits; `accept_with_tolerance` is admissible only for `rounding_drift`; `manual_review` routes straight to the exception queue. The LLM narrows the search; the DP decides.
* **Design Decision (Heuristic fallback):** With `GROQ_API_KEY` unset, T3 falls back to a deterministic refund-aware wide-window matcher, so the pipeline runs end to end without any LLM.
* **Design Decision (Confidence Gate):** The LLM must supply a confidence score, letting the business reject low-confidence claims.

### T4: The Arithmetic Verifier + Uniqueness Gate (Deterministic Gate)
The most important tier in the engine. **LLMs cannot do math reliably — and sum-equality is not proof.**

* **Design Decision (Trust but Verify):** T3 proposes a claim but cannot apply it. T4 sums the nets of the `proposed_entry_ids` and rejects any claim that misses the target (±₹0.50) or references an already-assigned `entry_id`.

* **Design Decision (The Uniqueness Gate — why sum-equality alone is insufficient):** This is the core correctness argument of the project. T2 finds *a* subset summing to the target credit, and T4 originally checked only that `sum == target` — which is **tautologically true for any subset the solver returns**. That check could never fail, so it provided no evidence at all.

  The real question is not *"does this subset sum to the credit?"* but ***"is this the only subset that does?"*** With 80–160 candidate payments in a window, a given target is typically reachable by many different subsets. Picking one and clearing it is a coin flip dressed up as arithmetic.

  So T4 re-derives the candidate pool for the credit and counts distinct solutions with `_count_solutions()` — the same exclusion-DP shape as T2, but enumerating every path into `dp[target]` instead of returning the first, capped once the count exceeds what matters:

  | Solution count | Verdict | Reason code | Confidence |
  |---|---|---|---|
  | `1` | Accept — the match is *proven* | `UNIQUE_MATCH` | `1.0` |
  | `2 – 10` | Accept only the **intersection** of all valid solutions (entries in *every* subset are proven regardless of which is true); flag the rest as ambiguous | `AMBIGUOUS_PARTIAL` | `0.75` |
  | `> 10` | **Reject.** Do not auto-clear. | `NON_UNIQUE` | — |

  This deliberately **lowers the auto-clear rate and raises precision toward 1.0**, which is the correct trade for a financial ledger: an unevidenced match is worse than an honest exception. Rejected credits surface as `SUM_COLLISION` in the exception queue with the solution count attached.

  The count is scoped to a ±14-day window around the credit's `value_date` (mirroring T3's widest window) to keep the enumeration tractable at scale.

## The Exception Queue

Every credit the engine declines to clear becomes a structured row via `build_exception_report()`, not a raw dataframe dump. Each carries a `break_code` (`WINDOW_DEFICIT`, `NO_CANDIDATES`, `SUM_COLLISION`, `UNRESOLVED`, or T3's classification), `materiality` (HIGH >₹50k, MEDIUM >₹5k, LOW), `age_days`, `delta_inr` against the closest subset found, the T3 `hypothesis`, a derived `suggested_action`, and JSON `evidence` naming the closest candidate entry_ids. A break is only actionable if the controller can see *why* it broke and *what to do next*.

## Cash Position Module

`recon/cashflow.py` turns the verified reconciliation output into a forward-looking treasury view — reconciliation is a means, cash visibility is the end.

`build_cash_position()` returns `verified_settled` (value inside auto-cleared credits), `at_risk` (credits stuck in `UNRESOLVED`/`SUM_COLLISION`), a 7-day `expected_inflow` forecast (unassigned payments projected to credit at `settled_at + settlement_cycle`), a `confidence_interval` bracketing the forecast between "every at-risk credit is lost" and "every one is recovered", and `cash_at_risk_by_age` bucketed `<3d / 3-7d / >7d`.

* **Design Decision (The uniqueness gate pays for itself here):** because T4 refuses to auto-clear unevidenced matches, `verified_settled` is a figure a controller can actually rely on. A higher auto-clear rate bought with coin-flip matches would make this number *confidently wrong* — the worst possible property for a cash forecast.
* All arithmetic runs in integer paise and converts to rupees only at the boundary.

## Audit Trail

`export_audit_trail()` emits one row per payment so any assignment can be defended after the fact:

| Column | Meaning |
|---|---|
| `run_id` | uuid4 generated at `reconcile()` call time and stamped onto `result_df.attrs`, so every row ties back to the run that produced it (overridable via `reconcile(..., run_id=...)`) |
| `entry_id` / `payment_id` | ledger row identity |
| `assigned_utr` | the credit it cleared into, or NULL |
| `assigned_tier` / `tier_name` | which tier made the call (e.g. `T2 · Subset-sum DP`) |
| `reason_code` | `UNIQUE_MATCH` / `AMBIGUOUS_PARTIAL` / `NON_UNIQUE` / `SUM_MATCH` / `UNRESOLVED` |
| `confidence` | float |
| `evidence` | JSON — e.g. `{"tier": 3, "solution_count": 1}` or `{"tier": 2, "excess_paise": 145}`. `solution_count` comes from T4's uniqueness gate; `excess_paise` from T2's exclusion-DP |
| `reconciled_at` | ISO-8601 UTC timestamp |

Downloadable as CSV from the Reconciliation page.

## Evaluation Harness

Because the engine is stochastic at T3, it requires a formal evaluation harness. The custom harness (`eval/harness.py`) generates synthetic ledgers with injected breaks (late settlements, refunds, chargebacks) and evaluates the engine's Precision, Recall, and Auto-Clear Rate at every tier. It also reports wall-clock throughput.

### Measured results — 5,538 payments / 83 credits, seed 42, medium difficulty

| metric | T0 | T0..T1 | T0..T2 | T0..T3 | T0..T4 | T0..T4 @0.95 |
|---|---|---|---|---|---|---|
| auto_clear_rate | 0.000 | 0.000 | 0.701 | 0.772 | 0.701 | 0.044 |
| precision | – | – | 0.625 | **0.578** | **0.625** | **0.808** |
| credits_cleared | 0/83 | 0/83 | 55/83 | 64/83 | 55/83 | 3/83 |

The T3 → T4 columns are the uniqueness gate in one line: T3's extra claims raise auto-clear to 0.772 but *drop* precision to 0.578. T4 rejects the unevidenced claims, and precision recovers to 0.625. Tightening the confidence gate to 0.95 pushes precision to 0.808. **Precision is bought with auto-clear rate, deliberately.**

`Reconciled 5538 payments across 83 credits in 1.72s — Throughput: 3,212 payments/sec.`

### Performance: the bitset subset-sum

T2's inner loop is the whole engine's cost centre. The original DP walked a Python dict of reachable sums, one step per (item, reachable-sum) pair — millions of interpreter iterations per credit, and it gave up at an iteration cap, abandoning credits it could have solved.

It now carries the reachable-sum set as a **bitmask inside one Python integer**: bit *s* set means "sum *s* is reachable", so a whole relaxation step is `reach |= reach << v` — a single C-speed big-integer shift instead of a Python loop. Per-prefix snapshots of the mask let the chosen items be recovered by walking backwards (if the target was already reachable without item *k*, it wasn't needed). The budget is now memory (bits) rather than iterations.

The effect is not just speed. Because the search now *completes* instead of hitting an iteration cap, T2 clears **55/83 credits instead of 30/83**, lifting overall recall from 0.304 to 0.438:

| | before | after |
|---|---|---|
| full pipeline | 233.7s | **1.72s** |
| throughput | 24 /sec | **3,212 /sec** |
| `make eval` (ablation) | ~20 min | **7s** |
| credits cleared | 30/83 | **55/83** |

### A scaling note on date windows

T1 and T2 partition candidates by settlement-date window, anchoring each window to the **previous distinct `value_date`** — several credits can share a `value_date`, and keying off the immediately preceding row hands every credit after the first an inverted, empty window.

That constraint also shapes the generator: `capture_window_days` scales with `num_payments` (270 days at 5,000 payments). Cramming 5,000 payments into the original 30-day window puts up to 9 credits on a single date, which collapses every window into the same multi-batch candidate pool and inflates the exclusion-DP's excess from the designed ₹50–500 to ₹15,000–33,000 — past the iteration cap, clearing **zero** credits.
