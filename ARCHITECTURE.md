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
* **Design Decision (Heuristic fallback):** With `GROQ_API_KEY` unset, T3 falls back to a deterministic refund-aware wide-window matcher, so the pipeline runs end to end without any LLM. The same fallback catches a mid-run quota exhaustion: the first worker to see a `429` sets a shared circuit-breaker `Event`, later workers skip the network entirely, and whatever is still uncleared is handed to the heuristic solver — pre-seeded with the entries T3 has already claimed this run so it can't double-assign them.
* **Design Decision (Parallel classify, sequential resolve):** the per-credit LLM calls are independent network round-trips, so they fan out across a thread pool against a *frozen* snapshot of the used-index set (used only to shape the prompt — a stale snapshot can at worst make a suggestion less apt). Deterministic resolution then runs strictly sequentially in bank-sorted order against the *live* used-index set, which is what prevents two credits resolving onto the same payment.
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
  | `2 – 10`, enumeration exhaustive | Accept only the **intersection** of all valid solutions (entries in *every* subset are proven regardless of which is true); flag the rest as ambiguous | `AMBIGUOUS_PARTIAL` | `0.75` |
  | `2 – 10`, enumeration **not** exhaustive | Accept the proposed subset — it already passed the ±₹0.50 sum check and all its indices are available — at reduced confidence rather than discarding T3's work | `AMBIGUOUS_PARTIAL` | `0.65` |
  | `> 10` | **Reject.** Do not auto-clear. | `NON_UNIQUE` | — |

  This deliberately **lowers the auto-clear rate and raises precision**, which is the correct trade for a financial ledger: an unevidenced match is worse than an honest exception. Rejected credits surface as `SUM_COLLISION` in the exception queue with the solution count attached.

  **Why the non-exhaustive branch exists.** With 100–140 candidate entries in a wide window, the enumeration DP hits its state ceiling (`MAX_DP_STATES`) before it can prove *anything* — `exhaustive` comes back `False` having found zero solutions. Treating that as `NON_UNIQUE` rejected **every** T3 claim, so the `T0..T4` column collapsed back onto the `T0..T2` numbers and T3 contributed nothing through the verifier. The fix is two-part: (1) scope the candidate pool to a **±7-day** first-pass window so the enumeration reaches `exhaustive=True` far more often, letting the normal `AMBIGUOUS_PARTIAL` path fire; (2) when it still doesn't, fall back to accepting the sum-checked subset at `0.65` confidence — lower than a proven `0.75`, high enough to auto-clear, and still gated by the confidence slider for anyone who wants only proven matches.

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
| `reason_code` | `UNIQUE_MATCH` / `AMBIGUOUS_PARTIAL` / `NON_UNIQUE` / `SUM_MATCH` / `UNRESOLVED` (`AMBIGUOUS_PARTIAL` covers both the exhaustive-intersection case at `0.75` and the non-exhaustive sum-checked fallback at `0.65`) |
| `confidence` | float |
| `evidence` | JSON — e.g. `{"tier": 3, "solution_count": 1}` or `{"tier": 2, "excess_paise": 145}`. `solution_count` comes from T4's uniqueness gate; `excess_paise` from T2's exclusion-DP |
| `reconciled_at` | ISO-8601 UTC timestamp |

Downloadable as CSV from the Reconciliation page.

## Evaluation Harness

Because the engine is stochastic at T3, it requires a formal evaluation harness. The custom harness (`eval/harness.py`) generates synthetic ledgers with injected breaks (late settlements, refunds, chargebacks) and evaluates the engine at every tier. It also reports wall-clock throughput.

**Metrics** (`eval/metrics.py`):

| metric | definition |
|---|---|
| `auto_clear_rate` | payments assigned / total — *coverage*, not correctness |
| `precision` | payments assigned **to the correct credit** / payments assigned |
| `recall` | correct assignments / total payments |
| `accuracy` | correct assignments / total payments — the fraction of the ledger placed correctly (equal to recall on this task). Replaces an earlier "F1" that combined a precision numerator over *assigned* with a recall denominator over *all payments* — different denominators, not a well-defined F1. |

Ground truth is joined to the engine output **by `payment_id`** (`set_index("payment_id").reindex(...)`) before any comparison — the two CSVs are never assumed to be in the same row order. `t0_keys` merges with `validate="m:1"` so a duplicate `payment_id` on the orders side raises instead of silently fanning out rows and corrupting every metric.

### Measured results — 5,538 payments / 83 credits, seed 42, medium difficulty

`GROQ_API_KEY` unset (T3 = deterministic heuristic), isolating the algorithmic behaviour:

| metric | T0 | T0..T1 | T0..T2 | T0..T3 | T0..T4 | T0..T4 @0.95 |
|---|---|---|---|---|---|---|
| auto_clear_rate | 0.000 | 0.000 | 0.701 | 0.772 | **0.772** | 0.044 |
| precision | – | – | 0.625 | 0.578 | **0.578** | **0.808** |
| recall / accuracy | 0.000 | 0.000 | 0.438 | 0.447 | **0.447** | 0.036 |
| credits_cleared | 0/83 | 0/83 | 55/83 | 64/83 | **64/83** | 3/83 |

T3 adds 9 credits (55 → 64) and raises auto-clear to 0.772, dropping precision to 0.578 — the heuristic's extra matches aren't all right. T4 now *keeps* those 64 via its non-exhaustive `AMBIGUOUS_PARTIAL @ 0.65` branch instead of rejecting them all. Tightening the confidence gate to 0.95 trades almost all the auto-clear rate back for **0.808 precision**. **Precision is bought with auto-clear rate, deliberately** — `@0.95` is the setting a controller who needs a trustworthy figure would run.

Previously the `T0..T4` column read identically to `T0..T2` (55/83, precision 0.625): the uniqueness gate rejected every T3 claim because the enumeration DP hit its state ceiling on 100–140-entry pools before proving anything. Narrowing the pool window to ±7 days and adding the non-exhaustive branch is what puts T3 into production through the verifier.

`Reconciled 5538 payments across 83 credits in ~1.4s — Throughput ~3,900 payments/sec.`

### Performance: the bitset subset-sum

T2's inner loop is the whole engine's cost centre. The original DP walked a Python dict of reachable sums, one step per (item, reachable-sum) pair — millions of interpreter iterations per credit, and it gave up at an iteration cap, abandoning credits it could have solved.

It now carries the reachable-sum set as a **bitmask inside one Python integer**: bit *s* set means "sum *s* is reachable", so a whole relaxation step is `reach |= reach << v` — a single C-speed big-integer shift instead of a Python loop. Per-prefix snapshots of the mask let the chosen items be recovered by walking backwards (if the target was already reachable without item *k*, it wasn't needed). The budget is now memory (bits) rather than iterations.

The effect is not just speed. Because the search now *completes* instead of hitting an iteration cap, T2 clears **55/83 credits instead of 30/83**, lifting overall recall from 0.304 to 0.438:

| | before | after |
|---|---|---|
| full pipeline | 233.7s | **~1.4s** |
| throughput | 24 /sec | **~3,900 /sec** |
| `make eval` (ablation) | ~20 min | **~10s** |
| credits cleared (T2) | 30/83 | **55/83** |

### A scaling note on date windows

T1 and T2 partition candidates by settlement-date window, anchoring each window to the **previous distinct `value_date`** — several credits can share a `value_date`, and keying off the immediately preceding row hands every credit after the first an inverted, empty window.

That constraint also shapes the generator: `capture_window_days` scales with `num_payments` (270 days at 5,000 payments). Cramming 5,000 payments into the original 30-day window puts up to 9 credits on a single date, which collapses every window into the same multi-batch candidate pool and inflates the exclusion-DP's excess from the designed ₹50–500 to ₹15,000–33,000 — past the iteration cap, clearing **zero** credits.
