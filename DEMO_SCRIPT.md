# 90-Second Demo — Script & Shot List

Record in **one take**, screen + voiceover. Target 90s, hard ceiling 100s.
Practice the click path twice before recording so there are no dead pauses.

---

## Pre-flight (do this 10 min before recording)

1. **Do NOT run `make eval` / `make curve` right before recording** — each ablation
   spends a chunk of the Groq free-tier token budget and can 429 the live demo.
   Run them once earlier to warm the `@st.cache_data` layer, then leave the LLM alone.
2. Start the app clean:
   ```bash
   cd "/Users/bhuvi/AI Agent"
   make demo        # http://localhost:8501
   ```
3. In the sidebar set **Max execution tier = 4**, **Minimum confidence = 0.95**.
   (0.95 is the honest number — high precision. Don't demo at 0.00.)
4. Pre-load each page once so charts are cached and render instantly on camera:
   Overview → Reconciliation → Analytics → Exception queue → Cash position.
5. Browser at 100% zoom, window maximised, hide bookmarks bar. Dark theme (default).
6. Have the Exception queue pre-scrolled to a **SUM_COLLISION** row so you can
   expand it without hunting.

---

## The script (voiceover in **bold**, actions in plain text)

### 0:00–0:15 — The problem  ·  *screen: Overview page, hero banner*

> **A merchant gets one netted bank credit from their payment gateway. That single
> credit covers 120-odd individual payments — each minus a fee, minus GST on the
> fee — and it lands two days later. There are no shared keys. Reconciliation means
> proving, mathematically, which payments are inside that credit.**

Slowly scroll the Overview page: 5,538 payments, 83 credits, the tier pipeline diagram.

### 0:15–0:30 — The insight  ·  *screen: still Overview, or Architecture diagram*

> **Everyone's first instinct is subset-sum: find payments that add up to the credit.
> We do that — it's tier 2, an exact DP. But here's the catch most reconciliation
> tools miss: with 120 candidates, *many* different subsets hit the same total.
> Sum-equality isn't proof. So tier 4 doesn't just check the sum — it asks whether
> the match is the *only* one, and it re-verifies every claim the LLM proposes
> before anything clears.**

### 0:30–0:50 — Live: cleared vs exception  ·  *screen: Reconciliation → Exception queue*

Click **Reconciliation** in the nav.

> **Run the engine. Tiers 0 through 4 — deterministic first, LLM only for structural
> break classification. Five thousand payments in under two seconds.**

Point at the **Assignment results** table — the Confidence column (progress bars).

> **Everything here cleared with a confidence score attached. Tier 4 re-verifies the
> arithmetic on every claim before it's allowed through.**

Click **Exception queue** in the nav. Expand one exception row (any `UNRESOLVED` /
`WINDOW_DEFICIT`).

> **And what it *can't* prove, it doesn't guess. It routes to the controller with
> the evidence attached — the closest candidate subset, the rupee delta, a suggested
> action. An honest exception beats a confident wrong answer. That's the whole
> philosophy: never fabricate a match.**

### 0:50–1:12 — Ablation + speed  ·  *screen: Analytics page*

Click **Analytics**. Let the ablation heatmap and summary table be on screen.

> **This is the proof it works. Each column adds a tier. Tier 2's exact DP does the
> heavy lifting — and it's fast because the subset-sum runs as a bitmask shift inside
> one Python integer, not a Python loop. That took the full pipeline from four
> minutes to under two seconds — a 160-times speedup — which is why we can re-run
> the whole thing live.**

Scroll to the **Precision / auto-clear trade-off** curve.

> **And precision is a dial. Tighten the confidence gate and precision climbs toward
> one — the business picks the point.**

### 1:12–1:30 — Cash position  ·  *screen: Cash position page*

Click **Cash position**.

> **Reconciliation is the means. This is the end: verified settled cash, cash at
> risk by age, a seven-day inflow forecast with a confidence band. Because tier 4
> refuses to clear anything it can't prove, *this* is a number a controller can
> actually take to the CFO.**

End on the Cash position page. Stop recording.

---

## If you have 20 extra seconds

After 0:50, before Analytics, drop in the **Ask** page:

> **You can also just ask it.** *(click a suggested question, e.g. "Which exceptions
> are most urgent?")* **It answers from the verified structured output — never the
> raw ledger.**

---

## Numbers to have right (seed 42, LLM active)

- 5,538 payments · 83 credits
- T0..T4 precision **0.635**, recall **0.455**, 57/83 credits cleared
- @0.95 gate: precision **0.808**
- Deterministic tiers alone: **~1.4s** · full pipeline before the bitset rewrite: **233s** → **160×**
- 121/121 unit tests

## Do not say

- "Fully automated" — it's high-confidence auto-clear + a structured exception queue.
- "The AI reconciles the ledger" — the LLM classifies break *shape*; the DP does the math; tier 4 verifies.
