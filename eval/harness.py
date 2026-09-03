import argparse
import time

import pandas as pd

from recon.engine import load_data, load_ground_truth, reconcile
from eval.metrics import compute_metrics


def evaluate(data_dir="data", max_tier=1, min_confidence=0.0):
    orders, settlements, bank = load_data(data_dir)
    ground_truth = load_ground_truth(data_dir)

    start = time.perf_counter()
    result, cleared = reconcile(orders, settlements, bank, max_tier=max_tier, min_confidence=min_confidence)
    elapsed = time.perf_counter() - start

    metrics = compute_metrics(result, ground_truth, cleared, bank)
    metrics["timing"] = {
        "elapsed_seconds": elapsed,
        "num_payments": len(result),
        "num_credits": len(bank),
        "throughput": len(result) / elapsed if elapsed > 0 else 0.0,
    }
    return metrics


def print_timing(metrics):
    t = metrics.get("timing")
    if not t:
        return
    print(
        f"\n  Reconciled {t['num_payments']} payments across {t['num_credits']} "
        f"credits in {t['elapsed_seconds']:.2f}s"
    )
    print(f"  Throughput: {t['throughput']:.0f} payments/sec")


def run_ablation(data_dir="data", max_tier_limit=4, precision_gate=0.95):
    results = {}
    for mt in range(max_tier_limit + 1):
        label = f"T0..T{mt}" if mt > 0 else "T0 only"
        results[label] = evaluate(data_dir, max_tier=mt)
    if precision_gate > 0:
        label = f"T0..T{max_tier_limit} @{precision_gate}"
        results[label] = evaluate(
            data_dir, max_tier=max_tier_limit, min_confidence=precision_gate
        )
    return results


def run_curve(data_dir="data", max_tier=4):
    """Sweep the confidence gate.

    The gate is applied *after* every tier has run, so all thresholds
    share one reconciliation — re-running the tiers per threshold would
    repeat identical work. We reconcile once ungated and then re-apply
    each threshold to that single result.
    """
    thresholds = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]

    orders, settlements, bank = load_data(data_dir)
    ground_truth = load_ground_truth(data_dir)

    start = time.perf_counter()
    base_result, _base_cleared = reconcile(
        orders, settlements, bank, max_tier=max_tier, min_confidence=0.0
    )
    elapsed = time.perf_counter() - start

    results = {}
    for t in thresholds:
        gated, cleared = _apply_confidence_gate(base_result, t)
        metrics = compute_metrics(gated, ground_truth, cleared, bank)
        metrics["timing"] = {
            "elapsed_seconds": elapsed,
            "num_payments": len(gated),
            "num_credits": len(bank),
            "throughput": len(gated) / elapsed if elapsed > 0 else 0.0,
        }
        results[t] = metrics
    return results


def _apply_confidence_gate(result_df, min_confidence):
    """Re-derive (result, cleared) at a confidence threshold without
    re-running the tiers. Mirrors the gating reconcile() applies."""
    gated = result_df.copy()

    if min_confidence > 0.0:
        low_conf = gated["assigned_confidence"] < min_confidence
        if low_conf.any():
            gated.loc[low_conf, ["assigned_utr", "assigned_tier", "assigned_confidence"]] = [
                pd.NA, pd.NA, pd.NA
            ]

    cleared = {}
    for idx, utr in gated["assigned_utr"].items():
        if pd.notna(utr):
            cleared.setdefault(utr, []).append(idx)

    return gated, cleared


def _fmt(val, width=12):
    if val is None:
        return "-".center(width)
    if isinstance(val, float):
        return f"{val:.3f}".center(width)
    return str(val).center(width)


def print_ablation(results):
    labels = list(results.keys())

    print(f"\n{'='*70}")
    print(f"  Eval Harness — Tier Ablation")
    print(f"{'='*70}")

    header = f"  {'metric':<20}"
    for label in labels:
        header += f" {label:>14}"
    print(header)
    print(f"  {'-'*20}" + f" {'-'*14}" * len(labels))

    for metric_name in ["auto_clear_rate", "precision", "recall", "accuracy"]:
        row = f"  {metric_name:<20}"
        for label in labels:
            row += f" {_fmt(results[label]['overall'][metric_name], 14)}"
        print(row)

    credits_row = f"  {'credits_cleared':<20}"
    for label in labels:
        o = results[label]["overall"]
        val = f"{o['credits_cleared']}/{o['total_credits']}"
        credits_row += f" {val:>14}"
    print(credits_row)

    payments_row = f"  {'payments_assigned':<20}"
    for label in labels:
        o = results[label]["overall"]
        val = f"{o['payments_assigned']}/{o['total_payments']}"
        payments_row += f" {val:>14}"
    print(payments_row)

    print(f"\n  Per-break recall:")
    header2 = f"  {'break_type':<20} {'total':>6}"
    for label in labels:
        header2 += f" {label:>14}"
    print(header2)
    print(f"  {'-'*20} {'-'*6}" + f" {'-'*14}" * len(labels))

    all_breaks = []
    for r in results.values():
        for bt in r["per_break"]:
            if bt not in all_breaks:
                all_breaks.append(bt)

    for bt in all_breaks:
        first_result = next(iter(results.values()))
        bt_total = first_result["per_break"].get(bt, {}).get("total", 0)
        row = f"  {bt:<20} {bt_total:>6}"
        for label in labels:
            m = results[label]["per_break"].get(bt)
            if m:
                row += f" {m['recall']:>14.3f}"
            else:
                row += f" {'-':>14}"
        print(row)

    # Timing for the full-pipeline run (the last un-gated tier column).
    full_pipeline = [
        r for label, r in results.items()
        if "@" not in label and r.get("timing")
    ]
    if full_pipeline:
        print_timing(full_pipeline[-1])

    print()


def print_single(metrics, label=""):
    overall = metrics["overall"]
    per_break = metrics["per_break"]

    print(f"\n{'='*60}")
    if label:
        print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Credits cleared:  {overall['credits_cleared']} / {overall['total_credits']}")
    print(f"  Auto-clear rate:  {_fmt(overall['auto_clear_rate'])}")
    print(f"  Precision:        {_fmt(overall['precision'])}")
    print(f"  Recall:           {_fmt(overall['recall'])}")
    print(f"  Accuracy:         {_fmt(overall['accuracy'])}")

    print(f"\n  {'break_type':<25} {'total':>6} {'correct':>8} {'prec':>8} {'recall':>8}")
    print(f"  {'-'*25} {'-'*6} {'-'*8} {'-'*8} {'-'*8}")

    for bt, m in per_break.items():
        p = f"{m['precision']:.3f}" if m["precision"] is not None else "-"
        print(f"  {bt:<25} {m['total']:>6} {m['correct']:>8} {p:>8} {m['recall']:>8.3f}")

    print_timing(metrics)


def print_curve(results):
    print(f"\n{'='*60}")
    print(f"  Precision / Auto-Clear Curve (Confidence Gate)")
    print(f"{'='*60}")
    print(f"  {'min_conf':<10} | {'auto_clear':>12} | {'precision':>12} | {'cleared':>10}")
    print(f"  {'-'*10}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}")
    
    for t, m in results.items():
        o = m["overall"]
        ac = _fmt(o["auto_clear_rate"], 12)
        pr = _fmt(o["precision"], 12)
        cl = f"{o['credits_cleared']}/{o['total_credits']}"
        print(f"  {t:<10.2f} | {ac} | {pr} | {cl:>10}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Reconciliation eval harness")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--max-tier", type=int, default=1)
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--curve", action="store_true")
    args = parser.parse_args()

    if args.ablation:
        results = run_ablation(args.data_dir, max_tier_limit=args.max_tier)
        print_ablation(results)
    elif args.curve:
        results = run_curve(args.data_dir, max_tier=args.max_tier)
        print_curve(results)
    else:
        metrics = evaluate(args.data_dir, args.max_tier)
        print_single(metrics, f"Tiers T0..T{args.max_tier}")


if __name__ == "__main__":
    main()
