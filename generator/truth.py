import numpy as np
import pandas as pd
from datetime import date, timedelta

from generator.config import GeneratorConfig, BREAK_FREQUENCIES, BREAK_TYPES


def _add_working_days(start, days):
    current = start
    remaining = abs(days)
    step = 1 if days >= 0 else -1
    while remaining > 0:
        current += timedelta(days=step)
        if current.weekday() < 5:
            remaining -= 1
    return current


def _next_weekday(d):
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def _generate_base_payments(config):
    rng = np.random.default_rng(config.seed)
    start = date(2025, 1, 6)

    records = []
    for i in range(config.num_payments):
        gross = int(rng.integers(200, 50001))
        offset = int(rng.integers(0, config.capture_window_days))
        captured_at = _next_weekday(start + timedelta(days=offset))
        fee = round(gross * config.fee_rate)
        gst = round(fee * config.gst_rate)
        net = gross - fee - gst

        records.append({
            "order_id": f"ORD_{i:06d}",
            "payment_id": f"PAY_{i:06d}",
            "gross": gross,
            "fee": fee,
            "gst": gst,
            "net": net,
            "captured_at": captured_at,
            "settled_at": _add_working_days(captured_at, config.settlement_cycle),
            "credit_utr": None,
            "credit_batch_id": None,
            "break_type": "clean",
        })

    return records, rng


def _assign_break_types(records, config, rng):
    freqs = BREAK_FREQUENCIES[config.difficulty]
    thresholds = []
    cumulative = 0.0
    for bt in BREAK_TYPES:
        cumulative += freqs[bt]
        thresholds.append((cumulative, bt))

    for record in records:
        roll = float(rng.random())
        for threshold, bt in thresholds:
            if roll < threshold:
                record["break_type"] = bt
                break

    return records


def _apply_pre_credit_mutations(records, config, rng):
    for record in records:
        if record["break_type"] == "late_settlement":
            extra = int(rng.integers(1, 4))
            record["settled_at"] = _add_working_days(record["settled_at"], extra)

        elif record["break_type"] == "rounding_drift":
            drift = int(rng.integers(1, config.rounding_drift_max_paise + 1))
            if rng.random() < 0.5:
                drift = -drift
            record["net"] += drift

    return records


def _assign_credits(records, config, rng):
    by_date = sorted(records, key=lambda r: (r["settled_at"], r["payment_id"]))

    batches = []
    batch_id = 0
    utr_seq = 0
    used_utrs = []
    pos = 0

    while pos < len(by_date):
        remaining = len(by_date) - pos
        cap = min(
            int(rng.integers(config.batch_size_min, config.batch_size_max + 1)),
            remaining,
        )
        chunk = by_date[pos : pos + cap]
        pos += cap

        has_split = any(r["break_type"] == "netting_split" for r in chunk)

        if has_split and len(chunk) >= 10:
            lo = max(len(chunk) // 4, 3)
            hi = min(len(chunk) * 3 // 4, len(chunk) - 3)
            if lo >= hi:
                split = len(chunk) // 2
            else:
                split = int(rng.integers(lo, hi + 1))

            part_a, part_b = chunk[:split], chunk[split:]

            utr_seq += 1
            utr_a = f"UTR{utr_seq:012d}"
            for r in part_a:
                r["credit_utr"] = utr_a
                r["credit_batch_id"] = batch_id
            batches.append({"batch_id": batch_id, "utr": utr_a})
            used_utrs.append(utr_a)
            batch_id += 1

            utr_seq += 1
            utr_b = f"UTR{utr_seq:012d}"
            for r in part_b:
                r["credit_utr"] = utr_b
                r["credit_batch_id"] = batch_id
            batches.append({"batch_id": batch_id, "utr": utr_b})
            used_utrs.append(utr_b)
            batch_id += 1

        else:
            has_dup = any(r["break_type"] == "duplicate_utr" for r in chunk)

            if has_dup and used_utrs:
                utr = str(rng.choice(used_utrs))
            else:
                utr_seq += 1
                utr = f"UTR{utr_seq:012d}"

            for r in chunk:
                r["credit_utr"] = utr
                r["credit_batch_id"] = batch_id

            batches.append({"batch_id": batch_id, "utr": utr})
            used_utrs.append(utr)
            batch_id += 1

    return by_date, batches


def _create_adjustments(records, batches, config, rng):
    batch_ids = sorted(b["batch_id"] for b in batches)
    utr_of = {b["batch_id"]: b["utr"] for b in batches}

    latest_date = {}
    for r in records:
        bid = r["credit_batch_id"]
        if bid not in latest_date or r["settled_at"] > latest_date[bid]:
            latest_date[bid] = r["settled_at"]

    adjustments = []

    for record in records:
        bt = record["break_type"]
        cur = record["credit_batch_id"]
        later = [b for b in batch_ids if b > cur]

        if bt == "refund":
            if not later:
                record["break_type"] = "clean"
                continue
            target = int(rng.choice(later))
            adjustments.append({
                "order_id": record["order_id"],
                "payment_id": record["payment_id"],
                "gross": -record["gross"],
                "fee": -record["fee"],
                "gst": -record["gst"],
                "net": -record["net"],
                "captured_at": record["captured_at"],
                "settled_at": latest_date[target],
                "credit_utr": utr_of[target],
                "credit_batch_id": target,
                "break_type": "refund",
            })

        elif bt == "chargeback_reversal":
            if len(later) < 2:
                record["break_type"] = "clean"
                continue

            debit_idx = int(rng.integers(0, len(later) - 1))
            debit_bid = later[debit_idx]
            reversal_candidates = [b for b in later if b > debit_bid]

            if not reversal_candidates:
                record["break_type"] = "clean"
                continue

            reversal_bid = int(rng.choice(reversal_candidates))

            adjustments.append({
                "order_id": record["order_id"],
                "payment_id": record["payment_id"],
                "gross": -record["gross"],
                "fee": 0,
                "gst": 0,
                "net": -record["gross"],
                "captured_at": record["captured_at"],
                "settled_at": latest_date[debit_bid],
                "credit_utr": utr_of[debit_bid],
                "credit_batch_id": debit_bid,
                "break_type": "chargeback_reversal",
            })

            adjustments.append({
                "order_id": record["order_id"],
                "payment_id": record["payment_id"],
                "gross": record["gross"],
                "fee": 0,
                "gst": 0,
                "net": record["gross"],
                "captured_at": record["captured_at"],
                "settled_at": latest_date[reversal_bid],
                "credit_utr": utr_of[reversal_bid],
                "credit_batch_id": reversal_bid,
                "break_type": "chargeback_reversal",
            })

    return adjustments


TRUTH_COLUMNS = [
    "order_id",
    "payment_id",
    "gross",
    "fee",
    "gst",
    "net",
    "captured_at",
    "settled_at",
    "credit_utr",
    "credit_batch_id",
    "break_type",
    "is_original",
]


def build_truth(config):
    records, rng = _generate_base_payments(config)
    records = _assign_break_types(records, config, rng)
    records = _apply_pre_credit_mutations(records, config, rng)
    records, batches = _assign_credits(records, config, rng)
    adjustments = _create_adjustments(records, batches, config, rng)

    for r in records:
        r["is_original"] = True
    for a in adjustments:
        a["is_original"] = False

    all_records = records + adjustments
    return pd.DataFrame(all_records)[TRUTH_COLUMNS]
