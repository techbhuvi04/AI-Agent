from dataclasses import dataclass
from enum import Enum
from typing import Dict


class Difficulty(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


BREAK_TYPES = [
    "late_settlement",
    "refund",
    "chargeback_reversal",
    "netting_split",
    "duplicate_utr",
    "rounding_drift",
    "missing_order",
]

BREAK_FREQUENCIES: Dict[Difficulty, Dict[str, float]] = {
    Difficulty.EASY: {
        "late_settlement": 0.03,
        "refund": 0.02,
        "chargeback_reversal": 0.01,
        "netting_split": 0.00,
        "duplicate_utr": 0.00,
        "rounding_drift": 0.02,
        "missing_order": 0.02,
    },
    Difficulty.MEDIUM: {
        "late_settlement": 0.08,
        "refund": 0.05,
        "chargeback_reversal": 0.03,
        "netting_split": 0.02,
        "duplicate_utr": 0.01,
        "rounding_drift": 0.05,
        "missing_order": 0.04,
    },
    Difficulty.HARD: {
        "late_settlement": 0.15,
        "refund": 0.10,
        "chargeback_reversal": 0.06,
        "netting_split": 0.05,
        "duplicate_utr": 0.03,
        "rounding_drift": 0.10,
        "missing_order": 0.08,
    },
}


@dataclass
class GeneratorConfig:
    seed: int = 42
    num_payments: int = 5000
    difficulty: Difficulty = Difficulty.MEDIUM
    fee_rate: float = 0.02
    gst_rate: float = 0.18
    settlement_cycle: int = 2
    batch_size_min: int = 80
    batch_size_max: int = 160
    # Scaled with num_payments: the tiered engine partitions candidates by
    # settlement-date window, so payments must spread across enough dates
    # that batches mostly land on their own value_date. Cramming 5000
    # payments into a 30-day window puts up to 9 credits on one date,
    # which collapses every window into the same multi-batch candidate
    # pool and makes the exclusion-DP's excess intractable.
    capture_window_days: int = 270
    rounding_drift_max_paise: int = 4
    output_dir: str = "data"
