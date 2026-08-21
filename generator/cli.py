import argparse
import os
from collections import Counter

from generator.config import GeneratorConfig, Difficulty
from generator.truth import build_truth
from generator.views import to_orders, to_settlements, to_bank, truth_to_rupees


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic reconciliation data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="medium")
    parser.add_argument("--num-payments", type=int, default=500)
    parser.add_argument("--output-dir", type=str, default="data")
    args = parser.parse_args()

    config = GeneratorConfig(
        seed=args.seed,
        difficulty=Difficulty(args.difficulty),
        num_payments=args.num_payments,
        output_dir=args.output_dir,
    )

    truth = build_truth(config)

    os.makedirs(config.output_dir, exist_ok=True)

    orders = to_orders(truth)
    settlements = to_settlements(truth)
    bank = to_bank(truth)
    truth_rupees = truth_to_rupees(truth)

    orders.to_csv(os.path.join(config.output_dir, "orders.csv"), index=False, float_format="%.2f")
    settlements.to_csv(os.path.join(config.output_dir, "settlements.csv"), index=False, float_format="%.2f")
    bank.to_csv(os.path.join(config.output_dir, "bank.csv"), index=False, float_format="%.2f")
    truth_rupees.to_csv(os.path.join(config.output_dir, "ground_truth.csv"), index=False, float_format="%.2f")

    break_counts = Counter(truth[truth["is_original"]]["break_type"])

    print(f"\nGenerated {len(truth)} truth rows ({config.num_payments} payments + adjustments)")
    print(f"Difficulty: {config.difficulty.value}")
    print(f"Seed: {config.seed}")
    print(f"\nOrders:      {len(orders)} rows")
    print(f"Settlements: {len(settlements)} rows")
    print(f"Bank:        {len(bank)} credits")
    print(f"\nBreak distribution (original payments):")
    print(f"  {'type':<25} {'count':>6}")
    print(f"  {'-'*25} {'-'*6}")
    for bt in ["clean"] + sorted(k for k in break_counts if k != "clean"):
        print(f"  {bt:<25} {break_counts.get(bt, 0):>6}")


if __name__ == "__main__":
    main()
