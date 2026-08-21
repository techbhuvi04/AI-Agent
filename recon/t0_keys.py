import pandas as pd


def run(orders, settlements):
    enriched = settlements.merge(
        orders, on="payment_id", how="left", suffixes=("", "_order")
    )
    enriched["order_matched"] = enriched["order_id"].notna()
    return enriched
