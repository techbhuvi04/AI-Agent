"""Forward cash position derived from verified reconciliation output.

All monetary arithmetic is done in integer paise and converted back to
rupees only at the boundary, so the totals here reconcile exactly with
what the engine cleared.
"""

from datetime import date, datetime, timedelta

import pandas as pd


AT_RISK_BREAK_CODES = {"UNRESOLVED", "SUM_COLLISION"}
FORECAST_DAYS = 7


def _to_paise(v):
    return int(round(v * 100))


def _to_rupees(paise):
    return round(paise / 100, 2)


def _coerce_date(d):
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    return pd.to_datetime(d).date()


def build_cash_position(result_df, bank_df, exception_report_df, run_date,
                        settlement_cycle=2):
    """Compute a 7-day forward cash position.

    verified_settled — payments sitting inside auto-cleared credits
    at_risk          — credits stuck in UNRESOLVED / SUM_COLLISION exceptions
    expected_inflow  — unassigned payments projected to credit at
                       settled_at + settlement_cycle days, bucketed by
                       projected credit date over the next 7 days
    """
    run_date = _coerce_date(run_date)

    assigned_mask = result_df["assigned_utr"].notna()
    verified_settled_paise = sum(
        _to_paise(v) for v in result_df.loc[assigned_mask, "net"]
    )

    at_risk_paise = 0
    if exception_report_df is not None and len(exception_report_df):
        at_risk_rows = exception_report_df[
            exception_report_df["break_code"].isin(AT_RISK_BREAK_CODES)
        ]
        at_risk_paise = sum(_to_paise(v) for v in at_risk_rows["credit_amount"])

    # --- 7-day forward inflow -------------------------------------------
    unassigned = result_df.loc[~assigned_mask]
    expected_inflow = {}
    forecast_dates = [run_date + timedelta(days=i) for i in range(FORECAST_DAYS)]
    for d in forecast_dates:
        expected_inflow[d.isoformat()] = 0.0

    if len(unassigned):
        settled = pd.to_datetime(unassigned["settled_at"]).dt.date
        projected = settled.apply(lambda d: d + timedelta(days=settlement_cycle))

        inflow_paise = {d.isoformat(): 0 for d in forecast_dates}
        forecast_set = {d.isoformat() for d in forecast_dates}
        for proj_date, net in zip(projected, unassigned["net"]):
            key = proj_date.isoformat()
            if key in forecast_set:
                inflow_paise[key] += _to_paise(net)

        expected_inflow = {k: _to_rupees(v) for k, v in inflow_paise.items()}

    expected_inflow_total_paise = sum(_to_paise(v) for v in expected_inflow.values())

    # --- Cash at risk by age --------------------------------------------
    at_risk_by_age_paise = {"<3d": 0, "3-7d": 0, ">7d": 0}
    if exception_report_df is not None and len(exception_report_df):
        for _, row in exception_report_df.iterrows():
            age = int(row["age_days"])
            amount_paise = _to_paise(row["credit_amount"])
            if age < 3:
                at_risk_by_age_paise["<3d"] += amount_paise
            elif age <= 7:
                at_risk_by_age_paise["3-7d"] += amount_paise
            else:
                at_risk_by_age_paise[">7d"] += amount_paise

    # --- Confidence interval --------------------------------------------
    # low  = forecast alone, treating every at-risk credit as lost
    # high = forecast plus full recovery of everything at risk
    low_paise = expected_inflow_total_paise
    high_paise = expected_inflow_total_paise + at_risk_paise

    return {
        "verified_settled": _to_rupees(verified_settled_paise),
        "at_risk": _to_rupees(at_risk_paise),
        "expected_inflow": expected_inflow,
        "expected_inflow_total": _to_rupees(expected_inflow_total_paise),
        "confidence_interval": (_to_rupees(low_paise), _to_rupees(high_paise)),
        "cash_at_risk_by_age": {
            k: _to_rupees(v) for k, v in at_risk_by_age_paise.items()
        },
    }
