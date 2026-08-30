"""Headless render checks for the Streamlit dashboard pages.

Each page is executed via Streamlit's AppTest harness against a small
generated dataset. This catches import errors, bad column references, and
API misuse that a plain `python -c "import"` would miss, since the page
bodies only execute inside a script run.
"""

import os
import sys

import pytest

from generator.config import GeneratorConfig, Difficulty
from generator.truth import build_truth
from generator.views import to_orders, to_settlements, to_bank, truth_to_rupees

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO = os.path.join(ROOT, "demo")

PAGES = [
    "home.py",
    "reconciliation.py",
    "analytics.py",
    "exceptions.py",
    "cashflow.py",
    "qa.py",
]


@pytest.fixture(scope="module")
def small_data_dir(tmp_path_factory):
    """A small ledger so the pages run fast but exercise real code paths."""
    out = tmp_path_factory.mktemp("dashboard_data")
    config = GeneratorConfig(
        seed=42,
        num_payments=300,
        difficulty=Difficulty.MEDIUM,
        output_dir=str(out),
    )
    truth = build_truth(config)

    to_orders(truth).to_csv(out / "orders.csv", index=False, float_format="%.2f")
    to_settlements(truth).to_csv(out / "settlements.csv", index=False, float_format="%.2f")
    to_bank(truth).to_csv(out / "bank.csv", index=False, float_format="%.2f")
    truth_to_rupees(truth).to_csv(out / "ground_truth.csv", index=False, float_format="%.2f")

    return str(out)


@pytest.fixture(autouse=True)
def _demo_on_path():
    # Pages do `from shared import ...`; app.py normally puts demo/ on the
    # path, but AppTest runs a page module directly.
    added = DEMO not in sys.path
    if added:
        sys.path.insert(0, DEMO)
    yield
    if added and DEMO in sys.path:
        sys.path.remove(DEMO)


def test_exceptions_page_handles_duplicate_utrs(tmp_path, small_data_dir):
    """The `duplicate_utr` break lets two bank rows share a UTR. Widget
    keys derived from the UTR alone collide and Streamlit refuses to
    render, so the queue must key on row position too."""
    import shutil

    import pandas as pd

    for name in ("orders.csv", "settlements.csv", "bank.csv", "ground_truth.csv"):
        shutil.copy(os.path.join(small_data_dir, name), tmp_path / name)

    # Force the condition directly rather than relying on the generator's
    # break sampling. Both added rows share a UTR and carry amounts no
    # subset can reach, so neither clears and both reach the queue —
    # a UTR that *clears* would be filtered out and never collide.
    bank = pd.read_csv(tmp_path / "bank.csv")
    dup = pd.DataFrame([
        {
            "value_date": bank.iloc[-1]["value_date"],
            "narration": "NEFT-UTR_DUP-DUPLICATE BATCH",
            "credit": 987654321.11,
            "utr": "UTR_DUP",
        },
        {
            "value_date": bank.iloc[-1]["value_date"],
            "narration": "NEFT-UTR_DUP-DUPLICATE BATCH",
            "credit": 987654321.22,
            "utr": "UTR_DUP",
        },
    ])
    bank = pd.concat([bank, dup], ignore_index=True)
    bank.to_csv(tmp_path / "bank.csv", index=False, float_format="%.2f")

    assert (bank["utr"] == "UTR_DUP").sum() == 2, "fixture lost its duplicate UTRs"

    at = AppTest.from_file(os.path.join(DEMO, "app_pages", "exceptions.py"), default_timeout=300)
    at.session_state["data_dir"] = str(tmp_path)
    at.session_state["max_tier"] = 4
    at.session_state["min_conf"] = 0.90
    at.session_state["investigated_utrs"] = set()

    at.run()

    assert not at.exception, (
        "exceptions page raised: " + "; ".join(e.value for e in at.exception)
    )


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_exception(page, small_data_dir):
    at = AppTest.from_file(os.path.join(DEMO, "app_pages", page), default_timeout=300)
    at.session_state["data_dir"] = small_data_dir
    at.session_state["max_tier"] = 4
    at.session_state["min_conf"] = 0.90
    at.session_state["investigated_utrs"] = set()

    at.run()

    assert not at.exception, (
        f"{page} raised: " + "; ".join(e.value for e in at.exception)
    )
    # Some pages use st.title(), others a custom st.html() hero banner —
    # either way the page must have rendered *something* substantial.
    rendered_something = len(at.title) >= 1 or len(at.main.children) > 3
    assert rendered_something, f"{page} rendered no meaningful content"
