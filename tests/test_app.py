"""
Automated checks for the model module and the Streamlit app.

Run from the project folder with:
    pytest tests/
"""

import sys
from pathlib import Path

import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.marketing_model import FEATURES, TARGET, gains_table, profit_curve, train_bundle  # noqa: E402

APP = str(ROOT / "app.py")


@pytest.fixture(scope="module")
def bundle():
    return train_bundle(str(ROOT / "data"))


# ---------------------------------------------------------------- model ----
def test_matches_notebook_results(bundle):
    """The dashboard models must reproduce the notebook's test-set numbers."""
    t = bundle.test
    assert len(bundle.data) == 41176 and len(t) == 8236
    assert roc_auc_score(t[TARGET], t["p_forest"]) == pytest.approx(0.8091, abs=1e-3)
    assert average_precision_score(t[TARGET], t["p_forest"]) == pytest.approx(0.4845, abs=2e-3)
    assert roc_auc_score(t[TARGET], t["p_tree"]) == pytest.approx(0.7966, abs=1e-4)
    assert bundle.tree.named_steps["model"].get_n_leaves() == 22


def test_no_leakage_column():
    assert "duration" not in FEATURES


def test_decision_path_matches_tree(bundle):
    """The leaf reached by the plain-language path must give the tree's own probability."""
    for i in range(25):
        row = bundle.test.iloc[[i]]
        rules, leaf = bundle.decision_path(row)
        assert 1 <= len(rules) <= 5
        assert leaf["subscription_rate"] == pytest.approx(bundle.test["p_tree"].iloc[i], abs=1e-9)


def test_campaign_metrics(bundle):
    t = bundle.test
    g = gains_table(t[TARGET], t["p_forest"])
    assert g["lift"].iloc[0] > 4 and g["cum_share_of_subscribers"].iloc[-1] == pytest.approx(1.0)
    curve = profit_curve(t[TARGET], t["p_forest"], 5, 50)
    assert curve["profit"].max() > 4 * curve["profit"].iloc[-1]


# ------------------------------------------------------------------ app ----
def run_app():
    at = AppTest.from_file(APP, default_timeout=600)
    at.run()
    return at


def metric(at, label):
    return next(m.value for m in at.metric if m.label == label)


def test_app_loads_without_errors():
    at = run_app()
    assert not at.exception
    assert len(at.tabs) == 3
    assert any(m.label.startswith("Chance of subscribing") for m in at.metric)


def test_previous_success_raises_probability():
    at = run_app()
    label = "Chance of subscribing (random forest)"
    before = float(metric(at, label).rstrip("%"))
    radio = next(r for r in at.radio if r.label == "Contacted in an earlier campaign?")
    radio.set_value("Yes").run()
    outcome = next(r for r in at.radio if r.label == "Outcome of the earlier campaign")
    outcome.set_value("success").run()
    assert not at.exception
    assert float(metric(at, label).rstrip("%")) > before


def test_all_presets_and_extreme_economics():
    at = run_app()
    preset = next(r for r in at.radio if r.label == "Conditions")
    for option in preset.options:
        preset.set_value(option).run()
        assert not at.exception, option
    at.sidebar.number_input[0].set_value(100.0).run()        # very expensive calls
    assert not at.exception
    share = next(s for s in at.slider if s.label == "Share of clients to call")
    share.set_value(100).run()
    assert not at.exception
