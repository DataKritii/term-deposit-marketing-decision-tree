"""
marketing_model.py
==================
Reusable model code for the Streamlit dashboard (app.py).

It reproduces the final models built in
notebooks/term_deposit_decision_tree_random_forest.ipynb:

    1. Load the Bank Marketing data, remove duplicates, drop the leaking `duration` column
    2. Recode pdays = 999 ("never contacted before") as -1
    3. Split 80/20, stratified on the target (same seed as the notebook)
    4. Fit a pruned decision tree (depth 5, at least 200 clients per leaf),
       chosen with the one-standard-error rule, for explanations
    5. Fit a random forest (300 trees, at least 20 clients per leaf), the selected model for scoring

Keeping this logic in one module means the notebook and the dashboard can't
drift apart: the automated tests check that the numbers match.
"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

# --------------------------------------------------------------------------
# Constants (identical to the notebook)
# --------------------------------------------------------------------------
RANDOM_STATE = 42
TARGET = "subscribed"
DATA_FILE = "bank-additional-full.csv"
DATA_URL = ("https://raw.githubusercontent.com/ahanagangopadhyay/improving-bank-telemarketing-campaigns/"
            "main/data/bank-additional-full.csv")

CATEGORICAL = ["job", "marital", "education", "default", "housing", "loan",
               "contact", "month", "day_of_week", "poutcome"]
NUMERIC = ["age", "campaign", "pdays", "previous",
           "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]
MACRO = ["emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]
FEATURES = CATEGORICAL + NUMERIC

LABELS = {
    "age": "Age", "job": "Job", "marital": "Marital status", "education": "Education",
    "default": "Credit in default", "housing": "Housing loan", "loan": "Personal loan",
    "contact": "Contact type", "month": "Month of contact", "day_of_week": "Day of contact",
    "campaign": "Calls in this campaign", "pdays": "Days since last campaign contact",
    "previous": "Contacts before this campaign", "poutcome": "Previous campaign outcome",
    "emp.var.rate": "Employment variation rate", "cons.price.idx": "Consumer price index",
    "cons.conf.idx": "Consumer confidence index", "euribor3m": "3-month Euribor rate (%)",
    "nr.employed": "Number employed (thousands)",
}

# Selected settings (notebook sections 7 and 8)
TREE_PARAMS = {"max_depth": 5, "min_samples_leaf": 200}                          # one-standard-error rule
FOREST_PARAMS = {"n_estimators": 300, "min_samples_leaf": 20, "max_features": "sqrt"}

# Typical economic conditions in the data (medians of the five indicators in each
# Euribor band), used as presets in the dashboard
ECONOMIC_PRESETS = {
    "High rates (Euribor ~5%, like 2008)": {"emp.var.rate": 1.4, "cons.price.idx": 93.994, "cons.conf.idx": -36.4,
                                            "euribor3m": 4.96, "nr.employed": 5228.1},
    "Falling rates (Euribor ~4%)": {"emp.var.rate": -0.1, "cons.price.idx": 93.2, "cons.conf.idx": -42.0,
                                    "euribor3m": 4.12, "nr.employed": 5195.8},
    "Low rates (Euribor ~1.3%, like 2009)": {"emp.var.rate": -1.8, "cons.price.idx": 92.893, "cons.conf.idx": -46.2,
                                             "euribor3m": 1.268, "nr.employed": 5099.1},
}


# --------------------------------------------------------------------------
# 1. Data
# --------------------------------------------------------------------------
def load_raw(data_dir: str) -> pd.DataFrame:
    """Read the semicolon-separated file from data_dir, downloading it first if it's missing."""
    path = os.path.join(data_dir, DATA_FILE)
    if not os.path.exists(path):
        os.makedirs(data_dir, exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, path)
    return pd.read_csv(path, sep=";")


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Apply the notebook's data decisions. Row order (chronological) is preserved."""
    df = raw.drop_duplicates().reset_index(drop=True)              # 12 exact duplicate rows
    df[TARGET] = (df["y"] == "yes").astype(int)
    df["pdays"] = df["pdays"].replace(999, -1)                     # 999 = never contacted before
    return df.drop(columns=["y", "duration"])                      # duration leaks the outcome


# --------------------------------------------------------------------------
# 2. Models
# --------------------------------------------------------------------------
def make_preprocessor() -> ColumnTransformer:
    """One-hot encode the categorical columns; pass numeric columns through unchanged
    (trees don't need scaling)."""
    return ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)],
                             remainder="passthrough", verbose_feature_names_out=False)


def make_tree() -> Pipeline:
    return Pipeline([("prep", make_preprocessor()),
                     ("model", DecisionTreeClassifier(random_state=RANDOM_STATE, **TREE_PARAMS))])


def make_forest() -> Pipeline:
    return Pipeline([("prep", make_preprocessor()),
                     ("model", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **FOREST_PARAMS))])


@dataclass
class MarketingBundle:
    tree: Pipeline
    forest: Pipeline
    train: pd.DataFrame
    test: pd.DataFrame            # with p_tree and p_forest columns added
    data: pd.DataFrame            # full cleaned data, chronological order

    def predict(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """(random forest probability, decision tree probability) of subscribing."""
        return (self.forest.predict_proba(X[FEATURES])[:, 1], self.tree.predict_proba(X[FEATURES])[:, 1])

    def decision_path(self, row: pd.DataFrame) -> tuple[list[str], dict]:
        """The decision tree's rules for one client, in plain words, plus its leaf's statistics."""
        prep = self.tree.named_steps["prep"]
        tree = self.tree.named_steps["model"].tree_
        x = prep.transform(row[FEATURES])
        x = x.toarray() if hasattr(x, "toarray") else np.asarray(x)
        names = prep.get_feature_names_out()
        onehot = _onehot_lookup(prep)
        node, rules = 0, []
        while tree.children_left[node] != -1:                       # until a leaf is reached
            f, thr = tree.feature[node], tree.threshold[node]
            went_left = x[0, f] <= thr
            rules.append(_describe(names[f], thr, went_left, onehot))
            node = tree.children_left[node] if went_left else tree.children_right[node]
        value = tree.value[node][0]
        rate = value[1] / value.sum() if value.sum() > 1.0001 else value[1]   # counts or fractions, by version
        return rules, {"clients": int(tree.n_node_samples[node]), "subscription_rate": float(rate)}


def _onehot_lookup(prep: ColumnTransformer) -> dict:
    """Map each one-hot column name (e.g. 'poutcome_success') to (feature, category)."""
    enc = prep.named_transformers_["cat"]
    lookup = {}
    for feat, cats in zip(CATEGORICAL, enc.categories_):
        for c in cats:
            lookup[f"{feat}_{c}"] = (feat, c)
    return lookup


def _describe(name: str, thr: float, went_left: bool, onehot: dict) -> str:
    """Turn a split like 'poutcome_success <= 0.5' into 'Previous campaign outcome is not success'."""
    if name in onehot:
        feat, cat = onehot[name]
        return f"{LABELS[feat]} is {'not ' if went_left else ''}{cat}"
    label = LABELS.get(name, name)
    if name == "pdays" and thr < 0:
        return f"{label}: {'never contacted' if went_left else 'contacted before'}"
    return f"{label} {'≤' if went_left else '>'} {thr:,.2f}"


def train_bundle(data_dir: str) -> MarketingBundle:
    """Run the notebook workflow and return the fitted tree and forest."""
    data = clean(load_raw(data_dir))
    train, test = train_test_split(data, test_size=0.2, stratify=data[TARGET], random_state=RANDOM_STATE)
    tree = make_tree().fit(train[FEATURES], train[TARGET])
    forest = make_forest().fit(train[FEATURES], train[TARGET])
    test = test.copy()
    test["p_forest"] = forest.predict_proba(test[FEATURES])[:, 1]
    test["p_tree"] = tree.predict_proba(test[FEATURES])[:, 1]
    return MarketingBundle(tree=tree, forest=forest, train=train, test=test, data=data)


# --------------------------------------------------------------------------
# 3. Campaign metrics
# --------------------------------------------------------------------------
def gains_table(y, p, n_groups: int = 10) -> pd.DataFrame:
    """Clients ranked by predicted probability and split into equal groups (1 = most likely).
    Lift = the group's subscription rate divided by the overall rate."""
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    d["group"] = pd.qcut(d["p"].rank(method="first", ascending=False), n_groups, labels=range(1, n_groups + 1))
    t = d.groupby("group", observed=True).agg(clients=("y", "size"), subscribers=("y", "sum"),
                                              avg_predicted=("p", "mean"), actual_rate=("y", "mean"))
    t["lift"] = t["actual_rate"] / d["y"].mean()
    t["cum_share_of_subscribers"] = t["subscribers"].cumsum() / t["subscribers"].sum()
    t["cum_share_called"] = t["clients"].cumsum() / t["clients"].sum()
    return t


def profit_curve(y, p, cost_per_call: float, value_per_subscription: float) -> pd.DataFrame:
    """Profit from calling the top k clients (ranked by p), for every k.
        profit(k) = value x subscribers among the top k  -  cost x k"""
    order = np.argsort(-np.asarray(p))
    y_sorted = np.asarray(y)[order]
    calls = np.arange(1, len(y_sorted) + 1)
    subs = np.cumsum(y_sorted)
    return pd.DataFrame({"calls": calls, "share_called": calls / len(calls), "subscribers": subs,
                         "profit": value_per_subscription * subs - cost_per_call * calls,
                         "p_cutoff": np.asarray(p)[order]})
