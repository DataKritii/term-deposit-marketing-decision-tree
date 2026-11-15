"""
Term Deposit Campaign Targeting Dashboard
=========================================
An interactive Streamlit dashboard on top of the decision tree and random forest
built in notebooks/term_deposit_decision_tree_random_forest.ipynb.

Run it locally with:
    streamlit run app.py

Three tabs:
    1. Score a client     : a client's chance of subscribing (random forest), the decision
                            tree's rules for that client, and how the economy changes it
    2. Campaign planner   : how many clients to call, given a cost per call and a value per
                            subscription; gains, lift and profit
    3. Model health       : model comparison, calibration, feature importance, and the
                            out-of-time test (the model ownership view)

Streamlit re-runs this script top to bottom every time an input changes. Slow steps
(training, permutation importance, the out-of-time test) are cached so they run only once.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

from src.marketing_model import (  # noqa: E402
    CATEGORICAL, ECONOMIC_PRESETS, FEATURES, LABELS, NUMERIC, RANDOM_STATE, TARGET,
    gains_table, make_forest, make_tree, profit_curve, train_bundle,
)

st.set_page_config(page_title="Term Deposit Campaign Targeting", page_icon="📞", layout="wide")
NAVY, AMBER, GREEN, RED, GREY = "#2C3E50", "#E8A33D", "#2E8B57", "#C0392B", "#95A5A6"


# --------------------------------------------------------------------------
# Cached work (runs once per server start)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Training the models (first load only)...")
def load_models():
    return train_bundle(str(APP_DIR / "data"))


def make_logit():
    prep = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
                              ("num", StandardScaler(), NUMERIC)])
    return Pipeline([("prep", prep), ("model", LogisticRegression(max_iter=3000))])


@st.cache_resource(show_spinner="Running the model comparison and out-of-time test (first load only)...")
def load_health(_b):
    """Test-set comparison, out-of-time test and permutation importance (notebook sections 9, 12, 14)."""
    t = _b.test
    logit = make_logit().fit(_b.train[FEATURES], _b.train[TARGET])
    p_logit = logit.predict_proba(t[FEATURES])[:, 1]
    comp = pd.DataFrame({
        "Model": ["Random forest (selected)", "Decision tree (pruned, 22 leaves)", "Logistic regression (reference)"],
        "Test AUC": [roc_auc_score(t[TARGET], t["p_forest"]), roc_auc_score(t[TARGET], t["p_tree"]), roc_auc_score(t[TARGET], p_logit)],
        "Test average precision": [average_precision_score(t[TARGET], t["p_forest"]),
                                   average_precision_score(t[TARGET], t["p_tree"]),
                                   average_precision_score(t[TARGET], p_logit)],
    })
    # Out-of-time: train on the first 80% of contacts (chronological), test on the last 20%
    data = _b.data
    cut = int(len(data) * 0.8)
    early, late = data.iloc[:cut], data.iloc[cut:]
    oot = []
    for name, m in [("Random forest (selected)", make_forest()), ("Decision tree (pruned, 22 leaves)", make_tree()),
                    ("Logistic regression (reference)", make_logit())]:
        p = m.fit(early[FEATURES], early[TARGET]).predict_proba(late[FEATURES])[:, 1]
        oot.append({"Model": name, "Out-of-time AUC": roc_auc_score(late[TARGET], p), "Average predicted": p.mean()})
    oot = pd.DataFrame(oot)
    comp = comp.merge(oot, on="Model")
    perm = permutation_importance(_b.forest, t[FEATURES], t[TARGET], scoring="roc_auc", n_repeats=3,
                                  random_state=RANDOM_STATE, n_jobs=-1)
    importance = pd.Series(perm.importances_mean, index=[LABELS[f] for f in FEATURES]).sort_values()
    periods = {"early_rate": early[TARGET].mean(), "late_rate": late[TARGET].mean(),
               "early_euribor": (early["euribor3m"].min(), early["euribor3m"].max()),
               "late_euribor": (late["euribor3m"].min(), late["euribor3m"].max())}
    return comp, importance, periods, data[["euribor3m", TARGET]].reset_index(drop=True)


b = load_models()
test = b.test
y_test = test[TARGET].values
p_forest = test["p_forest"].values
p_tree = test["p_tree"].values


def pct(x, d=1):
    return f"{x:.{d}%}"


def eur(x):
    return f"€{x:,.0f}"


def eur_signed(x):
    """Euros with cents and a proper minus sign: -0.5 -> −€0.50."""
    return f"{'−' if x < 0 else ''}€{abs(x):,.2f}"


# --------------------------------------------------------------------------
# Sidebar: campaign economics (used in tabs 1 and 2)
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("Campaign economics")
    cost = st.number_input("Cost per call (€)", min_value=0.5, max_value=100.0, value=5.0, step=0.5,
                           help="Agent time per client, including unanswered attempts.")
    value = st.number_input("Value per subscription (€)", min_value=1.0, max_value=1000.0, value=50.0, step=5.0,
                            help="Net margin the bank earns on a new term deposit.")
    breakeven = cost / value
    st.markdown(f"**Break-even probability:** call if p > {eur(cost)} ÷ {eur(value)} = **{pct(breakeven)}**")
    st.caption("Illustrative figures; real ones would come from finance.")
    st.divider()
    st.subheader("About")
    st.markdown("Predicts which bank clients will subscribe to a term deposit after a phone call. "
                "Trained on 41,176 calls by a Portuguese bank, May 2008 to November 2010.\n\n"
                "Call length (`duration`) is **excluded**: it's only known after the call, so using it would leak the outcome.")

st.title("📞 Term Deposit Campaign Targeting")
st.caption("Decision tree and random forest models that rank clients by their chance of subscribing, "
           "so a phone campaign reaches more subscribers with fewer calls.")

tab_score, tab_plan, tab_health = st.tabs(["👤 Score a client", "📈 Campaign planner", "🩺 Model health"])


# ==========================================================================
# TAB 1: SCORE A CLIENT
# ==========================================================================
with tab_score:
    st.subheader("How likely is this client to subscribe?")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Client**")
        age = st.slider("Age", 18, 95, 38)
        job = st.selectbox("Job", ["admin.", "blue-collar", "technician", "services", "management", "retired",
                                   "entrepreneur", "self-employed", "housemaid", "unemployed", "student", "unknown"])
        marital = st.selectbox("Marital status", ["married", "single", "divorced", "unknown"])
        education = st.selectbox("Education", ["university.degree", "high.school", "basic.9y", "professional.course",
                                               "basic.4y", "basic.6y", "illiterate", "unknown"])
        default = st.selectbox("Credit in default?", ["no", "unknown", "yes"])
        housing = st.selectbox("Housing loan?", ["yes", "no", "unknown"])
        loan = st.selectbox("Personal loan?", ["no", "yes", "unknown"])
    with c2:
        st.markdown("**This campaign**")
        contact = st.radio("Contact by", ["cellular", "telephone"], horizontal=True,
                           format_func=lambda c: "Mobile" if c == "cellular" else "Landline")
        month = st.selectbox("Month", ["may", "jun", "jul", "aug", "oct", "nov", "dec", "mar", "apr", "sep"],
                             format_func=str.capitalize)
        day = st.selectbox("Day", ["mon", "tue", "wed", "thu", "fri"], format_func=str.capitalize)
        campaign = st.slider("Calls to this client in this campaign (including this one)", 1, 15, 1)
        st.markdown("**Earlier campaigns**")
        contacted_before = st.radio("Contacted in an earlier campaign?", ["No", "Yes"], horizontal=True)
        if contacted_before == "Yes":
            pdays = st.slider("Days since last contact", 0, 27, 6)
            previous = st.slider("Number of earlier contacts", 1, 7, 1)
            poutcome = st.radio("Outcome of the earlier campaign", ["success", "failure"], horizontal=True,
                                format_func=str.capitalize)
        else:
            pdays, previous, poutcome = -1, 0, "nonexistent"      # the dataset's coding for "never contacted"
    with c3:
        st.markdown("**Economy at the time of the call**")
        preset = st.radio("Conditions", list(ECONOMIC_PRESETS), index=0,
                          help="Typical values of the five economic indicators in the data.")
        econ = ECONOMIC_PRESETS[preset]
        st.dataframe(pd.DataFrame({"Indicator": [LABELS[k] for k in econ], "Value": list(econ.values())}),
                     hide_index=True, width="stretch")

    row = pd.DataFrame([{"age": age, "job": job, "marital": marital, "education": education, "default": default,
                         "housing": housing, "loan": loan, "contact": contact, "month": month, "day_of_week": day,
                         "campaign": campaign, "pdays": pdays, "previous": previous, "poutcome": poutcome, **econ}])
    pf, pt = (float(v[0]) for v in b.predict(row))
    rank = (p_forest > pf).mean()      # share of test clients scored higher

    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Chance of subscribing (random forest)", pct(pf),
              help=f"Average across all clients: {pct(y_test.mean())}")
    m2.metric("Rank among all clients", f"Top {max(rank, 0.001):.0%}" if rank < 0.5 else f"Bottom {1 - rank:.0%}",
              help="Compared with the 8,236 clients in the test set.")
    m3.metric("Expected value of the call", eur_signed(pf * value - cost), help=f"{pct(pf)} × {eur(value)} − {eur(cost)}")
    if pf > breakeven:
        st.success(f"**Call this client.** {pct(pf)} is above the break-even probability of {pct(breakeven)}.")
    else:
        st.warning(f"**Lower priority.** {pct(pf)} is below the break-even probability of {pct(breakeven)}: "
                   "on average, the call costs more than it earns.")

    left, right = st.columns([3, 2], gap="large")
    with left:
        rules, leaf = b.decision_path(row)
        st.markdown("**Why: the decision tree's rules for this client**")
        st.markdown("\n".join(f"{i}. {r}" for i, r in enumerate(rules, 1)))
        st.markdown(f"➜ In training, **{leaf['clients']:,} clients** followed this exact path, and "
                    f"**{pct(leaf['subscription_rate'])}** of them subscribed.")
        st.caption("The 22-leaf decision tree is readable but slightly less accurate than the random forest, "
                   "so the two probabilities can differ. The forest's figure is used for decisions.")
    with right:
        # The same client under each economic preset
        probs = []
        for name, values in ECONOMIC_PRESETS.items():
            r = row.copy()
            for k, v in values.items():
                r[k] = v
            probs.append(float(b.predict(r)[0][0]))
        fig = go.Figure(go.Bar(x=probs, y=[n.split(" (")[0] for n in ECONOMIC_PRESETS], orientation="h",
                               marker_color=[AMBER if n == preset else NAVY for n in ECONOMIC_PRESETS],
                               text=[pct(p) for p in probs], textposition="outside"))
        fig.add_vline(x=breakeven, line=dict(color=RED, dash="dash"), annotation_text="Break-even",
                      annotation_position="bottom right")
        fig.update_layout(title="Same client, different economy", xaxis_tickformat=".0%", height=260,
                          margin=dict(t=40, b=10, l=10, r=40), xaxis_range=[0, max(probs + [breakeven]) * 1.3])
        st.plotly_chart(fig, width="stretch")
        st.caption("The economy shifts every client's chance of subscribing: in the data, the subscription rate "
                   "was about 5% when rates were high and about 24% when they were low.")


# ==========================================================================
# TAB 2: CAMPAIGN PLANNER
# ==========================================================================
with tab_plan:
    st.subheader("How many clients should we call?")
    st.markdown(f"Based on the **{len(test):,} test clients** the models never saw, called in order of the "
                "random forest's probability (most likely first).")

    curve = profit_curve(y_test, p_forest, cost, value)
    best = curve.loc[curve["profit"].idxmax()]
    share = st.slider("Share of clients to call", 1, 100, int(round(best["share_called"] * 100)), format="%d%%",
                      help="Starts at the most profitable share for your cost and value figures.") / 100
    k = max(1, int(round(share * len(curve)))) - 1
    pick = curve.iloc[k]
    everyone = curve.iloc[-1]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Calls", f"{int(pick['calls']):,}")
    k2.metric("Subscribers reached", f"{int(pick['subscribers']):,}",
              help=f"{pct(pick['subscribers'] / everyone['subscribers'])} of all subscribers in the test set")
    k3.metric("Conversion rate", pct(pick["subscribers"] / pick["calls"]), help=f"Calling everyone: {pct(y_test.mean())}")
    k4.metric("Campaign profit", eur(pick["profit"]), delta=f"{eur(pick['profit'] - everyone['profit'])} vs calling everyone")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        fig = go.Figure()
        fig.add_scatter(x=curve["share_called"], y=curve["profit"], mode="lines", line=dict(color=NAVY, width=3), name="Profit")
        fig.add_scatter(x=[pick["share_called"]], y=[pick["profit"]], mode="markers",
                        marker=dict(size=13, color=AMBER), name="Your plan")
        fig.add_vline(x=best["share_called"], line=dict(color=GREEN, dash="dash"),
                      annotation_text=f"Most profitable: {pct(best['share_called'], 0)}")
        fig.update_layout(title=f"Profit by share called ({eur(cost)} per call, {eur(value)} per subscription)",
                          xaxis_title="Share of clients called", yaxis_title="Profit (€)", xaxis_tickformat=".0%",
                          height=380, margin=dict(t=50), legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig, width="stretch")
    with c2:
        fig = go.Figure()
        for name, p, color in [("Random forest", p_forest, NAVY), ("Decision tree", p_tree, AMBER)]:
            order = np.argsort(-p)
            fig.add_scatter(x=np.arange(1, len(p) + 1) / len(p), y=np.cumsum(y_test[order]) / y_test.sum(),
                            mode="lines", line=dict(color=color, width=3), name=name)
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=GREY, dash="dash"), name="Random")
        fig.add_vline(x=share, line=dict(color=AMBER, dash="dot"))
        fig.update_layout(title="Gains: share of subscribers reached", xaxis_title="Share of clients called",
                          yaxis_title="Subscribers reached", xaxis_tickformat=".0%", yaxis_tickformat=".0%",
                          height=380, margin=dict(t=50), legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig, width="stretch")

    st.markdown("**Lift by group** (clients split into 10 equal groups, most likely first)")
    g = gains_table(y_test, p_forest).reset_index()
    g = g.rename(columns={"group": "Group", "clients": "Clients", "subscribers": "Subscribers",
                          "avg_predicted": "Avg predicted", "actual_rate": "Actual rate", "lift": "Lift",
                          "cum_share_of_subscribers": "Cumulative share of subscribers"}).drop(columns="cum_share_called")
    st.dataframe(g.style.format({"Avg predicted": "{:.1%}", "Actual rate": "{:.1%}", "Lift": "{:.1f}×",
                                 "Cumulative share of subscribers": "{:.0%}"}), hide_index=True, width="stretch")


# ==========================================================================
# TAB 3: MODEL HEALTH
# ==========================================================================
with tab_health:
    st.subheader("How good are the models, and will they hold up?")
    comp, importance, periods, history = load_health(b)

    st.markdown("**Model comparison**: random-split test set vs out-of-time test (train on early months, test on later ones)")
    st.dataframe(comp.style.format({"Test AUC": "{:.3f}", "Test average precision": "{:.3f}",
                                    "Out-of-time AUC": "{:.3f}", "Average predicted": "{:.1%}"}),
                 hide_index=True, width="stretch")
    st.caption(f"Average precision of a random ranking = the subscription rate, {y_test.mean():.3f}. "
               f"In the out-of-time test, the actual subscription rate was {pct(periods['late_rate'])}.")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("**Why the out-of-time test is so much harder**")
        st.markdown(f"""
- Early period: subscription rate **{pct(periods['early_rate'])}**, Euribor rate {periods['early_euribor'][0]:.2f}%–{periods['early_euribor'][1]:.2f}%
- Late period: subscription rate **{pct(periods['late_rate'])}**, Euribor rate {periods['late_euribor'][0]:.2f}%–{periods['late_euribor'][1]:.2f}%
- The late period's rates are at or below the **lowest** value in training. Trees can't extrapolate beyond their training range, so the
  single tree collapses; the forest suffers less; logistic regression, which can extend a straight line, holds up best.
- **Lesson:** a random split overstates post-deployment performance when conditions drift. Validate out of time, monitor input ranges,
  and retrain and recalibrate regularly.
""")
    with c2:
        roll = history.assign(contact=np.arange(len(history)))
        roll["rate"] = roll[TARGET].rolling(1000, min_periods=200).mean()
        fig = go.Figure()
        # Two y-axes: Euribor (0-5%) on the left, subscription rate (0-55%) on the right
        fig.add_scatter(x=roll["contact"], y=roll["euribor3m"], name="Euribor rate (%, left axis)", line=dict(color=NAVY, width=2))
        fig.add_scatter(x=roll["contact"], y=roll["rate"], name="Subscription rate (rolling 1,000 contacts, right axis)",
                        line=dict(color=AMBER, width=2), yaxis="y2")
        fig.add_vline(x=int(len(history) * 0.8), line=dict(color=RED, dash="dash"),
                      annotation_text="Out-of-time test starts", annotation_position="top left")
        fig.update_layout(title="Contacts in date order, May 2008 → Nov 2010", xaxis_title="Contact number",
                          yaxis=dict(title="Euribor rate (%)", rangemode="tozero"),
                          yaxis2=dict(title="Subscription rate", overlaying="y", side="right", tickformat=".0%",
                                      rangemode="tozero", showgrid=False),
                          height=380, margin=dict(t=50), legend=dict(orientation="h", y=-0.3))
        st.plotly_chart(fig, width="stretch")

    c3, c4 = st.columns(2, gap="large")
    with c3:
        g = gains_table(y_test, p_forest).reset_index()
        fig = go.Figure()
        fig.add_bar(x=g["group"], y=g["avg_predicted"], name="Average predicted", marker_color=NAVY)
        fig.add_bar(x=g["group"], y=g["actual_rate"], name="Actual rate", marker_color=AMBER)
        fig.update_layout(title="Calibration of the random forest (test set)", barmode="group", yaxis_tickformat=".0%",
                          xaxis_title="Group (1 = most likely 10%)", height=380, margin=dict(t=50),
                          legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig, width="stretch")
    with c4:
        top = importance.tail(10)
        fig = go.Figure(go.Bar(x=top.values, y=top.index, orientation="h", marker_color=NAVY))
        fig.update_layout(title="Permutation importance: AUC drop when a feature is shuffled", height=380,
                          margin=dict(t=50), xaxis_title="Drop in test AUC")
        st.plotly_chart(fig, width="stretch")

    st.markdown("**The pruned decision tree** (first 3 of 5 levels; 'value' = [share no, share yes])")
    st.image(str(APP_DIR / "reports" / "figures" / "02_decision_tree.png"), width="stretch")

    with st.expander("Limitations and how this model would be owned in production"):
        st.markdown("""
**Limitations**
- **One bank, Portugal, 2008–2010**, including the financial crisis. The method transfers; the numbers don't.
- **The economy dominates:** it sets the baseline for everyone more than it separates clients called on the same day.
- **Compressed probabilities:** fine for ranking; calibrate before using them as forecasts.
- **Illustrative economics:** real cost and value figures would come from finance.
- **Selection bias:** the bank chose whom to call, so the data only covers clients it decided to contact.

**Model ownership**
| Area | Practice |
|---|---|
| Documentation | Model card: purpose, data, the `duration` exclusion, settings and how they were chosen, out-of-time results, limitations |
| Validation | Out-of-time test on the latest months before every release |
| Monitoring | Each campaign: conversion by score group; input ranges, especially the economic indicators |
| Triggers | Inputs outside the training range, falling lift in the top group, or a base-rate shift → retrain and recalibrate |
| Fairness | Age is an input: confirm targeting by age is permitted for the product, and review outcomes by age band |
| Governance | Version control, change log, sign-off by marketing and model risk |
""")
