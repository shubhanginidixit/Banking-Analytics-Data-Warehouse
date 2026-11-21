"""Executive overview page."""

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from components.common import (
    ACCENT,
    RISK,
    cutoff_stats,
    gains_figure,
    grid,
    kpi_row,
    load_features,
    load_output,
    load_scores,
    money,
    note,
    page_header,
    percent,
    quintile_figure,
    ref_rule,
    score_config,
    score_distribution,
    scored_frame,
    section,
    show,
)


def portfolio_page() -> None:
    features = load_features()
    scores = load_scores()
    if scores.empty:
        st.warning("Out-of-fold scores are not available.")
        return
    defaults = int(features["target"].sum())
    base_rate = float(features["target"].mean())
    scored = scored_frame(scores, features)

    page_header(
        "Portfolio",
        f"{len(features):,} applications, {defaults} defaults, one review budget",
        "Every figure on this page is computed from data/features/loan_features.csv and the "
        "out-of-fold scores in artifacts/metrics/model_scores.csv. Move the threshold to see what "
        "a review queue would catch and what it would cost.",
    )

    kpi_row(
        [
            ("Applications", f"{len(features):,}", "One row per loan"),
            ("Defaults", f"{defaults}", "Observed outcome"),
            ("Base rate", percent(base_rate), "No model involved"),
            ("Median score", f"{scores['prob_default'].median():.3f}", "Out-of-fold probability"),
        ]
    )

    section("Review threshold", "Out-of-fold scores, so no loan is scored by a model that saw it")
    cutoff = st.slider(
        "Flag applications scoring at or above",
        min_value=0.05,
        max_value=0.90,
        value=0.40,
        step=0.01,
    )
    stats = cutoff_stats(scored, cutoff)
    kpi_row(
        [
            ("In the queue", f"{stats['flagged']}", f"{stats['flagged'] / max(stats['total'], 1):.1%} of the portfolio"),
            ("Defaults caught", f"{stats['tp']} / {stats['defaults']}", f"{stats['fn']} missed"),
            ("Precision", percent(stats["precision"]), f"{stats['fp']} false alarms"),
            ("Recall", percent(stats["recall"]), "Share of defaults found"),
        ]
    )

    show(score_distribution(scores, cutoff))
    st.caption(
        "Red segments are observed defaults; the shaded region is where applications would be flagged. "
        "The dashed line is the cutoff."
    )

    left, right = st.columns([1.15, 1], gap="large")
    with left:
        show(gains_figure(scores, cutoff))
    with right:
        st.markdown(
            f'<table class="matrix"><thead><tr><th></th><th>Flagged</th><th>Not flagged</th></tr></thead><tbody>'
            f'<tr><td>Defaulted</td><td class="num hit">{stats["tp"]}</td><td class="num miss">{stats["fn"]}</td></tr>'
            f'<tr><td>Repaid</td><td class="num">{stats["fp"]}</td><td class="num">{stats["tn"]}</td></tr>'
            f"</tbody></table>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="kpi" style="margin-top:1.1rem"><div class="kpi-label">Exposure in the queue</div>'
            f'<div class="kpi-value">{stats["exposure"] / 1e6:.1f}M CZK</div>'
            f'<div class="kpi-note">Sum of loan amounts flagged</div></div>',
            unsafe_allow_html=True,
        )
        note(
            f"At {cutoff:.2f} the queue holds {stats['flagged']} applications: {stats['tp']} of "
            f"{stats['defaults']} defaults and {stats['fp']} that would have repaid.",
            label="At this setting",
        )

    trade_rows = []
    for value in np.arange(0.05, 0.91, 0.01):
        entry = cutoff_stats(scored, float(value))
        for metric_name in ("precision", "recall"):
            trade_rows.append(
                {
                    "Cutoff": float(value),
                    "Metric": metric_name,
                    "Rate": entry[metric_name],
                    "In queue": entry["flagged"],
                    "Caught": f"{entry['tp']} / {entry['defaults']}",
                }
            )
    trade = pd.DataFrame(trade_rows)
    lines = (
        alt.Chart(trade)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("Cutoff:Q", title="Review cutoff", axis=alt.Axis(format="%")),
            y=alt.Y("Rate:Q", title="Precision or recall", axis=alt.Axis(format="%")),
            color=alt.Color(
                "Metric:N",
                scale=alt.Scale(domain=["precision", "recall"], range=[ACCENT, RISK]),
                title=None,
            ),
            tooltip=[
                "Metric",
                alt.Tooltip("Cutoff:Q", format=".0%"),
                alt.Tooltip("Rate:Q", format=".1%"),
                "In queue",
                "Caught",
            ],
        )
    )
    show((lines + ref_rule(cutoff)).properties(height=260).interactive())
    st.caption(
        "Every cutoff from 5% to 90%. The dashed line is the current setting. "
        "Higher cutoffs buy precision with recall."
    )

    section("Where the risk sits", f"Default rate by quintile, all {len(features):,} loans")
    left, right = st.columns(2, gap="large")
    with left:
        show(quintile_figure(features, "balance_at_loan", "Balance at loan"))
        st.caption(f"Q1 is everything below {money(features['balance_at_loan'].quantile(0.2))}.")
    with right:
        show(quintile_figure(features, "min_balance", "Minimum balance"))
        st.caption(f"Q1 is everything below {money(features['min_balance'].quantile(0.2))}.")

    note(
        "The bottom balance quintile defaults at "
        f"{features.loc[features['balance_at_loan'] <= features['balance_at_loan'].quantile(0.2), 'target'].mean():.1%}, "
        "against a portfolio rate of "
        f"{base_rate:.1%}. Balance capacity separates outcomes; transaction volume separates them far less.",
        label="What the split shows",
    )

    section("Behaviour segments", "KMeans k=2 on pre-loan behaviour, from artifacts/segments/segment_profiles.csv")
    profiles = load_output("segment_profiles.csv")
    if profiles is None or profiles.empty:
        st.warning("Segment profiles are not available.")
        return
    choice = st.selectbox(
        "Segment",
        options=profiles["segment"].tolist(),
        format_func=lambda value: f"Segment {value}",
    )
    profile_rows = profiles[profiles["segment"] == choice]
    if profile_rows.empty:
        st.warning("That segment has no profile row.")
        return
    profile = profile_rows.iloc[0]
    kpi_row(
        [
            ("Loans", f"{int(profile['size'])}", "Members of this segment"),
            ("Default rate", percent(profile["default_rate"]), "Observed"),
            ("Vs base rate", f"{profile['default_rate'] - base_rate:+.1%}", "Percentage points"),
            ("Mean transactions", f"{profile['n_txn']:.0f}", "Per loan"),
        ]
    )
    memberships = load_output("loan_segments.csv")
    if memberships is None or memberships.empty:
        st.warning("Loan segment memberships are not available.")
        return
    member_ids = memberships.query("segment == @choice")["loan_id"]
    members = features[features["loan_id"].isin(member_ids)].merge(
        scores[["loan_id", "prob_default"]], on="loan_id", how="left"
    )
    if members.empty:
        st.warning("This segment has no member loans.")
        return
    points = members.assign(Outcome=members["target"].map({0: "Repaid", 1: "Default"}))
    show(
        (
            ref_rule(cutoff, axis="y")
            + alt.Chart(points)
            .mark_circle(opacity=0.75)
            .encode(
                x=alt.X("balance_at_loan:Q", title="Balance at loan"),
                y=alt.Y("prob_default:Q", axis=alt.Axis(format="%"), title="Score"),
                size=alt.Size("amount:Q", title="Loan amount", scale=alt.Scale(range=[30, 400])),
                color=alt.Color(
                    "Outcome:N",
                    scale=alt.Scale(domain=["Repaid", "Default"], range=[ACCENT, RISK]),
                    title=None,
                ),
                tooltip=[
                    "loan_id",
                    alt.Tooltip("prob_default:Q", format=".3f"),
                    alt.Tooltip("amount:Q", format=","),
                    "Outcome",
                ],
            )
        )
        .properties(height=300)
        .interactive()
    )
    with st.expander("Member loans"):
        grid(
            members.sort_values("prob_default", ascending=False)
            .head(12)[["loan_id", "amount", "duration", "balance_at_loan", "min_balance", "txn_per_month", "prob_default", "target"]]
            .rename(
                columns={
                    "loan_id": "Loan",
                    "amount": "Amount",
                    "duration": "Months",
                    "balance_at_loan": "Balance",
                    "min_balance": "Min balance",
                    "txn_per_month": "Txn / month",
                    "prob_default": "Score",
                    "target": "Defaulted",
                }
            ),
            score_config("Amount", "Balance", "Min balance", places=0)
            | score_config("Txn / month", places=1)
            | score_config("Score"),
        )
    note(
        f"The {len(profiles)} segments differ by "
        f"{abs(profiles['default_rate'].max() - profiles['default_rate'].min()):.1%} in default rate. "
        "That is weak separation on this sample, and segment membership is not a model feature. "
        "Read it as a description of the book, not as a risk rule.",
        label="Caveat",
        kind="risk",
    )


# ------------------------------------------------------------- data & leakage
