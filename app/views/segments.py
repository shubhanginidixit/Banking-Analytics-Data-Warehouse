"""Unsupervised segmentation page."""

import altair as alt
import pandas as pd
import streamlit as st

from components.common import (
    ACCENT,
    CLUSTER_FEATURES,
    GOOD,
    INK,
    METHOD_NAMES,
    METHOD_ORDER,
    RISK,
    cluster_map_figures,
    cluster_rate_figure,
    grid,
    kpi_row,
    load_features,
    load_output,
    note,
    page_header,
    percent,
    score_config,
    section,
    show,
)

FEATURE_DETAILS = [
    ("n_txn", "Transactions before the loan", "Activity volume"),
    ("n_deposit", "Deposits before the loan", "Inflow frequency"),
    ("n_withdrawal", "Withdrawals before the loan", "Outflow frequency"),
    ("sum_deposit", "Total deposits before the loan", "Inflow size"),
    ("sum_withdrawal", "Total withdrawals before the loan", "Outflow size"),
    ("avg_amount", "Average transaction amount", "Typical ticket size"),
    ("std_amount", "Standard deviation of transaction amount", "Spending variability"),
    ("avg_balance", "Average account balance", "Buffer"),
    ("min_balance", "Minimum account balance", "Stress"),
    ("max_balance", "Maximum account balance", "Peak buffer"),
    ("balance_at_loan", "Balance on the loan date", "Position at application"),
    ("months_active", "Months of account history", "Relationship length"),
    ("txn_per_month", "Average transactions per active month", "Intensity"),
]

ALGORITHM_NOTES = [
    (
        "K-Means",
        "Places k centroids, assigns each loan to the nearest centroid, then moves the centroids to the cluster mean until stable.",
        "Fast, stable, full coverage, supports scoring new loans.",
        "Assumes roughly spherical clusters and needs k chosen up front.",
    ),
    (
        "K-Medoids",
        "Picks k real loans as medoids and assigns every loan to the nearest medoid. Five random starts keep the lowest inertia.",
        "Robust to outliers because medoids are real points.",
        "Initialisation-sensitive; the 10-run stability is only moderate.",
    ),
    (
        "AGNES (Ward)",
        "Starts with every loan as its own cluster and repeatedly merges the pair that increases within-cluster variance the least.",
        "Gives a hierarchy and does not assume a centroid shape.",
        "No direct predict method; the final cut still needs a choice.",
    ),
    (
        "DBSCAN",
        "Groups loans that have at least min_samples neighbours within eps. Points outside dense regions become noise.",
        "Finds arbitrary shapes and isolates a high-risk noise group.",
        "One dense core here; internal metrics need at least two clusters.",
    ),
]


def _method_order(frame: pd.DataFrame) -> list[str]:
    present = set(frame["method"]) if "method" in frame else set()
    return [method for method in METHOD_ORDER if method in present]


def _metric_table(comparison: pd.DataFrame) -> pd.DataFrame:
    view = comparison.copy()
    view["Method"] = view["method"].map(METHOD_NAMES).fillna(view["method"])
    view["Clusters"] = view["n_clusters"].astype("Int64")
    view["Silhouette"] = view["silhouette"]
    view["Davies-Bouldin"] = view["davies_bouldin"]
    view["Calinski-Harabasz"] = view["calinski_harabasz"]
    view["Coverage %"] = (view["coverage"] * 100).round(1)
    view["Noise %"] = (view["noise_share"] * 100).round(1)
    view["Stability ARI"] = view["stability_ari_mean"]
    view["Role"] = view["role"]
    return view[
        [
            "Method",
            "parameter",
            "Clusters",
            "Silhouette",
            "Davies-Bouldin",
            "Calinski-Harabasz",
            "Coverage %",
            "Noise %",
            "Stability ARI",
            "Role",
        ]
    ].rename(columns={"parameter": "Parameter"})


def _metric_chart(comparison: pd.DataFrame, column: str, title: str, higher: bool):
    frame = comparison.dropna(subset=[column]).copy()
    if frame.empty:
        return None
    frame["Method"] = frame["method"].map(METHOD_NAMES).fillna(frame["method"])
    order = [METHOD_NAMES[method] for method in _method_order(frame)]
    color = alt.Color(
        "Method:N",
        scale=alt.Scale(
            domain=order,
            range=[ACCENT, GOOD, "#8a6d3b", RISK][: len(order)],
        ),
        legend=None,
    )
    bars = (
        alt.Chart(frame)
        .mark_bar(size=26)
        .encode(
            y=alt.Y("Method:N", sort=order, title=None),
            x=alt.X(
                f"{column}:Q",
                title=title,
                scale=alt.Scale(zero=not higher),
            ),
            color=color,
            tooltip=[
                "Method",
                alt.Tooltip(f"{column}:Q", format=".3f"),
            ],
        )
    )
    labels = (
        alt.Chart(frame)
        .mark_text(align="left", dx=4, fontSize=9, color=INK)
        .encode(
            y=alt.Y("Method:N", sort=order),
            x=alt.X(f"{column}:Q"),
            text=alt.Text(f"{column}:Q", format=".3f"),
        )
    )
    return (bars + labels).properties(height=150)


def segments_page() -> None:
    features = load_features()
    labels = load_output("cluster_labels.csv")
    profiles = load_output("cluster_profiles.csv")
    comparison = load_output("clustering_comparison.csv")
    tuning = load_output("clustering_tuning.csv")
    stability = load_output("clustering_stability.csv")
    if any(frame is None for frame in (labels, profiles, comparison, tuning, stability)):
        st.error("Clustering outputs are missing. Run scripts/build_advanced_models.py.")
        return

    loans = len(features)
    base_rate = float(features["target"].mean())
    best = comparison.sort_values("silhouette", ascending=False).iloc[0]
    kmeans = comparison[comparison["method"] == "kmeans"].iloc[0]
    dbscan = comparison[comparison["method"] == "dbscan"].iloc[0]
    dbscan_profile = profiles[profiles["method"] == "dbscan"].set_index("cluster")
    noise_rate = (
        float(dbscan_profile.loc[-1, "default_rate"])
        if -1 in dbscan_profile.index
        else float("nan")
    )
    core_rate = (
        float(dbscan_profile.loc[0, "default_rate"])
        if 0 in dbscan_profile.index
        else float("nan")
    )

    page_header(
        "Unsupervised learning",
        "Four ways to segment the loan book",
        f"{loans:,} loans, {len(CLUSTER_FEATURES)} pre-loan behaviour features, and no target "
        "in the fitting step. The default flag is joined only after the clusters are fixed.",
    )

    kpi_row(
        [
            ("Loans", f"{loans:,}", "One row per loan application"),
            ("Behaviour features", f"{len(CLUSTER_FEATURES)}", "Pre-loan activity only"),
            ("Best silhouette", f"{best['silhouette']:.3f}", f"{METHOD_NAMES[best['method']]}"),
            ("K-Means default split", f"{kmeans['n_clusters']:.0f}", "Low vs high activity"),
        ]
    )

    section(
        "Why this dataset can be clustered",
        "G: dataset name, size, features and preprocessing",
    )
    st.markdown(
        "The clustering dataset is the same 682 loans used by the supervised model, "
        "but it keeps only the 13 behavioural columns. Loan amount, duration, payments, "
        "age and gender are deliberately excluded: they describe the product or the applicant, "
        "not how the account behaved before the loan. The target is never passed to a clustering "
        "algorithm or used to choose k or eps."
    )
    dataset = pd.DataFrame(
        [
            ["Source", "PKDD'99 Czech banking data, cleaned loan and transaction tables"],
            ["Unit of analysis", f"{loans:,} loan applications, {features['target'].sum():.0f} defaults"],
            ["Features", f"{len(CLUSTER_FEATURES)} pre-loan behavioural features"],
            ["Excluded", "Target, loan amount, duration, payments, age, gender"],
            ["Missing values", f"{int(features[CLUSTER_FEATURES].isna().sum().sum())}"],
            ["Scaling", "StandardScaler fitted on all 682 loans; PCA used only for the 2D map"],
            ["Cluster tendency", "Hopkins statistic 0.85 (offline check); correlations discussed below"],
            ["Target use", "Joined after fitting only, to profile cluster default rates"],
        ],
        columns=["Item", "Value"],
    )
    grid(dataset)
    with st.expander("The 13 features and why they are relevant", expanded=False):
        grid(
            pd.DataFrame(
                FEATURE_DETAILS,
                columns=["Feature", "What it measures", "Why it matters"],
            )
        )
    note(
        "Several features are correlated (for example n_txn and n_withdrawal, 0.98). "
        "That is expected in behavioural data: the comparison is about algorithm behaviour, "
        "not about finding independent factors. A six-feature pruned sensitivity check gives "
        "silhouette 0.308 for K-Means and 0.315 for K-Medoids, so the ranking is not an artefact "
        "of redundant columns.",
        label="Data note",
    )

    section(
        "Algorithm comparison",
        "H: four algorithms, same standardised feature space, same 682 loans",
    )
    table = _metric_table(comparison)
    grid(
        table,
        score_config("Silhouette", "Davies-Bouldin", "Calinski-Harabasz", "Stability ARI")
        | score_config("Coverage %", "Noise %", places=1)
        | {"Clusters": st.column_config.NumberColumn(format="%d")},
    )
    st.caption(
        "Silhouette: higher is better. Davies-Bouldin: lower is better. "
        "Calinski-Harabasz: higher is better. Stability ARI: agreement with the full-data "
        "labels over 10 random 90% subsamples. DBSCAN has one core cluster, so the three "
        "internal metrics are undefined; its row still reports coverage, noise and stability."
    )

    st.markdown("**Metric values**")
    columns = st.columns(3)
    for column, (name, title, higher) in zip(
        columns,
        [
            ("silhouette", "Silhouette", True),
            ("davies_bouldin", "Davies-Bouldin", False),
            ("calinski_harabasz", "Calinski-Harabasz", True),
        ],
    ):
        chart = _metric_chart(comparison, name, title, higher)
        if chart is not None:
            column.altair_chart(chart, width="stretch")

    stability_frame = stability.copy()
    stability_frame = stability_frame.set_index("method").loc[_method_order(stability_frame)]
    stability_frame = stability_frame.reset_index()
    stability_frame["Method"] = stability_frame["method"].map(METHOD_NAMES)
    stability_frame["Low"] = stability_frame["mean_ari"] - stability_frame["std_ari"]
    stability_frame["High"] = stability_frame["mean_ari"] + stability_frame["std_ari"]
    st.markdown("**Stability across 10 subsamples**")
    show(
        alt.Chart(stability_frame)
        .mark_bar(size=26, color=ACCENT)
        .encode(
            y=alt.Y("Method:N", sort=[METHOD_NAMES[m] for m in _method_order(stability_frame)], title=None),
            x=alt.X("mean_ari:Q", title="Mean Adjusted Rand Index", scale=alt.Scale(domain=[0, 1])),
            tooltip=[
                "Method",
                alt.Tooltip("mean_ari:Q", format=".3f"),
                alt.Tooltip("std_ari:Q", format=".3f"),
                alt.Tooltip("min_ari:Q", format=".3f"),
                alt.Tooltip("max_ari:Q", format=".3f"),
            ],
        )
        .properties(height=150)
    )

    section("How each algorithm works", "H: working principle, strengths and limitations")
    grid(
        pd.DataFrame(
            ALGORITHM_NOTES,
            columns=["Algorithm", "Working principle", "Strength", "Limitation"],
        )
    )
    with st.expander("Parameter selection", expanded=False):
        selected = tuning[tuning["selected"]].copy()
        selected["Method"] = selected["method"].map(METHOD_NAMES).fillna(selected["method"])
        selected["Valid"] = selected["valid"].map({True: "yes", False: "no"})
        selected["Silhouette"] = selected["silhouette"]
        selected["Min cluster %"] = (selected["min_cluster_share"] * 100).round(1)
        selected["Noise %"] = (selected["noise_share"] * 100).round(1)
        grid(
            selected[
                [
                    "Method",
                    "param_name",
                    "param_value",
                    "n_clusters",
                    "Silhouette",
                    "Min cluster %",
                    "Noise %",
                    "Valid",
                ]
            ].rename(
                columns={
                    "param_name": "Parameter",
                    "param_value": "Value",
                    "n_clusters": "Clusters",
                }
            ),
            score_config("Silhouette") | score_config("Min cluster %", "Noise %", places=1),
        )
        st.caption(
            "Partitional algorithms search k=2..6 and keep the highest silhouette subject to "
            "every cluster holding at least 5% of loans. K-Means and K-Medoids choose k=2; "
            "AGNES chooses k=2. DBSCAN scans eps 0.8..2.0 with min_samples=5; no eps produced "
            "two clusters of at least 5% at reasonable noise, so eps=1.4 is used as a core/noise "
            "layer (81.1% coverage). The full search is in artifacts/metrics/clustering_tuning.csv."
        )

    section("Cluster maps", "H: the same loans projected to two principal components")
    columns = st.columns(2)
    for column, (title, figure) in zip(columns * 2, cluster_map_figures(features, labels)):
        column.markdown(f"**{title}**")
        with column:
            show(figure)
    st.caption(
        "PCA is used only for this picture; all clustering and metrics use the full 13-dimensional "
        "standardised space. Nearby points have similar pre-loan behaviour, but distances in a "
        "two-axis projection are approximate."
    )

    section("Cluster profiles", "I: sizes, behaviour and post-hoc default rates")
    view = profiles.copy()
    view["Method"] = view["method"].map(METHOD_NAMES).fillna(view["method"])
    view["Segment"] = view["cluster"].map(lambda value: "noise" if value == -1 else str(value))
    view["Share %"] = (view["share"] * 100).round(1)
    view["Default %"] = (view["default_rate"] * 100).round(1)
    view = view[
        [
            "Method",
            "Segment",
            "size",
            "Share %",
            "Default %",
            "avg_balance",
            "avg_n_txn",
            "avg_months_active",
            "avg_txn_per_month",
            "avg_sum_deposit",
        ]
    ].rename(
        columns={
            "size": "Loans",
            "avg_balance": "Avg balance",
            "avg_n_txn": "Avg transactions",
            "avg_months_active": "Avg months",
            "avg_txn_per_month": "Txns / month",
            "avg_sum_deposit": "Avg deposits",
        }
    )
    grid(
        view,
        score_config("Share %", "Default %", places=1)
        | score_config("Avg balance", "Avg deposits", places=0)
        | score_config("Avg transactions", "Avg months", "Txns / month", places=1),
    )
    show(cluster_rate_figure(profiles, base_rate))
    st.caption(
        f"The dashed line is the {percent(base_rate)} portfolio default rate. Red bars sit above it. "
        "DBSCAN's noise group is the clearest risk flag."
    )

    section("What the segments mean", "I: interpretation and practical use")
    low = profiles[(profiles["method"] == "kmeans") & (profiles["cluster"] == 1)].iloc[0]
    high = profiles[(profiles["method"] == "kmeans") & (profiles["cluster"] == 0)].iloc[0]
    st.markdown(
        f"- **Low-activity segment (K-Means cluster 1):** {low['size']:.0f} loans, "
        f"{low['default_rate']:.1%} default, {low['avg_n_txn']:.0f} transactions and "
        f"{low['avg_months_active']:.0f} months of history. This is the higher-risk group. "
        "Use it for early engagement, repayment reminders and a manual check before approval.\n"
        f"- **High-activity segment (K-Means cluster 0):** {high['size']:.0f} loans, "
        f"{high['default_rate']:.1%} default, {high['avg_n_txn']:.0f} transactions and "
        f"{high['avg_months_active']:.0f} months of history. Standard monitoring is enough; "
        "this is the lower-risk group.\n"
        f"- **DBSCAN noise:** {dbscan_profile.loc[-1, 'size']:.0f} loans, "
        f"{noise_rate:.1%} default versus {core_rate:.1%} in the core. Treat noise as an "
        "outlier flag for manual review, not as a fifth customer segment."
    )
    note(
        "K-Means is the recommended primary segmentation: highest silhouette among valid "
        "partitional solutions (0.294), highest stability (ARI 0.931), full coverage and a "
        "predict method for new loans. K-Medoids is close on fit but unstable across subsamples "
        "(ARI 0.534). AGNES is less stable (0.421) and has no predict method. DBSCAN is a "
        "useful complementary outlier detector, not a multi-segment solution.",
        label="Verdict",
    )

    section("Metrics used", "J: what each number means and which direction is better")
    grid(
        pd.DataFrame(
            [
                ["Silhouette", "Cohesion inside a cluster versus separation from the next", "Higher", "0.294 (K-Means)"],
                ["Davies-Bouldin", "Average similarity between each cluster and its closest neighbour", "Lower", "1.346 (K-Means)"],
                ["Calinski-Harabasz", "Between-cluster dispersion divided by within-cluster dispersion", "Higher", "306.3 (K-Means)"],
                ["Adjusted Rand Index", "Agreement between full-data and subsample labels", "Higher", "0.931 (K-Means)"],
            ],
            columns=["Metric", "What it measures", "Better", "Selected value"],
        )
    )
    st.caption(
        "Clustering metrics do not use the default label. Accuracy, precision, recall and F1 "
        "belong to the supervised model; using them to choose clusters would leak the target. "
        "The default rates above are descriptive, computed after the clusters are fixed."
    )
