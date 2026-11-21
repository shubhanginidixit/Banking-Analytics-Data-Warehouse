"""Data model card page."""

import pandas as pd
import streamlit as st

from components.common import (
    FOREIGN_KEYS,
    SCHEMA,
    grid,
    kpi_row,
    load_features,
    mart_counts,
    note,
    page_header,
    row_count_chart,
    section,
    show,
)
from data_access import query

ANALYTICS_LAYER = {
    "loan_features": "Leakage-safe modelling table",
    "loan_risk_dashboard": "Features plus score and segment",
    "model_scores": "Out-of-fold review scores",
    "advanced_scores": "MLP and CBA scores",
    "cv_comparison": "Repeated-CV comparison",
    "metrics_comparison": "Held-out comparison",
    "tuning_results": "Randomized search results",
    "capacity_model_comparison": "Capacity-aware model check",
    "capacity_model_fold_results": "Capacity-aware fold results",
    "advanced_metrics": "MLP and CBA metrics",
    "loan_segments": "KMeans loan segments",
    "segment_profiles": "KMeans segment profiles",
    "cluster_labels": "K-Means, K-medoids, AGNES, DBSCAN labels",
    "cluster_profiles": "Cluster profiles for all four methods",
    "pattern_itemsets": "FP-Growth and Apriori itemsets",
    "pattern_rules": "Association rules",
    "cba_rules": "Classifier rules",
}

DOT_HEAD = """digraph galaxy {
  graph [rankdir=LR, bgcolor="transparent", pad=0.2, nodesep=0.45, ranksep=0.9];
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11,
        color="#e4e1d8", fillcolor="#fbfaf7", fontcolor="#191b1f", margin="0.14,0.08"];
  edge [color="#6b6f76", arrowsize=0.7, fontname="Helvetica", fontsize=9,
        fontcolor="#6b6f76"];
"""


def galaxy_dot(counts: pd.DataFrame) -> str:
    sizes = dict(zip(counts["table"], counts["rows"]))
    order = [
        "fact_loans",
        "fact_transactions",
        "dim_account",
        "dim_branch",
        "dim_customer",
        "dim_date",
        "dim_loan_type",
    ]
    nodes = "\n".join(
        f'  {name} [label="{name}\\n{int(sizes.get(name, 0)):,}"'
        + (', fillcolor="#1f4e6b", fontcolor="white"' if name.startswith("fact_") else "")
        + "];"
        for name in order
    )
    edges = "\n".join(f'  {src} -> {dst} [label="{key}"];' for src, key, dst in FOREIGN_KEYS)
    return f"{DOT_HEAD}{nodes}\n\n{edges}\n}}"


@st.cache_data(show_spinner=False, ttl=600)
def schema_frame() -> pd.DataFrame:
    columns = query(
        "SELECT table_name, column_name, data_type, ordinal_position "
        "FROM information_schema.columns "
        "WHERE table_schema = 'mart' "
        "ORDER BY table_name, ordinal_position"
    )
    columns["data_type"] = (
        columns["data_type"]
        .str.replace("TIMESTAMP_NS", "TIMESTAMP", regex=False)
        .str.replace(r"^ENUM\(.*\)$", "ENUM", regex=True)
    )
    counts = mart_counts().rename(columns={"table": "table_name"})
    return columns.merge(counts, on="table_name", how="left")


@st.cache_data(show_spinner=False, ttl=600)
def analytics_counts() -> pd.DataFrame:
    tables = query(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'analytics' ORDER BY table_name"
    )["table_name"].tolist()
    if not tables:
        return pd.DataFrame(columns=["Table", "Purpose", "Rows"])
    counts = query(
        " UNION ALL ".join(
            f"SELECT '{table}' AS table_name, count(*) AS rows FROM analytics.{table}"
            for table in tables
        )
    )
    return pd.DataFrame(
        {
            "Table": "analytics." + counts["table_name"],
            "Purpose": counts["table_name"].map(ANALYTICS_LAYER).fillna("Saved output"),
            "Rows": counts["rows"].astype(int),
        }
    )


def data_model_page() -> None:
    features = load_features()
    columns = schema_frame()
    counts = mart_counts()
    analytics = analytics_counts()
    dimensions = int(counts["table"].str.startswith("dim_").sum())
    facts = int(counts["table"].str.startswith("fact_").sum())

    page_header(
        "Dashboard 1 · Data Warehouse Design",
        "A galaxy schema with two facts and one conformed date spine",
        "Kimball bottom-up design: dimensions carry descriptive context, facts carry the "
        "measurable events. Both facts share account, branch and date keys, so deposits and "
        "loans can be compared on the same axes.",
    )

    kpi_row(
        [
            ("Dimensions", f"{dimensions}", "Shared descriptive context"),
            ("Fact tables", f"{facts}", "Loans and transactions"),
            ("Mart columns", f"{len(columns)}", "Keys, attributes, measures"),
            ("Analytics tables", f"{len(analytics)}", "Features, scores, rules"),
            ("Modelling rows", f"{len(features):,}", "One row per loan"),
        ]
    )

    section("Galaxy schema", f"{facts} fact tables share {dimensions} conformed dimensions")
    st.graphviz_chart(galaxy_dot(counts), width="stretch")
    st.caption(
        "Blue nodes are fact tables, light nodes are dimensions, and edge labels are join "
        "keys. One fact plus its dimensions is a star; two facts sharing conformed account, "
        "branch, and date keys make the pair a galaxy."
    )

    with st.expander("Column dictionary"):
        st.markdown("Pick a mart table to inspect its columns and keys.")
        chosen = st.selectbox("Table", counts["table"].tolist())
        table_columns = columns[columns["table_name"] == chosen].copy()
        foreign = {(src, key) for src, key, _ in FOREIGN_KEYS}
        primary_key = SCHEMA.get(chosen)

        def key_role(row) -> str:
            if primary_key == row["column_name"]:
                return "Primary key"
            if (chosen, row["column_name"]) in foreign:
                return "Foreign key"
            return "Attribute"

        table_columns["Key"] = table_columns.apply(key_role, axis=1)
        grid(
            table_columns[["column_name", "data_type", "Key"]].rename(
                columns={"column_name": "Column", "data_type": "Type"}
            )
        )

    section("Analytics layer", "Derived tables loaded into the same database")
    chart_frame = (
        analytics.assign(
            Table=analytics["Table"].str.replace("analytics.", "", regex=False)
        )
        .sort_values("Rows", ascending=False)
        .head(10)
    )
    show(row_count_chart(chart_frame))
    with st.expander("All analytics tables"):
        grid(analytics, {"Rows": st.column_config.NumberColumn(format="%d")})

    note(
        f"{facts} facts, {dimensions} dimensions, and {len(FOREIGN_KEYS)} foreign-key joins. "
        "`dim_loan_type` holds one explicit unknown member because PKDD'99 has no loan product "
        "or interest rate; the fact keeps `interest_rate` null instead of inventing a rate. "
        "The date dimension is conformed, so loan and transaction dates roll up through the "
        "same calendar.",
        label="Design notes",
    )
    st.caption("This card reads the local DuckDB catalog; Snowflake mode serves the four analytics tables.")
