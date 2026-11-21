"""Snowflake deployment page."""

import pandas as pd
import streamlit as st

from components.common import (
    ROOT,
    grid,
    kpi_row,
    note,
    page_header,
    section,
)
from data_access import data_source_label, snowflake_enabled, snowflake_query

INVENTORY = pd.DataFrame(
    [
        ("RAW", "RAW_LOAN", "682", "Loan application extract"),
        ("RAW", "RAW_TRANS", "1,056,320", "Account transaction extract"),
        ("RAW", "RAW_CLIENT", "5,369", "Client extract"),
        ("RAW", "RAW_ACCOUNT", "4,500", "Account extract"),
        ("RAW", "RAW_DISP", "5,369", "Client-account disposition"),
        ("MART", "MART_LOAN_FEATURES", "682", "Leakage-safe feature mart"),
        ("MART", "MART_MODEL_SCORES", "682", "Out-of-fold review scores"),
        ("MART", "MART_LOAN_SEGMENTS", "682", "K-Means segment labels"),
        ("MART", "MART_LOAN_RISK", "682", "Features, score, and segment"),
        ("MART", "MART_CLUSTER_LABELS", "682", "Four clustering methods"),
        ("MART", "MART_CLUSTERING_COMPARISON", "4", "Clustering metrics and verdict"),
        ("MART", "V_LOAN_PERF", "view", "Loan performance with score and segment"),
        ("MART", "V_TRANS_MONTHLY", "view", "Pre-loan transaction monthly rollup"),
    ],
    columns=["Schema", "Object", "Expected rows", "Purpose"],
)

LIVE_COUNTS = """
SELECT 'RAW.RAW_CLIENT' AS table_name, COUNT(*) AS rows
FROM LOAN_DEFAULT_DW.RAW.RAW_CLIENT
UNION ALL SELECT 'RAW.RAW_ACCOUNT', COUNT(*) FROM LOAN_DEFAULT_DW.RAW.RAW_ACCOUNT
UNION ALL SELECT 'RAW.RAW_DISP', COUNT(*) FROM LOAN_DEFAULT_DW.RAW.RAW_DISP
UNION ALL SELECT 'RAW.RAW_LOAN', COUNT(*) FROM LOAN_DEFAULT_DW.RAW.RAW_LOAN
UNION ALL SELECT 'RAW.RAW_TRANS', COUNT(*) FROM LOAN_DEFAULT_DW.RAW.RAW_TRANS
UNION ALL SELECT 'MART.MART_LOAN_FEATURES', COUNT(*) FROM LOAN_DEFAULT_DW.MART.MART_LOAN_FEATURES
UNION ALL SELECT 'MART.MART_MODEL_SCORES', COUNT(*) FROM LOAN_DEFAULT_DW.MART.MART_MODEL_SCORES
UNION ALL SELECT 'MART.MART_LOAN_SEGMENTS', COUNT(*) FROM LOAN_DEFAULT_DW.MART.MART_LOAN_SEGMENTS
UNION ALL SELECT 'MART.MART_LOAN_RISK', COUNT(*) FROM LOAN_DEFAULT_DW.MART.MART_LOAN_RISK
UNION ALL SELECT 'MART.MART_CLUSTER_LABELS', COUNT(*) FROM LOAN_DEFAULT_DW.MART.MART_CLUSTER_LABELS
UNION ALL SELECT 'MART.MART_CLUSTERING_COMPARISON', COUNT(*) FROM LOAN_DEFAULT_DW.MART.MART_CLUSTERING_COMPARISON
ORDER BY table_name
"""


@st.cache_data(show_spinner="Reading Snowflake row counts...", ttl=120)
def snowflake_counts() -> pd.DataFrame:
    return snowflake_query(LIVE_COUNTS)


@st.cache_data(show_spinner="Reading Snowflake views...", ttl=120)
def snowflake_samples() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    performance = snowflake_query(
        "SELECT * FROM LOAN_DEFAULT_DW.MART.V_LOAN_PERF LIMIT 8"
    )
    monthly = snowflake_query(
        "SELECT * FROM LOAN_DEFAULT_DW.MART.V_TRANS_MONTHLY LIMIT 8"
    )
    comparison = snowflake_query(
        "SELECT method, parameter, n_clusters, silhouette, coverage, "
        "stability_ari_mean, role, recommended "
        "FROM LOAN_DEFAULT_DW.MART.MART_CLUSTERING_COMPARISON "
        "ORDER BY recommended DESC, silhouette DESC"
    )
    return performance, monthly, comparison


def snowflake_page() -> None:
    page_header(
        "Dashboard 2 · Snowflake deployment",
        "The same warehouse, in Snowflake",
        "The local DuckDB build and the Snowflake deployment share one schema contract. "
        "Raw extracts stay in RAW; every modelled table and reporting view lives in MART.",
    )

    enabled = snowflake_enabled()
    kpi_row(
        [
            ("Current source", data_source_label(), "DATA_SOURCE in .env"),
            ("Warehouse", "PBI_WH", "XSMALL, auto-suspend"),
            ("Schemas", "RAW + MART", "Separated by layer"),
            ("Loader", "idempotent", "Truncate, then write"),
        ]
    )

    section("Deployment status", "What the dashboard is reading")
    if enabled:
        note(
            "The app is reading the six MART tables from Snowflake. The local "
            "artifacts remain in use for model metrics and pattern outputs.",
            "Snowflake mode",
        )
    else:
        note(
            "The app is reading DuckDB. The Snowflake objects, DDL, loader, and "
            "verification queries are ready; set DATA_SOURCE=snowflake after the "
            "upload to switch the six MART tables to live reads.",
            "Local mode",
            "risk",
        )

    section("Schema split", "Raw and modelled objects are not mixed")
    st.markdown(
        """
        **RAW** holds the five source extracts: `RAW_LOAN`, `RAW_TRANS`,
        `RAW_CLIENT`, `RAW_ACCOUNT`, and `RAW_DISP`.

        **MART** holds the feature mart, scores, segments, risk table, the
        four-method cluster labels, the clustering comparison, and the two
        reporting views `V_LOAN_PERF` and `V_TRANS_MONTHLY`.

        The loader routes each CSV to its schema and truncates the target first,
        so a rerun replaces rows instead of appending duplicates.
        """
    )

    section("Object inventory", "Expected after a clean load")
    grid(INVENTORY)

    section("Row counts", "Live when Snowflake is enabled")
    if enabled:
        try:
            grid(snowflake_counts())
        except Exception as exc:
            st.error(f"Snowflake count query failed: {exc}")
    else:
        note(
            "These are the counts checked by sql/snowflake/verify_load.sql. "
            "After the upload, this panel switches to live Snowflake counts.",
            "Verification target",
        )
        grid(
            INVENTORY[INVENTORY["Expected rows"] != "view"][
                ["Schema", "Object", "Expected rows"]
            ]
        )

    section("Reporting views", "SQL that a reviewer can run")
    if enabled:
        try:
            performance, monthly, comparison = snowflake_samples()
            st.caption("MART.V_LOAN_PERF")
            st.dataframe(performance, hide_index=True, width="stretch")
            st.caption("MART.V_TRANS_MONTHLY")
            st.dataframe(monthly, hide_index=True, width="stretch")
            st.caption("MART.MART_CLUSTERING_COMPARISON")
            st.dataframe(comparison, hide_index=True, width="stretch")
        except Exception as exc:
            st.error(f"Snowflake view query failed: {exc}")
    else:
        st.code(
            "SELECT * FROM LOAN_DEFAULT_DW.MART.V_LOAN_PERF LIMIT 10;\n"
            "SELECT * FROM LOAN_DEFAULT_DW.MART.V_TRANS_MONTHLY LIMIT 10;\n"
            "SELECT * FROM LOAN_DEFAULT_DW.MART.MART_CLUSTERING_COMPARISON;",
            language="sql",
        )

    with st.expander("Verification script"):
        st.code(
            (ROOT / "sql" / "snowflake" / "verify_load.sql").read_text(
                encoding="utf-8"
            ),
            language="sql",
        )

    with st.expander("Deployment commands"):
        st.code(
            "venv\\Scripts\\python.exe scripts\\upload_to_snowflake.py\n"
            "-- then run sql/snowflake/verify_load.sql in Snowsight",
            language="powershell",
        )
