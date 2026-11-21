"""Data and leakage page."""

import pandas as pd
import streamlit as st

from components.common import (
    kpi_row,
    leakage_audit,
    leakage_figure,
    load_features,
    mart_counts,
    note,
    page_header,
    percent,
    row_count_chart,
    section,
    show,
    transaction_summary,
)


def data_page() -> None:
    features = load_features()
    counts = mart_counts()
    audit = leakage_audit()
    summary = transaction_summary()
    matches = int((audit["used"] == audit["n_txn"]).sum())

    page_header(
        "Dashboard 3 · Supervised Learning Dataset",
        f"{summary['total']:,} transactions, {len(features):,} loans, no lookahead",
        "The modelling table is built from the local DuckDB galaxy schema. One rule keeps it "
        "honest: every transaction used for a loan happened strictly before the loan date.",
    )

    section("Galaxy schema", "Row counts read from DuckDB mart")
    schema = counts.rename(columns={"table": "Table", "rows": "Rows"})
    schema["Role"] = schema["Table"].map(
        lambda name: "Fact" if name.startswith("fact_") else "Dimension"
    )
    show(row_count_chart(schema, "Table", "Rows", "Role"))
    st.dataframe(
        schema,
        hide_index=True,
        width="stretch",
        column_config={"Rows": st.column_config.NumberColumn(format="%d")},
    )

    section("Leakage check", "Pick a loan; the split is computed from fact_transactions")
    loan_dates = dict(zip(features["loan_id"], features["parsed_date"]))
    loan_amounts = dict(zip(features["loan_id"], features["amount"]))
    loan_id = st.selectbox(
        "Loan",
        options=features["loan_id"].tolist(),
        format_func=lambda value: f"{value} | {loan_dates[value]} | {loan_amounts[value]:,.0f} CZK",
    )
    chosen_rows = audit[audit["loan_id"] == loan_id]
    if chosen_rows.empty:
        st.warning("This loan is missing from the leakage audit.")
        return
    chosen = chosen_rows.iloc[0]
    excluded = int(chosen["total"] - chosen["used"])
    kpi_row(
        [
            ("Loan date", str(loan_dates[loan_id]), "Cutoff for this loan"),
            ("Account", f"{int(chosen['account_id'])}", "Transaction key"),
            ("Transactions used", f"{int(chosen['used'])}", "Strictly before the loan"),
            ("Transactions excluded", f"{excluded:,}", "On or after the loan"),
        ]
    )
    used = int(chosen["used"])
    total = max(used + excluded, 1)
    show(leakage_figure(used, excluded))
    st.caption(
        f"{used} of {total:,} transactions on this account fall strictly before the loan date. "
        f"The other {excluded:,} are excluded from the feature build."
    )

    note(
        f"Across all {len(features)} loans, n_txn equals the count of transactions strictly before "
        f"the loan date: {matches} of {len(features)} match. The audit counts only "
        "`date_id < loan_date_id`, so same-day rows are excluded by construction.",
        label="Verified",
    )

    section("Feature provenance", "Where each modelling column comes from")
    provenance = pd.DataFrame(
        [
            ["Loan terms", "loan.csv", "amount, duration, payments, date"],
            ["Transaction volume", "trans.csv", "n_txn, n_deposit, n_withdrawal, sum_deposit, sum_withdrawal"],
            ["Transaction size", "trans.csv", "avg_amount, std_amount"],
            ["Balance history", "trans.csv", "avg_balance, min_balance, max_balance, balance_at_loan"],
            ["Activity window", "trans.csv", "months_active, txn_per_month"],
            ["Customer", "client.csv", "gender, age, district_id"],
            ["Ownership", "disp.csv, account.csv", "account_id, client_id"],
        ],
        columns=["Group", "Source", "Columns"],
    )
    st.dataframe(provenance, hide_index=True, width="stretch")

    section("Quality checks", "Computed from the modelling table")
    account_rows = counts.loc[counts["table"] == "dim_account", "rows"]
    account_note = (
        f"Of {int(account_rows.iloc[0]):,} accounts"
        if not account_rows.empty
        else "Distinct accounts"
    )
    checks = pd.DataFrame(
        [
            ["Rows", f"{len(features):,}", "One per loan"],
            ["Columns", f"{features.shape[1]}", "Terms, behaviour, context, target"],
            ["Missing cells", f"{int(features.isna().sum().sum())}", "None"],
            ["Duplicate loan ids", f"{int(features['loan_id'].duplicated().sum())}", "None"],
            ["Defaults", f"{int(features['target'].sum())}", percent(features["target"].mean())],
            ["Transaction dates", f"{summary['min_date']} to {summary['max_date']}", "From fact_transactions"],
            ["Transactions", f"{summary['total']:,}", "Before and after loans"],
            ["Accounts with transactions", f"{summary['accounts']:,}", account_note],
        ],
        columns=["Check", "Result", "Note"],
    )
    st.dataframe(checks, hide_index=True, width="stretch")
