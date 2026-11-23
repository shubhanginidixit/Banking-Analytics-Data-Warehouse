"""Loan default review dashboard.

Every figure is read or computed from the artefacts produced by the notebooks
and tools/ scripts: data/features/loan_features.csv, artifacts/*.csv and
artifacts/models/*.pkl. Charts are built with Altair; nothing is typed in by hand.
"""

from __future__ import annotations

import sys
from ast import literal_eval
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from data_access import (
    data_source_label,
    query,
    read_table,
    snowflake_enabled,
    snowflake_leakage,
    table_counts as db_table_counts,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
CLEAN = DATA / "clean"
ARTIFACTS = ROOT / "artifacts"
MODEL_DIR = ARTIFACTS / "models"

INK = "#191b1f"
MUTED = "#6b6f76"
RULE = "#e4e1d8"
ACCENT = "#1f4e6b"
RISK = "#a63d2f"
GOOD = "#2f6b4f"

DROP = [
    "loan_id",
    "account_id",
    "client_id",
    "district_id",
    "status",
    "date",
    "parsed_date",
    "gender",
    "target",
]

SIM_INPUTS = {
    "amount": "Loan amount",
    "duration": "Duration",
    "balance_at_loan": "Balance at loan date",
    "min_balance": "Minimum balance",
    "txn_per_month": "Transactions per month",
}

SCHEMA = {
    "dim_date": "date_id",
    "dim_branch": "branch_id",
    "dim_customer": "customer_id",
    "dim_account": "account_id",
    "dim_loan_type": "loan_type_id",
    "fact_loans": "loan_id",
    "fact_transactions": "transaction_id",
}

FOREIGN_KEYS = [
    ("dim_account", "customer_id", "dim_customer"),
    ("dim_customer", "home_branch_id", "dim_branch"),
    ("fact_loans", "account_id", "dim_account"),
    ("fact_loans", "customer_id", "dim_customer"),
    ("fact_loans", "branch_id", "dim_branch"),
    ("fact_loans", "loan_type_id", "dim_loan_type"),
    ("fact_loans", "date_id", "dim_date"),
    ("fact_transactions", "account_id", "dim_account"),
    ("fact_transactions", "branch_id", "dim_branch"),
    ("fact_transactions", "date_id", "dim_date"),
]

CLUSTER_FEATURES = [
    "n_txn",
    "n_deposit",
    "n_withdrawal",
    "sum_deposit",
    "sum_withdrawal",
    "avg_amount",
    "std_amount",
    "avg_balance",
    "min_balance",
    "max_balance",
    "balance_at_loan",
    "months_active",
    "txn_per_month",
]

METHOD_NAMES = {
    "kmeans": "K-Means",
    "kmedoids": "K-Medoids",
    "agnes": "AGNES (Ward)",
    "dbscan": "DBSCAN",
}
METHOD_ORDER = list(METHOD_NAMES)


def stress_text(key: str, value: float) -> str:
    if key in {"amount", "balance_at_loan", "min_balance"}:
        return f"{value:,.0f} CZK"
    if key == "duration":
        return f"{value:.0f} months"
    return f"{value:.1f} per month"

THEME_CSS = """
    <style>
    :root {
      --paper: #fbfaf7; --ink: #191b1f; --muted: #6b6f76; --rule: #e4e1d8;
      --accent: #1f4e6b; --risk: #a63d2f; --good: #2f6b4f;
      --serif: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Palatino, Georgia, serif;
      --sans: "Segoe UI", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif;
    }
    .stApp { background: var(--paper); color: var(--ink); font-family: var(--sans); }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] { background: #f3f0e7; border-right: 1px solid var(--rule); }
    .block-container { max-width: 1160px; padding-top: 2.4rem; padding-bottom: 4rem; }
    h1, h2, h3 { font-family: var(--serif); font-weight: 500; letter-spacing: -0.015em; color: var(--ink); }
    h1 { font-size: 2.3rem; line-height: 1.15; margin: 0.15rem 0 0.45rem; }
    h3 { font-size: 1.12rem; margin-top: 1.4rem; }
    p, li { line-height: 1.6; }
    a { color: var(--accent); }
    .eyebrow { font-size: 0.68rem; font-weight: 700; letter-spacing: 0.16em; text-transform: uppercase; color: var(--accent); }
    .lede { max-width: 64ch; color: var(--muted); font-size: 0.97rem; margin: 0.1rem 0 1.5rem; }
    .kpi { border-top: 1px solid var(--ink); padding: 0.55rem 0 0; }
    .kpi-label { font-size: 0.63rem; font-weight: 700; letter-spacing: 0.13em; text-transform: uppercase; color: var(--muted); }
    .kpi-value { font-family: var(--serif); font-size: 1.72rem; line-height: 1.15; margin-top: 0.2rem; font-variant-numeric: tabular-nums; }
    .kpi-note { font-size: 0.74rem; color: var(--muted); margin-top: 0.12rem; }
    .section { display: flex; align-items: baseline; gap: 0.9rem; border-top: 1px solid var(--rule); margin: 2.4rem 0 0.9rem; padding-top: 0.8rem; }
    .section span { font-size: 0.69rem; font-weight: 700; letter-spacing: 0.15em; text-transform: uppercase; color: var(--ink); white-space: nowrap; }
    .section em { font-style: normal; font-size: 0.78rem; color: var(--muted); }
    .note { border-left: 2px solid var(--accent); padding: 0.05rem 0 0.05rem 0.85rem; margin: 0.6rem 0 1.2rem; font-size: 0.9rem; line-height: 1.55; }
    .note b { display: block; font-size: 0.62rem; font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase; color: var(--accent); margin-bottom: 0.22rem; }
    .note.risk { border-color: var(--risk); }
    .note.risk b { color: var(--risk); }
    .brand { font-family: var(--serif); font-size: 1.14rem; line-height: 1.2; }
    .brand-sub { font-size: 0.72rem; color: var(--muted); line-height: 1.45; margin: 0.25rem 0 1.1rem; }
    .side-label { font-size: 0.61rem; font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase; color: var(--muted); margin: 1.3rem 0 0.35rem; }
    .side-line { display: flex; justify-content: space-between; font-size: 0.78rem; border-bottom: 1px solid var(--rule); padding: 0.24rem 0; }
    .side-line span:last-child { font-variant-numeric: tabular-nums; }
    .prob { font-family: var(--serif); font-size: 3.2rem; line-height: 1; font-variant-numeric: tabular-nums; letter-spacing: -0.02em; }
    .prob.risk { color: var(--risk); }
    .prob.calm { color: var(--good); }
    .verdict { font-size: 0.9rem; line-height: 1.5; margin-top: 0.35rem; color: var(--ink); }
    .matrix { border-collapse: collapse; width: 100%; font-size: 0.82rem; }
    .matrix th, .matrix td { border-bottom: 1px solid var(--rule); padding: 0.42rem 0.5rem; text-align: right; }
    .matrix th:first-child, .matrix td:first-child { text-align: left; }
    .matrix thead th { font-size: 0.61rem; letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted); font-weight: 700; }
    .matrix td.num { font-variant-numeric: tabular-nums; }
    .matrix .hit { color: var(--good); font-weight: 600; }
    .matrix .miss { color: var(--risk); font-weight: 600; }
    [data-testid="stDataFrame"] { border: 1px solid var(--rule); }
    .stTabs [data-baseweb="tab-list"] { gap: 1.5rem; border-bottom: 1px solid var(--rule); }
    .stTabs [data-baseweb="tab"] { padding: 0.4rem 0; font-size: 0.8rem; }
    [data-testid="stExpander"] details { border: 1px solid var(--rule); border-radius: 0; }
    div[data-testid="stImage"] figcaption { font-size: 0.74rem; color: var(--muted); }
    </style>
"""


def apply_theme() -> None:
    st.set_page_config(
        page_title="Loan default review",
        layout="wide",
        initial_sidebar_state="auto",
    )
    st.markdown(THEME_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------- data access


@st.cache_data(show_spinner=False, ttl=600)
def load_features() -> pd.DataFrame:
    fallback = DATA / "features" / "loan_features.csv"
    return read_table("analytics.loan_features", fallback)


@st.cache_data(show_spinner=False, ttl=600)
def load_output(name: str) -> pd.DataFrame | None:
    path = next(ARTIFACTS.rglob(name), None)
    if path is None:
        return None
    return read_table(f"analytics.{Path(name).stem}", path)


@st.cache_data(show_spinner=False, ttl=600)
def load_scores() -> pd.DataFrame:
    return read_table("analytics.model_scores", ARTIFACTS / "metrics" / "model_scores.csv")


@st.cache_data(show_spinner=False, ttl=600)
def mart_counts() -> pd.DataFrame:
    try:
        return db_table_counts()
    except Exception:
        rows = []
        for path in sorted(CLEAN.glob("*.csv")):
            with path.open("rb") as handle:
                rows.append({"table": path.stem, "rows": sum(1 for _ in handle) - 1})
        return pd.DataFrame(rows)


@st.cache_data(show_spinner=False, ttl=600)
def transaction_summary() -> dict:
    row = query(
        "SELECT MIN(date_id) AS min_date, MAX(date_id) AS max_date, "
        "COUNT(*) AS total, COUNT(DISTINCT account_id) AS accounts "
        "FROM mart.fact_transactions"
    ).iloc[0]
    return {
        "min_date": int(row["min_date"]),
        "max_date": int(row["max_date"]),
        "total": int(row["total"]),
        "accounts": int(row["accounts"]),
    }


@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load(MODEL_DIR / "best_GradientBoosting_baseline_tuned.pkl")


@st.cache_data(show_spinner="Checking transaction cutoff...", ttl=600)
def leakage_audit() -> pd.DataFrame:
    if snowflake_enabled():
        return snowflake_leakage()
    return query(
        "SELECT f.loan_id, f.account_id, f.loan_date_id AS cutoff, f.n_txn, "
        "COUNT(t.date_id) FILTER (WHERE t.date_id < f.loan_date_id) AS used, "
        "COUNT(t.date_id) AS total "
        "FROM analytics.loan_features f "
        "LEFT JOIN mart.fact_transactions t ON t.account_id = f.account_id "
        "GROUP BY f.loan_id, f.account_id, f.loan_date_id, f.n_txn "
        "ORDER BY f.loan_id"
    )


# ------------------------------------------------------------------- helpers


def page_header(eyebrow: str, title: str, lede: str) -> None:
    st.markdown(
        f'<div class="eyebrow">{eyebrow}</div><h1>{title}</h1><p class="lede">{lede}</p>',
        unsafe_allow_html=True,
    )


def section(title: str, note: str = "") -> None:
    extra = f"<em>{note}</em>" if note else ""
    st.markdown(
        f'<div class="section"><span>{title}</span>{extra}</div>',
        unsafe_allow_html=True,
    )


def kpi_row(items: list[tuple[str, str, str]]) -> None:
    columns = st.columns(len(items))
    for column, (label, value, note) in zip(columns, items):
        column.markdown(
            f'<div class="kpi"><div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div>'
            f'<div class="kpi-note">{note}</div></div>',
            unsafe_allow_html=True,
        )


def note(text: str, label: str = "Finding", kind: str = "") -> None:
    st.markdown(
        f'<div class="note {kind}"><b>{label}</b>{text}</div>',
        unsafe_allow_html=True,
    )


def show(chart) -> None:
    st.altair_chart(chart, width="stretch")


def ref_rule(value: float, axis: str = "x", color: str = INK, dash=(4, 3), width: int = 1):
    field = "x" if axis == "x" else "y"
    rule = alt.Chart(pd.DataFrame({field: [value]})).mark_rule(
        color=color,
        strokeWidth=width,
        **({"strokeDash": list(dash)} if dash else {}),
    )
    return rule.encode(x="x:Q") if axis == "x" else rule.encode(y="y:Q")


def grid(frame: pd.DataFrame, config: dict | None = None) -> None:
    st.dataframe(frame, hide_index=True, width="stretch", column_config=config)


def score_config(*names: str, places: int = 3) -> dict:
    return {name: st.column_config.NumberColumn(format=f"%.{places}f") for name in names}


def percent(value: float) -> str:
    return f"{value:.1%}"


def money(value: float) -> str:
    return f"{value:,.0f} CZK"


def parse_itemset(value: str) -> str:
    text = str(value).strip()
    if text.startswith("frozenset(") and text.endswith(")"):
        text = text[len("frozenset(") : -1]
    try:
        parsed = literal_eval(text)
    except (ValueError, SyntaxError):
        parsed = text
    if isinstance(parsed, (set, frozenset, list, tuple)):
        items = sorted(str(item) for item in parsed)
    else:
        items = [str(parsed)]

    labels = {
        "type_PRIJEM": "Incoming money",
        "type_VYDAJ": "Outgoing money",
        "type_VYBER": "Cash withdrawal",
        "high_amount": "Large transaction",
        "negative_balance": "Negative balance",
        "default": "Loan default",
    }
    return ", ".join(labels.get(item, item.replace("_", " ")) for item in items)


def model_columns(model, features: pd.DataFrame) -> list[str]:
    stored = getattr(model, "feature_names_in_", None)
    if stored is not None:
        return [str(column) for column in stored]
    return [column for column in features.columns if column not in DROP]


def median_row(features: pd.DataFrame, columns: list[str]) -> pd.Series:
    base = features.drop(columns=[column for column in DROP if column in features.columns])
    missing = [column for column in columns if column not in base.columns]
    if missing:
        raise KeyError(f"Missing model features: {missing}")
    return base[columns].median(numeric_only=True)


def applicant_row(base: pd.Series, values: dict) -> pd.Series:
    row = base.copy()
    row["amount"] = float(values["amount"])
    row["duration"] = float(values["duration"])
    row["payments"] = float(values["amount"]) / float(values["duration"])
    row["balance_at_loan"] = float(values["balance_at_loan"])
    row["min_balance"] = float(values["min_balance"])
    row["txn_per_month"] = float(values["txn_per_month"])
    return row


def predict(model, row: pd.Series, columns: list[str]) -> float:
    frame = pd.DataFrame([row[columns].astype(float)], columns=columns)
    return float(model.predict_proba(frame)[0, 1])


def scored_frame(scores: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    return scores.merge(features[["loan_id", "amount"]], on="loan_id", how="left")


def cutoff_stats(data: pd.DataFrame, cutoff: float) -> dict:
    if data.empty:
        return {
            "flagged": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
            "precision": 0.0,
            "recall": 0.0,
            "exposure": 0.0,
            "defaults": 0,
            "total": 0,
        }
    flagged = data["prob_default"] >= cutoff
    is_default = data["target"] == 1
    tp = int((flagged & is_default).sum())
    fp = int((flagged & ~is_default).sum())
    fn = int((~flagged & is_default).sum())
    tn = int((~flagged & ~is_default).sum())
    return {
        "flagged": int(flagged.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "exposure": float(data.loc[flagged, "amount"].sum()),
        "defaults": int(is_default.sum()),
        "total": len(data),
    }


def score_distribution(scores: pd.DataFrame, cutoff: float, applicant: float | None = None):
    bins = np.arange(0, 1.02, 0.02)
    frame = pd.DataFrame(
        {
            "Low": bins[:-1],
            "High": bins[1:],
            "Repaid": np.histogram(scores.loc[scores["target"] == 0, "prob_default"], bins=bins)[0],
            "Default": np.histogram(scores.loc[scores["target"] == 1, "prob_default"], bins=bins)[0],
        }
    )
    long = frame.melt(
        id_vars=["Low", "High"],
        value_vars=["Repaid", "Default"],
        var_name="Outcome",
        value_name="Loans",
    )
    bars = alt.Chart(long).mark_bar().encode(
        x=alt.X("Low:Q", title="Out-of-fold probability"),
        x2="High:Q",
        y=alt.Y("Loans:Q", title="Loans", stack="zero"),
        color=alt.Color(
            "Outcome:N",
            scale=alt.Scale(domain=["Repaid", "Default"], range=[ACCENT, RISK]),
            title=None,
        ),
        tooltip=[
            alt.Tooltip("Low:Q", format=".2f", title="Score from"),
            alt.Tooltip("High:Q", format=".2f", title="to"),
            "Outcome",
            alt.Tooltip("Loans:Q", format=","),
        ],
    )
    top = int((frame["Repaid"] + frame["Default"]).max())
    shade = (
        alt.Chart(pd.DataFrame({"x": [cutoff], "x2": [1.0], "y": [0], "y2": [top]}))
        .mark_rect(color=RISK, opacity=0.07)
        .encode(x="x:Q", x2="x2:Q", y="y:Q", y2="y2:Q")
    )
    layers = [shade, bars, ref_rule(cutoff)]
    if applicant is not None:
        layers.append(ref_rule(applicant, color=RISK, dash=(), width=2))
    return alt.layer(*layers).properties(height=220).interactive()


def row_count_chart(
    frame: pd.DataFrame, label="Table", value="Rows", color=None, log=True
):
    values = frame[value].astype(float)
    if log:
        scale = alt.Scale(
            type="log",
            domain=[max(float(values.min()) * 0.6, 0.5), float(values.max()) * 5.0],
        )
        x_title = f"{value} (log scale)"
    else:
        scale = alt.Scale(domain=[0, float(values.max()) * 1.18])
        x_title = value
    base = alt.Chart(frame).encode(
        y=alt.Y(
            f"{label}:N",
            sort="-x",
            title=None,
            axis=alt.Axis(labelLimit=300, labelOverlap=False),
        ),
        x=alt.X(f"{value}:Q", scale=scale, title=x_title),
        tooltip=[label, alt.Tooltip(f"{value}:Q", format=",")],
    )
    paint = alt.value(ACCENT)
    if color:
        categories = frame[color].dropna().unique().tolist()
        palette = [ACCENT, RISK, GOOD, MUTED, INK]
        paint = alt.Color(
            f"{color}:N",
            scale=alt.Scale(
                domain=categories,
                range=(palette * len(categories))[: len(categories)],
            ),
            title=None,
        )
    chart = (base.mark_circle(size=110) if log else base.mark_bar()).encode(color=paint)
    text = alt.Chart(frame).mark_text(align="left", dx=6, color=INK, fontSize=9, clip=True).encode(
        y=alt.Y(f"{label}:N", sort="-x", title=None),
        x=alt.X(f"{value}:Q", scale=scale),
        text=alt.Text(f"{value}:Q", format=","),
    )
    return (chart + text).properties(height=max(240, len(frame) * 30)).interactive()


def gains_figure(scores: pd.DataFrame, cutoff: float):
    ordered = scores.sort_values("prob_default", ascending=False).reset_index(drop=True)
    share = np.arange(1, len(ordered) + 1) / len(ordered)
    captured = ordered["target"].cumsum() / ordered["target"].sum()
    frame = pd.DataFrame({"Share reviewed": share, "Defaults captured": captured})
    line = alt.Chart(frame).mark_line(color=ACCENT, strokeWidth=2).encode(
        x=alt.X("Share reviewed:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("Defaults captured:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1.02])),
        tooltip=[
            alt.Tooltip("Share reviewed:Q", format=".1%"),
            alt.Tooltip("Defaults captured:Q", format=".1%"),
        ],
    )
    diagonal = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(
        color=RULE, strokeDash=[4, 3]
    ).encode(x="x:Q", y="y:Q")
    layers = [line, diagonal]
    flagged = int((ordered["prob_default"] >= cutoff).sum())
    if flagged:
        point = pd.DataFrame(
            {
                "Share reviewed": [share[flagged - 1]],
                "Defaults captured": [captured.iloc[flagged - 1]],
                "Label": [f"cutoff {cutoff:.2f}"],
            }
        )
        layers.append(
            alt.Chart(point).mark_point(color=RISK, size=70, filled=True).encode(
                x="Share reviewed:Q", y="Defaults captured:Q"
            )
        )
        layers.append(
            alt.Chart(point).mark_text(dx=8, dy=-8, color=RISK, fontSize=10).encode(
                x="Share reviewed:Q", y="Defaults captured:Q", text="Label:N"
            )
        )
    return alt.layer(*layers).properties(height=300).interactive()


def quintile_figure(features: pd.DataFrame, column: str, label: str):
    labels = ["Q1", "Q2", "Q3", "Q4", "Q5"]
    bands = pd.qcut(features[column], 5, labels=labels, duplicates="drop")
    rates = features.groupby(bands, observed=True)["target"].mean()
    counts = features.groupby(bands, observed=True).agg(
        Loans=("target", "size"), Defaults=("target", "sum")
    )
    counts.index.name = "Quintile"
    frame = rates.rename_axis("Quintile").reset_index(name="Default rate").merge(
        counts.reset_index(), on="Quintile"
    )
    frame["Quintile"] = frame["Quintile"].astype(str)
    top = max(float(rates.max()) * 1.25, 0.05)
    base_rate = float(features["target"].mean())
    base = alt.Chart(frame).mark_bar(size=28).encode(
        x=alt.X("Quintile:N", title=f"{label} quintile", sort=labels),
        y=alt.Y("Default rate:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, top])),
        color=alt.condition(alt.datum.Quintile == "Q1", alt.value(RISK), alt.value(ACCENT)),
        tooltip=[
            "Quintile",
            alt.Tooltip("Default rate:Q", format=".1%"),
            "Loans",
            "Defaults",
        ],
    )
    text = alt.Chart(frame).mark_text(dy=-6, fontSize=9).encode(
        x="Quintile:N", y="Default rate:Q", text=alt.Text("Default rate:Q", format=".1%")
    )
    return (base + text + ref_rule(base_rate, axis="y", color=MUTED, dash=(3, 3))).properties(
        height=280
    ).interactive()


def heatmap_figure(table: pd.DataFrame, labels: list[str], row_labels: list[str]):
    frame = table.copy()
    frame.index = row_labels
    frame = frame.reset_index().rename(columns={"index": "Run"})
    long = frame.melt(id_vars="Run", var_name="column", value_name="Score")
    long["Metric"] = long["column"].map(dict(zip(table.columns, labels)))
    base = alt.Chart(long).encode(
        x=alt.X("Metric:N", sort=labels, title=None),
        y=alt.Y("Run:N", sort=row_labels, title=None),
    )
    rect = base.mark_rect().encode(
        color=alt.Color(
            "Score:Q",
            scale=alt.Scale(domain=[0, 1], range=["#f7f4ec", "#c9d6de", ACCENT]),
            title="Score",
        ),
        tooltip=["Run", "Metric", alt.Tooltip("Score:Q", format=".3f")],
    )
    text = base.mark_text(fontSize=9).encode(
        text=alt.Text("Score:Q", format=".2f"),
        color=alt.condition(alt.datum.Score > 0.62, alt.value("white"), alt.value(INK)),
    )
    return (rect + text).properties(height=max(320, len(table) * 30)).interactive()


def driver_figure(labels: list[str], deltas: list[float]):
    order = np.argsort(deltas)
    frame = pd.DataFrame(
        {
            "Driver": [labels[index] for index in order],
            "Delta": [deltas[index] * 100 for index in order],
        }
    )
    bars = alt.Chart(frame).mark_bar(size=20).encode(
        y=alt.Y("Driver:N", sort=None, title=None),
        x=alt.X("Delta:Q", title="Change in default probability (pp)"),
        color=alt.condition(alt.datum.Delta > 0, alt.value(RISK), alt.value(GOOD)),
        tooltip=["Driver", alt.Tooltip("Delta:Q", format="+.1f")],
    )
    return (bars + ref_rule(0, color=RULE, dash=())).properties(height=280).interactive()


def leakage_figure(used: int, excluded: int):
    frame = pd.DataFrame(
        {
            "Segment": ["Used", "Excluded"],
            "Count": [used, excluded],
            "Label": [f"{used:,} used", f"{excluded:,} excluded"],
        }
    )
    order = ["Used", "Excluded"]
    base = alt.Chart(frame).mark_bar(height=30).encode(
        y=alt.Y("Segment:N", title=None, sort=order),
        x=alt.X(
            "Count:Q",
            title="Transactions on the account, before and after the loan date",
            scale=alt.Scale(domain=[0, max(used, excluded) * 1.35]),
        ),
        color=alt.Color(
            "Segment:N",
            scale=alt.Scale(domain=order, range=[ACCENT, RULE]),
            legend=None,
        ),
        tooltip=["Segment", alt.Tooltip("Count:Q", format=",")],
    )
    text = alt.Chart(frame).mark_text(align="left", dx=6, color=INK, fontSize=10).encode(
        x="Count:Q", y=alt.Y("Segment:N", sort=order), text="Label:N"
    )
    return (base + text).properties(height=130)


def cluster_rate_figure(profiles: pd.DataFrame, base_rate: float):
    names = METHOD_NAMES
    frame = profiles.copy()
    frame["method"] = pd.Categorical(
        frame["method"], METHOD_ORDER, ordered=True
    )
    frame = frame.sort_values(["method", "cluster"]).reset_index(drop=True)
    frame["Label"] = [
        f"{names.get(row.method, row.method)} "
        + ("noise" if row.cluster == -1 else str(row.cluster))
        for row in frame.itertuples()
    ]
    frame["Rate"] = frame["default_rate"] * 100
    labels = frame["Label"].tolist()
    bars = alt.Chart(frame).mark_bar(size=30).encode(
        x=alt.X(
            "Label:N",
            sort=labels,
            scale=alt.Scale(domain=labels),
            axis=alt.Axis(labelAngle=-30),
            title=None,
        ),
        y=alt.Y(
            "Rate:Q",
            axis=alt.Axis(format=".0f", title="Default rate (%)"),
            scale=alt.Scale(domain=[0, max(float(frame["Rate"].max()) * 1.35, 5)]),
        ),
        color=alt.condition(alt.datum.Rate > base_rate * 100, alt.value(RISK), alt.value(ACCENT)),
        tooltip=["Label", alt.Tooltip("Rate:Q", format=".1f")],
    )
    text = alt.Chart(frame).mark_text(dy=-6, fontSize=9).encode(
        x=alt.X("Label:N", sort=labels, scale=alt.Scale(domain=labels)),
        y="Rate:Q",
        text=alt.Text("Rate:Q", format=".1f"),
    )
    rule = ref_rule(base_rate * 100, axis="y", color=MUTED, dash=(3, 3))
    return (bars + rule + text).properties(height=300).interactive()


def cluster_map_figures(features: pd.DataFrame, labels: pd.DataFrame):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    matrix = StandardScaler().fit_transform(features[CLUSTER_FEATURES].astype(float))
    pca = PCA(n_components=2, random_state=42)
    points = pca.fit_transform(matrix)
    joined = labels.set_index("loan_id").reindex(features["loan_id"])
    palette = ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f"]

    figures = []
    for key, title in METHOD_NAMES.items():
        values = joined[key].fillna(-1).astype(int).to_numpy()
        frame = pd.DataFrame(
            {
                "PC1": points[:, 0],
                "PC2": points[:, 1],
                "loan_id": features["loan_id"].to_numpy(),
                "Cluster": ["noise" if value == -1 else str(value) for value in values],
            }
        )
        unique = sorted(set(values))
        order = (["noise"] if -1 in unique else []) + [
            str(value) for value in unique if value != -1
        ]
        named = [value for value in order if value != "noise"]
        palette_map = {value: palette[index % len(palette)] for index, value in enumerate(named)}
        range_ = [MUTED if value == "noise" else palette_map[value] for value in order]
        chart = alt.Chart(frame).mark_circle(size=18, opacity=0.7).encode(
            x=alt.X("PC1:Q", title=f"PC 1 ({pca.explained_variance_ratio_[0]:.0%})"),
            y=alt.Y("PC2:Q", title=f"PC 2 ({pca.explained_variance_ratio_[1]:.0%})"),
            color=alt.Color(
                "Cluster:N",
                scale=alt.Scale(domain=order, range=range_),
                legend=alt.Legend(title=None, orient="bottom", columns=3),
            ),
            tooltip=["loan_id", "Cluster"],
        ).properties(height=300).interactive()
        centers = frame.groupby("Cluster", as_index=False)[["PC1", "PC2"]].median()
        text = alt.Chart(centers).mark_text(
            fontSize=9, fontWeight="bold", color=INK, dx=6
        ).encode(x="PC1:Q", y="PC2:Q", text="Cluster:N")
        figures.append((title, chart + text))
    return figures


def roc_pr_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.metrics import precision_recall_curve, roc_curve

    scores = load_scores()
    sources = [("Gradient Boosting", scores["prob_default"], scores["target"])]
    advanced = load_output("advanced_scores.csv")
    if advanced is not None and len(advanced):
        sources += [
            ("MLP", advanced["mlp_prob_default"], advanced["target"]),
            ("CBA rules", advanced["cba_score"], advanced["target"]),
        ]

    roc_rows, pr_rows = [], []
    for name, probability, target in sources:
        fpr, tpr, _ = roc_curve(target, probability)
        precision, recall, _ = precision_recall_curve(target, probability)
        roc_rows += [
            {"Model": name, "False positive rate": x, "True positive rate": y}
            for x, y in zip(fpr, tpr)
        ]
        pr_rows += [
            {"Model": name, "Recall": x, "Precision": y}
            for x, y in zip(recall, precision)
        ]
    return pd.DataFrame(roc_rows), pd.DataFrame(pr_rows)


def confusion_figure(data: pd.DataFrame, cutoff: float, stats: dict | None = None):
    stats = stats or cutoff_stats(data, cutoff)
    frame = pd.DataFrame(
        {
            "Actual": ["Default", "Default", "Repaid", "Repaid"],
            "Flagged": ["Flagged", "Not flagged", "Flagged", "Not flagged"],
            "Count": [stats["tp"], stats["fn"], stats["fp"], stats["tn"]],
        }
    )
    base = alt.Chart(frame).encode(
        x=alt.X("Flagged:N", sort=["Flagged", "Not flagged"], title=None),
        y=alt.Y("Actual:N", sort=["Default", "Repaid"], title=None),
    )
    rect = base.mark_rect().encode(
        color=alt.Color("Count:Q", scale=alt.Scale(scheme="blues"), title="Loans"),
        tooltip=["Actual", "Flagged", "Count"],
    )
    text = base.mark_text(fontSize=16, fontWeight="bold").encode(
        text="Count:Q",
        color=alt.condition(
            alt.datum.Count > frame["Count"].max() / 2,
            alt.value("white"),
            alt.value(INK),
        ),
    )
    return (rect + text).properties(height=200).interactive()


def sidebar(features: pd.DataFrame) -> None:
    st.sidebar.markdown('<div class="brand">Loan default review</div>', unsafe_allow_html=True)
    st.sidebar.markdown(
        '<div class="brand-sub">PKDD\'99 banking records<br>Pre-loan behaviour only</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown('<div class="side-label">Source</div>', unsafe_allow_html=True)
    st.sidebar.markdown(
        f'<div class="side-line"><span>Analytics</span><span>{data_source_label()}</span></div>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown('<div class="side-label">Corpus</div>', unsafe_allow_html=True)
    defaults = int(features["target"].sum())
    rows = [
        ("Applications", f"{len(features):,}"),
        ("Defaults", f"{defaults:,}"),
        ("Base rate", percent(features["target"].mean())),
    ]
    for label, value in rows:
        st.sidebar.markdown(
            f'<div class="side-line"><span>{label}</span><span>{value}</span></div>',
            unsafe_allow_html=True,
        )
