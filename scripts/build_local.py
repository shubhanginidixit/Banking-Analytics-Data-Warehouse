"""Rebuild the whole local warehouse: raw CSV -> mart -> feature mart -> analytics."""
import sys
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from build_advanced_models import main as build_advanced
from build_analytics import main as build_analytics
from etl_pipeline import LOCAL_DB, main as run_etl
from tools.capacity_aware_model import main as build_capacity
from tools.pattern_mining import main as build_patterns

DB = LOCAL_DB
FEATURE_SQL = ROOT / "sql" / "local" / "03_feature_mart.sql"
FEATURES_CSV = ROOT / "data" / "features" / "loan_features.csv"


def validate_feature_export(con: duckdb.DuckDBPyConnection) -> None:
    """The CSV is what notebooks and Snowflake upload; it must match the warehouse."""
    frame = pd.read_csv(FEATURES_CSV)
    db = con.execute("SELECT * FROM analytics.loan_features ORDER BY loan_id").fetchdf()

    extra = set(db.columns) - set(frame.columns)
    missing = set(frame.columns) - set(db.columns)
    if extra != {"date", "loan_date_id"} or missing != {"loan_date"}:
        raise AssertionError(
            f"feature export columns drifted: extra={sorted(extra)}, missing={sorted(missing)}"
        )

    expected = db.drop(columns=["date", "loan_date_id"]).assign(
        loan_date=db["date"].astype("int64")
    )
    expected["parsed_date"] = pd.to_datetime(expected["parsed_date"])
    frame["parsed_date"] = pd.to_datetime(frame["parsed_date"])
    expected = expected[frame.columns]

    pd.testing.assert_frame_equal(frame, expected, check_dtype=False)
    print(f"feature CSV matches DB: {len(frame)} rows")


def validate() -> None:
    con = duckdb.connect(str(DB), read_only=True)
    checks = [
        ("fact_loans", "SELECT count(*) FROM mart.fact_loans", lambda n: n > 0),
        ("fact_transactions", "SELECT count(*) FROM mart.fact_transactions", lambda n: n > 0),
        ("dim_customer", "SELECT count(*) FROM mart.dim_customer", lambda n: n > 0),
        ("loan features", "SELECT count(*) FROM analytics.loan_features", lambda n: n > 0),
        ("defaults", "SELECT sum(target) FROM analytics.loan_features", lambda n: n > 0),
        (
            "feature rows match fact_loans",
            "SELECT (SELECT count(*) FROM analytics.loan_features) "
            "- (SELECT count(*) FROM mart.fact_loans)",
            lambda n: n == 0,
        ),
        (
            "features with post-loan rows",
            "SELECT count(*) FROM analytics.loan_features f "
            "WHERE f.n_txn <> ("
            "  SELECT count(*) FROM mart.fact_transactions t "
            "  WHERE t.account_id = f.account_id AND t.date_id < f.loan_date_id"
            ")",
            lambda n: n == 0,
        ),
    ]
    failures = []
    for label, sql, valid in checks:
        value = con.execute(sql).fetchone()[0]
        print(f"{label}: {value}")
        if not valid(value):
            failures.append(f"{label}: {value}")
    try:
        validate_feature_export(con)
    except AssertionError as exc:
        print(exc)
        failures.append("feature CSV matches DB")
    con.close()
    if failures:
        raise SystemExit("Validation failed: " + ", ".join(failures))


def export_features() -> None:
    """Keep the Snowflake upload and notebook input in step with DuckDB."""
    FEATURES_CSV.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB), read_only=True)
    frame = con.execute(
        "SELECT loan_id, account_id, CAST(date AS INTEGER) AS loan_date, amount, duration, payments, "
        "status, parsed_date, target, client_id, n_txn, n_deposit, n_withdrawal, "
        "sum_deposit, sum_withdrawal, avg_amount, std_amount, avg_balance, min_balance, "
        "max_balance, balance_at_loan, months_active, txn_per_month, gender, age, "
        "district_id, gender_bin "
        "FROM analytics.loan_features ORDER BY loan_id"
    ).fetchdf()
    con.close()
    frame.to_csv(FEATURES_CSV, index=False)
    print(f"Exported features: {FEATURES_CSV}")


def main() -> None:
    run_etl()

    print("Building feature mart...")
    con = duckdb.connect(str(DB))
    con.execute(FEATURE_SQL.read_text(encoding="utf-8"))
    con.execute("CHECKPOINT")
    con.close()
    export_features()

    print("Training advanced models...")
    build_advanced()

    print("Running capacity-aware challenger...")
    build_capacity()

    print("Mining patterns...")
    build_patterns()

    print("Loading saved analytics outputs...")
    build_analytics()

    print("Validating...")
    validate()


if __name__ == "__main__":
    main()
