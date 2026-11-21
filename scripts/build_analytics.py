"""Load frozen model and analysis outputs into the local DuckDB analytics schema."""
from pathlib import Path

import duckdb
import pandas as pd

from etl_pipeline import LOCAL_DB, write_tables

ROOT = Path(__file__).resolve().parent.parent
DB = LOCAL_DB
ARTIFACTS = ROOT / "artifacts"

TABLES = {
    "model_scores": ARTIFACTS / "metrics" / "model_scores.csv",
    "cv_comparison": ARTIFACTS / "metrics" / "cv_comparison.csv",
    "metrics_comparison": ARTIFACTS / "metrics" / "metrics_comparison.csv",
    "tuning_results": ARTIFACTS / "metrics" / "tuning_results.csv",
    "capacity_model_comparison": ARTIFACTS / "metrics" / "capacity_model_comparison.csv",
    "capacity_model_fold_results": ARTIFACTS / "metrics" / "capacity_model_fold_results.csv",
    "loan_segments": ARTIFACTS / "segments" / "loan_segments.csv",
    "segment_profiles": ARTIFACTS / "segments" / "segment_profiles.csv",
    "pattern_itemsets": ARTIFACTS / "rules" / "pattern_itemsets.csv",
    "pattern_rules": ARTIFACTS / "rules" / "pattern_rules.csv",
    "loan_risk_dashboard": ARTIFACTS / "features" / "loan_risk_dashboard.csv",
    "advanced_metrics": ARTIFACTS / "metrics" / "advanced_metrics.csv",
    "advanced_scores": ARTIFACTS / "metrics" / "advanced_scores.csv",
    "cba_rules": ARTIFACTS / "rules" / "cba_rules.csv",
    "cluster_labels": ARTIFACTS / "segments" / "cluster_labels.csv",
    "cluster_profiles": ARTIFACTS / "segments" / "cluster_profiles.csv",
    "clustering_comparison": ARTIFACTS / "metrics" / "clustering_comparison.csv",
    "clustering_tuning": ARTIFACTS / "metrics" / "clustering_tuning.csv",
    "clustering_stability": ARTIFACTS / "metrics" / "clustering_stability.csv",
}


def main() -> None:
    missing = [f"{table} ({path})" for table, path in TABLES.items() if not path.exists()]
    if missing:
        raise SystemExit("Missing analytics outputs:\n  " + "\n  ".join(missing))
    con = duckdb.connect(str(DB))
    con.execute("CREATE SCHEMA IF NOT EXISTS analytics")
    write_tables(con, "analytics", {table: pd.read_csv(path) for table, path in TABLES.items()})
    for table in TABLES:
        rows = con.execute(f"SELECT count(*) FROM analytics.{table}").fetchall()[0][0]
        print(f"{table}: {rows} rows")
    con.execute("CHECKPOINT")
    con.close()


if __name__ == "__main__":
    main()
