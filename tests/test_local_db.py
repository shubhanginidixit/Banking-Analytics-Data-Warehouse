from pathlib import Path

import duckdb
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "banking_dw.duckdb"


@pytest.fixture(scope="module")
def con():
    if not DB.exists():
        pytest.skip("Local DuckDB not built yet")
    connection = duckdb.connect(str(DB), read_only=True)
    yield connection
    connection.close()


def test_mart_counts(con):
    loans, transactions, customers = con.execute(
        "SELECT (SELECT count(*) FROM mart.fact_loans), "
        "(SELECT count(*) FROM mart.fact_transactions), "
        "(SELECT count(*) FROM mart.dim_customer)"
    ).fetchone()
    assert loans > 0
    assert transactions > 0
    assert customers > 0


def test_feature_mart_matches_fact_loans(con):
    rows, defaults = con.execute(
        "SELECT count(*), sum(target) FROM analytics.loan_features"
    ).fetchone()
    fact_rows, fact_defaults = con.execute(
        "SELECT count(*), count(*) FILTER (WHERE status_code IN ('B', 'D')) "
        "FROM mart.fact_loans"
    ).fetchone()
    assert (rows, defaults) == (fact_rows, fact_defaults)


def test_feature_counts_use_only_pre_loan_transactions(con):
    mismatches = con.execute(
        """
        SELECT count(*)
        FROM analytics.loan_features f
        WHERE f.n_txn <> (
            SELECT count(*)
            FROM mart.fact_transactions t
            WHERE t.account_id = f.account_id
              AND t.date_id < f.loan_date_id
        )
        """
    ).fetchone()[0]
    assert mismatches == 0


def test_gender_bin(con):
    rows = con.execute(
        "SELECT gender, gender_bin, count(*) FROM analytics.loan_features GROUP BY 1, 2"
    ).fetchall()
    total = con.execute("SELECT count(*) FROM analytics.loan_features").fetchone()[0]
    assert sum(count for _, _, count in rows) == total
    assert all(
        (gender == "Female") == (gender_bin == 1) for gender, gender_bin, _ in rows
    )


def test_analytics_outputs(con):
    features = con.execute("SELECT count(*) FROM analytics.loan_features").fetchone()[0]
    scores = con.execute("SELECT count(*) FROM analytics.model_scores").fetchone()[0]
    segments = con.execute("SELECT count(*) FROM analytics.loan_segments").fetchone()[0]
    configurations = con.execute("SELECT count(*) FROM analytics.cv_comparison").fetchone()[0]
    assert scores == features
    assert segments == features
    assert configurations > 0


def test_advanced_outputs(con):
    features = con.execute("SELECT count(*) FROM analytics.loan_features").fetchone()[0]
    metrics = con.execute("SELECT count(*) FROM analytics.advanced_metrics").fetchone()[0]
    scores = con.execute("SELECT count(*) FROM analytics.advanced_scores").fetchone()[0]
    rules = con.execute("SELECT count(*) FROM analytics.cba_rules").fetchone()[0]
    labels = con.execute(
        "SELECT count(*), count(kmeans), count(kmedoids), count(agnes), count(dbscan) "
        "FROM analytics.cluster_labels"
    ).fetchone()
    columns = [row[0] for row in con.execute("DESCRIBE analytics.cluster_labels").fetchall()]
    assert metrics > 0
    assert scores == features
    assert rules >= 1
    assert labels == (features, features, features, features, features)
    assert columns == ["loan_id", "kmeans", "kmedoids", "agnes", "dbscan"]


def test_clustering_outputs(con):
    features = con.execute("SELECT count(*) FROM analytics.loan_features").fetchone()[0]
    comparison = con.execute(
        "SELECT * FROM analytics.clustering_comparison"
    ).fetchdf()
    assert len(comparison) == 4
    assert set(comparison["method"]) == {"kmeans", "kmedoids", "agnes", "dbscan"}
    assert comparison["recommended"].sum() == 1
    assert comparison.loc[comparison["recommended"], "method"].iat[0] == "kmeans"

    kmeans = comparison.set_index("method").loc["kmeans"]
    assert 0 < kmeans["silhouette"] < 1
    assert kmeans["davies_bouldin"] > 0
    assert kmeans["calinski_harabasz"] > 0
    assert kmeans["coverage"] == 1.0
    assert kmeans["noise_share"] == 0.0

    dbscan = comparison.set_index("method").loc["dbscan"]
    assert dbscan["n_clusters"] == 1
    assert dbscan["noise_share"] > 0
    assert pd.isna(dbscan["silhouette"])

    profiles = con.execute("SELECT * FROM analytics.cluster_profiles").fetchdf()
    assert len(profiles) == 8
    assert set(profiles["method"]) == {"kmeans", "kmedoids", "agnes", "dbscan"}
    for _, group in profiles.groupby("method"):
        assert group["size"].sum() == features
    kmeans_profiles = profiles[profiles["method"] == "kmeans"]
    assert set(kmeans_profiles["cluster"]) == {0, 1}
    assert kmeans_profiles["default_rate"].between(0, 1).all()

    stability = con.execute("SELECT * FROM analytics.clustering_stability").fetchdf()
    assert len(stability) == 4
    assert (stability["n_runs"] == 10).all()
    assert stability["mean_ari"].between(0, 1).all()
    assert (stability["std_ari"] >= 0).all()
    assert (stability["min_ari"] <= stability["mean_ari"]).all()
    assert (stability["mean_ari"] <= stability["max_ari"]).all()
