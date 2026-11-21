"""Train the advanced models and clusterings that are not in the original notebooks.

Outputs:
  artifacts/metrics/advanced_metrics.csv
  artifacts/rules/cba_rules.csv
  artifacts/models/mlp_classifier.pkl
  artifacts/segments/cluster_labels.csv
  artifacts/segments/cluster_profiles.csv

Run with: venv\\Scripts\\python.exe scripts\\build_advanced_models.py
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    average_precision_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    f1_score,
    pairwise_distances,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from etl_pipeline import LOCAL_DB

ROOT = Path(__file__).resolve().parent.parent
DB = LOCAL_DB
ARTIFACTS = ROOT / "artifacts"
SEED = 42

DROP = {
    "loan_id",
    "account_id",
    "loan_date_id",
    "client_id",
    "district_id",
    "status",
    "date",
    "parsed_date",
    "gender",
    "target",
}

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
CLUSTER_METHODS = ["kmeans", "kmedoids", "agnes", "dbscan"]
MIN_CLUSTER_SHARE = 0.05

BIN_SPECS = {
    "amount": ([-np.inf, 50_000, 100_000, np.inf], ["low", "mid", "high"]),
    "duration": ([-np.inf, 12, 24, np.inf], ["short", "mid", "long"]),
    "payments": ([-np.inf, 3_000, 7_000, np.inf], ["low", "mid", "high"]),
    "n_txn": ([-np.inf, 20, 100, np.inf], ["few", "some", "many"]),
    "n_deposit": ([-np.inf, 10, 50, np.inf], ["few", "some", "many"]),
    "n_withdrawal": ([-np.inf, 10, 50, np.inf], ["few", "some", "many"]),
    "avg_balance": ([-np.inf, 0, 20_000, 60_000, np.inf], ["negative", "low", "mid", "high"]),
    "min_balance": ([-np.inf, 0, 10_000, 40_000, np.inf], ["negative", "low", "mid", "high"]),
    "balance_at_loan": ([-np.inf, 0, 20_000, 60_000, np.inf], ["negative", "low", "mid", "high"]),
    "months_active": ([-np.inf, 12, 36, np.inf], ["new", "mid", "old"]),
    "txn_per_month": ([-np.inf, 1, 5, np.inf], ["low", "mid", "high"]),
    "age": ([-np.inf, 30, 45, 60, np.inf], ["young", "mid", "senior", "old"]),
    "gender_bin": ({0: "male", 1: "female"}, None),
}


def load_features() -> pd.DataFrame:
    if not DB.exists():
        raise FileNotFoundError(f"Local database not built: {DB}")
    con = duckdb.connect(str(DB), read_only=True)
    frame = con.execute("SELECT * FROM analytics.loan_features").fetchdf()
    con.close()
    return frame


def scores_at_threshold(y_true: pd.Series, scores: np.ndarray) -> dict:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(
        precision[:-1] + recall[:-1], 1e-12
    )
    threshold = float(thresholds[int(np.nanargmax(f1))])
    predicted = scores >= threshold
    return {
        "threshold": threshold,
        "accuracy": float((predicted == y_true).mean()),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "average_precision": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
    }


def run_mlp(x: pd.DataFrame, y: pd.Series) -> tuple[dict, pd.DataFrame, object]:
    model = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(32, 16),
            alpha=0.001,
            learning_rate_init=0.01,
            max_iter=3000,
            early_stopping=True,
            n_iter_no_change=40,
            random_state=SEED,
        ),
    )
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = cross_val_predict(model, x, y, cv=folds, method="predict_proba")[:, 1]
    metrics = {"model": "MLP", **scores_at_threshold(y, scores)}
    model.fit(x, y)
    return metrics, pd.DataFrame({"prob_default": scores}), model


def binned_items(frame: pd.DataFrame) -> pd.DataFrame:
    """Turn each numeric feature into one interpretable bin per row."""
    cats = pd.DataFrame(index=frame.index)
    for name, (bins, labels) in BIN_SPECS.items():
        cats[name] = pd.cut(frame[name], bins, labels=labels) if labels else frame[name].map(bins)
    return pd.get_dummies(cats, prefix_sep="_").astype(bool)


def mine_default_rules(items: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    basket = items.copy()
    basket["target_default"] = y.eq(1).to_numpy()
    frequent = apriori(basket, min_support=0.03, use_colnames=True, max_len=3)
    if frequent.empty:
        return pd.DataFrame(
            columns=["antecedents", "support", "confidence", "lift", "coverage"]
        )
    rules = association_rules(frequent, metric="confidence", min_threshold=0.5)
    rules = rules[rules["consequents"] == frozenset({"target_default"})].copy()
    if rules.empty:
        return pd.DataFrame(
            columns=["antecedents", "support", "confidence", "lift", "coverage"]
        )
    rules["coverage"] = rules["support"] / rules["confidence"]
    rules["antecedents"] = rules["antecedents"].apply(
        lambda value: ", ".join(sorted(value))
    )
    return rules[["antecedents", "support", "confidence", "lift", "coverage"]].sort_values(
        ["confidence", "lift"], ascending=False
    )


def rule_scores(items: pd.DataFrame, rules: pd.DataFrame) -> np.ndarray:
    scores = np.zeros(len(items), dtype=float)
    for rule in rules.itertuples(index=False):
        columns = rule.antecedents.split(", ")
        mask = items[columns].all(axis=1).to_numpy()
        scores[mask] = np.maximum(scores[mask], rule.confidence)
    return scores


def run_cba(x: pd.DataFrame, y: pd.Series) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = np.zeros(len(y), dtype=float)
    for train_index, test_index in folds.split(x, y):
        train_items = binned_items(x.iloc[train_index])
        test_items = binned_items(x.iloc[test_index])
        rules = mine_default_rules(train_items, y.iloc[train_index])
        if not rules.empty:
            scores[test_index] = rule_scores(test_items, rules)
    metrics = {"model": "CBA rules", **scores_at_threshold(y, scores)}
    final_rules = mine_default_rules(binned_items(x), y)
    return metrics, pd.DataFrame({"prob_default": scores}), final_rules


def k_medoids(
    distance: np.ndarray, k: int, seed: int = SEED
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    medoids = rng.choice(len(distance), k, replace=False)
    for _ in range(100):
        labels = np.argmin(distance[:, medoids], axis=1)
        updated = []
        for cluster in range(k):
            members = np.where(labels == cluster)[0]
            if len(members) == 0:
                updated.append(int(rng.choice(len(distance))))
                continue
            block = distance[np.ix_(members, members)]
            updated.append(int(members[np.argmin(block.sum(axis=0))]))
        updated = np.array(updated)
        if set(updated) == set(medoids):
            break
        medoids = updated
    labels = np.argmin(distance[:, medoids], axis=1)
    return labels, medoids


def fit_kmedoids(distance: np.ndarray, k: int, n_init: int = 5) -> np.ndarray:
    points = np.arange(len(distance))
    best_inertia = np.inf
    best_labels = None
    for offset in range(n_init):
        labels, medoids = k_medoids(distance, k, seed=SEED + offset)
        inertia = float(distance[points, medoids[labels]].sum())
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels
    return best_labels


def merge_small_clusters(
    labels: np.ndarray, min_share: float = MIN_CLUSTER_SHARE
) -> np.ndarray:
    merged = labels.copy()
    counts = pd.Series(labels).value_counts()
    small = counts[(counts.index != -1) & (counts / len(labels) < min_share)].index
    if len(small):
        merged[np.isin(merged, small)] = -1
    return merged


def cluster_scores(scaled: np.ndarray, labels: np.ndarray) -> dict:
    mask = labels != -1
    values = labels[mask]
    n_clusters = int(len(np.unique(values)))
    scores = {
        "n_clusters": n_clusters,
        "silhouette": np.nan,
        "davies_bouldin": np.nan,
        "calinski_harabasz": np.nan,
        "noise_share": float((~mask).mean()),
        "coverage": float(mask.mean()),
        "min_cluster_share": float(
            pd.Series(values).value_counts(normalize=True).min()
        ) if n_clusters >= 2 else np.nan,
    }
    if n_clusters >= 2 and mask.sum() > n_clusters:
        scores["silhouette"] = float(silhouette_score(scaled[mask], values))
        scores["davies_bouldin"] = float(davies_bouldin_score(scaled[mask], values))
        scores["calinski_harabasz"] = float(
            calinski_harabasz_score(scaled[mask], values)
        )
    return scores


def select_partitional(scaled: np.ndarray, fit) -> tuple[dict, list[dict]]:
    rows = []
    candidates = []
    for k in range(2, 7):
        labels = fit(k)
        scores = cluster_scores(scaled, labels)
        row = {
            "param_name": "k",
            "param_value": k,
            **scores,
            "valid": scores["min_cluster_share"] >= MIN_CLUSTER_SHARE,
            "selected": False,
        }
        rows.append(row)
        candidates.append(
            {
                "param_value": k,
                "labels": labels,
                "scores": scores,
            }
        )
    valid = [candidate for row, candidate in zip(rows, candidates) if row["valid"]]
    if not valid:
        valid = candidates
    best = max(valid, key=lambda item: item["scores"]["silhouette"])
    for row in rows:
        row["selected"] = row["param_value"] == best["param_value"]
    return best, rows


def select_dbscan(scaled: np.ndarray) -> tuple[dict, list[dict]]:
    rows = []
    candidates = []
    for eps in np.arange(0.8, 2.01, 0.1):
        raw = DBSCAN(eps=float(eps), min_samples=5).fit_predict(scaled)
        labels = merge_small_clusters(raw)
        scores = cluster_scores(scaled, labels)
        row = {
            "param_name": "eps",
            "param_value": round(float(eps), 1),
            **scores,
            "valid": scores["n_clusters"] >= 2
            and 0.10 <= scores["noise_share"] <= 0.20,
            "selected": False,
        }
        rows.append(row)
        candidates.append(
            {
                "param_value": round(float(eps), 1),
                "labels": labels,
                "scores": scores,
            }
        )
    valid = [candidate for row, candidate in zip(rows, candidates) if row["valid"]]
    if valid:
        best = max(valid, key=lambda item: item["scores"]["silhouette"])
    else:
        core = [
            candidate
            for row, candidate in zip(rows, candidates)
            if row["coverage"] >= 0.75 and row["noise_share"] <= 0.20
        ]
        if core:
            best = min(core, key=lambda item: item["param_value"])
        else:
            best = max(candidates, key=lambda item: item["scores"]["coverage"])
    for row in rows:
        row["selected"] = row["param_value"] == best["param_value"]
    return best, rows


def stability_scores(
    scaled: np.ndarray, fit_predict, n_runs: int = 10, fraction: float = 0.9
) -> dict:
    full = fit_predict(scaled)
    rng = np.random.default_rng(SEED)
    values = []
    for _ in range(n_runs):
        index = rng.choice(
            len(scaled), size=int(round(fraction * len(scaled))), replace=False
        )
        values.append(
            float(adjusted_rand_score(full[index], fit_predict(scaled[index])))
        )
    return {
        "mean_ari": float(np.mean(values)),
        "std_ari": float(np.std(values)),
        "min_ari": float(np.min(values)),
        "max_ari": float(np.max(values)),
        "n_runs": n_runs,
    }


def build_profiles(
    labels: pd.DataFrame, features: pd.DataFrame, y: pd.Series
) -> pd.DataFrame:
    rows = []
    for method in CLUSTER_METHODS:
        for cluster, group in labels.groupby(method):
            index = group.index
            rows.append(
                {
                    "method": method,
                    "cluster": int(cluster),
                    "size": len(group),
                    "share": len(group) / len(labels),
                    "default_rate": float(y.iloc[index].mean()),
                    "avg_amount": float(features.iloc[index]["amount"].mean()),
                    "avg_balance": float(features.iloc[index]["avg_balance"].mean()),
                    "avg_age": float(features.iloc[index]["age"].mean()),
                    "avg_n_txn": float(features.iloc[index]["n_txn"].mean()),
                    "avg_months_active": float(
                        features.iloc[index]["months_active"].mean()
                    ),
                    "avg_txn_per_month": float(
                        features.iloc[index]["txn_per_month"].mean()
                    ),
                    "avg_sum_deposit": float(
                        features.iloc[index]["sum_deposit"].mean()
                    ),
                    "avg_sum_withdrawal": float(
                        features.iloc[index]["sum_withdrawal"].mean()
                    ),
                    "avg_balance_at_loan": float(
                        features.iloc[index]["balance_at_loan"].mean()
                    ),
                }
            )
    profiles = pd.DataFrame(rows)
    order = {method: index for index, method in enumerate(CLUSTER_METHODS)}
    profiles["_order"] = profiles["method"].map(order)
    return (
        profiles.sort_values(["_order", "cluster"])
        .drop(columns="_order")
        .reset_index(drop=True)
    )


def kmeans_segment_outputs(
    features: pd.DataFrame, labels: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = features.assign(segment=labels["kmeans"].to_numpy())
    rows = []
    for segment, group in frame.groupby("segment"):
        row = {"segment": int(segment)}
        row.update({feature: float(group[feature].mean()) for feature in CLUSTER_FEATURES})
        row["size"] = int(len(group))
        row["default_rate"] = float(group["target"].mean())
        rows.append(row)
    segments = frame[["loan_id", "segment"]].sort_values("loan_id").reset_index(drop=True)
    return segments, pd.DataFrame(rows)


def save_cluster_figures(
    comparison: pd.DataFrame,
    stability: pd.DataFrame,
    labels: pd.DataFrame,
    profiles: pd.DataFrame,
    features: pd.DataFrame,
    scaled: np.ndarray,
) -> None:
    figures = ARTIFACTS / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    display = {
        "kmeans": "K-Means",
        "kmedoids": "K-Medoids",
        "agnes": "AGNES (Ward)",
        "dbscan": "DBSCAN",
    }
    palette = {
        "kmeans": "#2f6f9f",
        "kmedoids": "#66c2a5",
        "agnes": "#8da0cb",
        "dbscan": "#d64545",
    }
    params = comparison.set_index("method")["parameter"].to_dict()
    metrics = comparison.set_index("method").loc[CLUSTER_METHODS]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    flat = axes.ravel()
    for axis, column, title in zip(
        flat[:3],
        ["silhouette", "davies_bouldin", "calinski_harabasz"],
        [
            "Silhouette (higher is better)",
            "Davies-Bouldin (lower is better)",
            "Calinski-Harabasz (higher is better)",
        ],
    ):
        values = metrics[column].astype(float)
        bars = axis.bar(
            [display[method] for method in CLUSTER_METHODS],
            values,
            color=[palette[method] for method in CLUSTER_METHODS],
        )
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=15)
        for bar, value in zip(bars, values):
            height = 0.0 if pd.isna(value) else float(value)
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                "n/a" if pd.isna(value) else f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    stability_frame = stability.set_index("method").loc[CLUSTER_METHODS]
    bars = flat[3].bar(
        [display[method] for method in CLUSTER_METHODS],
        stability_frame["mean_ari"],
        yerr=stability_frame["std_ari"],
        capsize=3,
        color=[palette[method] for method in CLUSTER_METHODS],
    )
    flat[3].set_title("Stability ARI (higher is better)")
    flat[3].tick_params(axis="x", rotation=15)
    for bar, value in zip(bars, stability_frame["mean_ari"]):
        flat[3].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(figures / "clustering_metrics.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    pca = PCA(n_components=2, random_state=SEED)
    points = pca.fit_transform(scaled)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for axis, method in zip(axes.flat, CLUSTER_METHODS):
        values = labels[method].to_numpy()
        for cluster in sorted(set(values)):
            mask = values == cluster
            color = "#b0b0b0" if cluster == -1 else palette[method]
            alpha = 0.45 if cluster == -1 else 0.75
            axis.scatter(
                points[mask, 0],
                points[mask, 1],
                s=16,
                alpha=alpha,
                color=color,
                label="noise" if cluster == -1 else f"cluster {cluster}",
            )
        axis.set_title(f"{display[method]} ({params[method]})")
        axis.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%})")
        axis.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%})")
        axis.legend(fontsize=7, markerscale=1.5, frameon=False)
    fig.tight_layout()
    fig.savefig(figures / "clustering_pca.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    base_rate = float(features["target"].mean()) * 100
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    for axis, method in zip(axes.flat, CLUSTER_METHODS):
        frame = profiles[profiles["method"] == method].sort_values("cluster")
        names = ["noise" if value == -1 else str(value) for value in frame["cluster"]]
        rates = frame["default_rate"].to_numpy() * 100
        colors = ["#d64545" if rate > base_rate else "#2f6f9f" for rate in rates]
        bars = axis.bar(names, rates, color=colors)
        axis.axhline(base_rate, color="#8a8f98", linestyle="--", linewidth=1)
        axis.set_title(display[method])
        axis.set_ylabel("Default rate (%)")
        for bar, rate in zip(bars, rates):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{rate:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    fig.tight_layout()
    fig.savefig(
        figures / "clustering_default_rates.png", dpi=160, bbox_inches="tight"
    )
    plt.close(fig)


def run_clusters(
    features: pd.DataFrame, y: pd.Series
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    behaviour = features[CLUSTER_FEATURES].astype(float)
    scaled = StandardScaler().fit_transform(behaviour)
    distance = pairwise_distances(scaled)

    kmeans_best, kmeans_rows = select_partitional(
        scaled,
        lambda k: KMeans(
            n_clusters=k, n_init=10, random_state=SEED
        ).fit_predict(scaled),
    )
    kmedoids_best, kmedoids_rows = select_partitional(
        scaled, lambda k: fit_kmedoids(distance, k)
    )
    agnes_best, agnes_rows = select_partitional(
        scaled,
        lambda k: AgglomerativeClustering(
            n_clusters=k, linkage="ward"
        ).fit_predict(scaled),
    )
    dbscan_best, dbscan_rows = select_dbscan(scaled)

    labels = features[["loan_id"]].copy()
    labels["kmeans"] = kmeans_best["labels"]
    labels["kmedoids"] = kmedoids_best["labels"]
    labels["agnes"] = agnes_best["labels"]
    labels["dbscan"] = dbscan_best["labels"]

    profiles = build_profiles(labels, features, y)

    stability_rows = [
        {
            "method": "kmeans",
            **stability_scores(
                scaled,
                lambda data: KMeans(
                    n_clusters=kmeans_best["param_value"],
                    n_init=10,
                    random_state=SEED,
                ).fit_predict(data),
            ),
        },
        {
            "method": "kmedoids",
            **stability_scores(
                scaled,
                lambda data: fit_kmedoids(
                    pairwise_distances(data), kmedoids_best["param_value"]
                ),
            ),
        },
        {
            "method": "agnes",
            **stability_scores(
                scaled,
                lambda data: AgglomerativeClustering(
                    n_clusters=agnes_best["param_value"], linkage="ward"
                ).fit_predict(data),
            ),
        },
        {
            "method": "dbscan",
            **stability_scores(
                scaled,
                lambda data: merge_small_clusters(
                    DBSCAN(
                        eps=dbscan_best["param_value"], min_samples=5
                    ).fit_predict(data)
                ),
            ),
        },
    ]
    stability = pd.DataFrame(stability_rows)

    roles = {
        "kmeans": "primary segmentation",
        "kmedoids": "alternative partitional",
        "agnes": "hierarchical alternative",
        "dbscan": "outlier / risk flag",
    }
    comparison_rows = []
    for method, best in [
        ("kmeans", kmeans_best),
        ("kmedoids", kmedoids_best),
        ("agnes", agnes_best),
        ("dbscan", dbscan_best),
    ]:
        parameter = (
            f"eps={best['param_value']:.1f}, min_samples=5"
            if method == "dbscan"
            else f"k={best['param_value']}"
        )
        comparison_rows.append(
            {
                "method": method,
                "parameter": parameter,
                **best["scores"],
                "role": roles[method],
                "recommended": method == "kmeans",
            }
        )
    comparison = pd.DataFrame(comparison_rows).merge(
        stability[["method", "mean_ari", "std_ari"]].rename(
            columns={"mean_ari": "stability_ari_mean", "std_ari": "stability_ari_std"}
        ),
        on="method",
        how="left",
    )
    comparison["_order"] = comparison["method"].map(
        {method: index for index, method in enumerate(CLUSTER_METHODS)}
    )
    comparison = (
        comparison.sort_values("_order")
        .drop(columns="_order")
        .reset_index(drop=True)
    )
    comparison = comparison[
        [
            "method",
            "parameter",
            "n_clusters",
            "silhouette",
            "davies_bouldin",
            "calinski_harabasz",
            "noise_share",
            "coverage",
            "min_cluster_share",
            "stability_ari_mean",
            "stability_ari_std",
            "role",
            "recommended",
        ]
    ]

    tuning = pd.DataFrame(
        [
            {"method": "kmeans", **row}
            for row in kmeans_rows
        ]
        + [{"method": "kmedoids", **row} for row in kmedoids_rows]
        + [{"method": "agnes", **row} for row in agnes_rows]
        + [{"method": "dbscan", **row} for row in dbscan_rows]
    )

    segments, segment_profile = kmeans_segment_outputs(features, labels)

    save_cluster_figures(comparison, stability, labels, profiles, features, scaled)

    labels = labels.sort_values("loan_id").reset_index(drop=True)
    return labels, profiles, comparison, tuning, stability, segments, segment_profile


def main() -> None:
    features = load_features().reset_index(drop=True)
    x = features.drop(columns=[column for column in DROP if column in features])
    y = features["target"]

    print("Training MLP...")
    mlp_metrics, mlp_scores, mlp_model = run_mlp(x, y)

    print("Mining CBA rules...")
    cba_metrics, cba_scores, cba_rules = run_cba(x, y)

    for folder in ("metrics", "models", "rules", "segments", "figures"):
        (ARTIFACTS / folder).mkdir(parents=True, exist_ok=True)

    metrics = pd.DataFrame([mlp_metrics, cba_metrics])
    metrics.to_csv(ARTIFACTS / "metrics" / "advanced_metrics.csv", index=False)

    joblib.dump(mlp_model, ARTIFACTS / "models" / "mlp_classifier.pkl")

    cba_rules.to_csv(ARTIFACTS / "rules" / "cba_rules.csv", index=False)

    scored = features[["loan_id"]].copy()
    scored["mlp_prob_default"] = mlp_scores["prob_default"].to_numpy()
    scored["cba_score"] = cba_scores["prob_default"].to_numpy()
    scored["target"] = y.to_numpy()
    scored.to_csv(ARTIFACTS / "metrics" / "advanced_scores.csv", index=False)

    print("Clustering...")
    (
        labels,
        profiles,
        comparison,
        tuning,
        stability,
        segments,
        segment_profile,
    ) = run_clusters(features, y)
    labels.to_csv(ARTIFACTS / "segments" / "cluster_labels.csv", index=False)
    profiles.to_csv(ARTIFACTS / "segments" / "cluster_profiles.csv", index=False)
    segments.to_csv(ARTIFACTS / "segments" / "loan_segments.csv", index=False)
    segment_profile.to_csv(
        ARTIFACTS / "segments" / "segment_profiles.csv", index=False
    )
    comparison.to_csv(
        ARTIFACTS / "metrics" / "clustering_comparison.csv", index=False
    )
    tuning.to_csv(ARTIFACTS / "metrics" / "clustering_tuning.csv", index=False)
    stability.to_csv(
        ARTIFACTS / "metrics" / "clustering_stability.csv", index=False
    )

    print(metrics.to_string(index=False))
    print(f"rules: {len(cba_rules)}")
    print(comparison.to_string(index=False))
    print(stability.to_string(index=False))


if __name__ == "__main__":
    main()
