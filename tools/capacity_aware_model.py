"""Evaluate and persist a capacity-aware Gradient Boosting model.

The experiment keeps the existing headline model unchanged and adds a
dataset-specific model version with financial-capacity features and a
training-only decision-threshold search for the default class.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "features" / "loan_features.csv"
OUTPUT_PATH = ROOT / "artifacts"
MODEL_PATH = OUTPUT_PATH / "models" / "capacity_aware_gradient_boosting.pkl"
METRICS_PATH = OUTPUT_PATH / "metrics" / "capacity_model_comparison.csv"
FOLDS_PATH = OUTPUT_PATH / "metrics" / "capacity_model_fold_results.csv"
METADATA_PATH = OUTPUT_PATH / "models" / "capacity_aware_metadata.json"

DROP_COLUMNS = {
    "loan_id",
    "account_id",
    "client_id",
    "district_id",
    "status",
    "date",
    "parsed_date",
    "gender",
    "target",
}
MODEL_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 1,
    "subsample": 0.7,
    "random_state": SEED,
}
THRESHOLDS = np.arange(0.20, 0.71, 0.01)


def make_capacity_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add stable, pre-loan measures of loan burden relative to balance."""
    result = frame.copy()
    average_balance = result["avg_balance"].clip(lower=0)
    balance_at_loan = result["balance_at_loan"].clip(lower=0)

    result["loan_to_avg_balance"] = np.log1p(
        (result["amount"] / (average_balance + 1)).clip(upper=50)
    )
    result["loan_to_balance_at_loan"] = np.log1p(
        (result["amount"] / (balance_at_loan + 1)).clip(upper=50)
    )
    result["payment_to_avg_balance"] = np.log1p(
        (result["payments"] / (average_balance + 1)).clip(upper=50)
    )
    result["negative_min_balance"] = (result["min_balance"] < 0).astype(int)
    return result


def pipeline() -> object:
    return make_pipeline(
        StandardScaler(),
        GradientBoostingClassifier(**MODEL_PARAMS),
    )


def select_threshold(x_train: pd.DataFrame, y_train: pd.Series) -> float:
    """Select the Class 1 F1 threshold from inner out-of-fold predictions."""
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    probabilities = np.zeros(len(y_train), dtype=float)
    for inner_train, inner_valid in inner.split(x_train, y_train):
        estimator = pipeline()
        estimator.fit(x_train.iloc[inner_train], y_train.iloc[inner_train])
        probabilities[inner_valid] = estimator.predict_proba(
            x_train.iloc[inner_valid]
        )[:, 1]

    scores = [
        f1_score(y_train, probabilities >= threshold, zero_division=0)
        for threshold in THRESHOLDS
    ]
    return float(THRESHOLDS[int(np.argmax(scores))])


def evaluate(
    features: pd.DataFrame,
    target: pd.Series,
    model_name: str,
) -> tuple[pd.DataFrame, float]:
    """Evaluate one feature set with nested threshold selection."""
    # The baseline model already has the project's definitive 5 x 10
    # comparison. This experimental nested run uses 5 x 3 to keep the
    # additional inner threshold searches practical.
    outer = RepeatedStratifiedKFold(
        n_splits=5,
        n_repeats=3,
        random_state=SEED,
    )
    rows = []
    for fold, (train_index, valid_index) in enumerate(
        outer.split(features, target),
        start=1,
    ):
        x_train = features.iloc[train_index]
        x_valid = features.iloc[valid_index]
        y_train = target.iloc[train_index]
        y_valid = target.iloc[valid_index]
        threshold = select_threshold(x_train, y_train)
        estimator = pipeline()
        estimator.fit(x_train, y_train)
        probabilities = estimator.predict_proba(x_valid)[:, 1]
        rows.append(
            {
                "model": model_name,
                "fold": fold,
                "threshold": threshold,
                "f1_default_threshold": f1_score(
                    y_valid, probabilities >= 0.5, zero_division=0
                ),
                "f1_tuned_threshold": f1_score(
                    y_valid, probabilities >= threshold, zero_division=0
                ),
                "average_precision": average_precision_score(y_valid, probabilities),
                "roc_auc": roc_auc_score(y_valid, probabilities),
            }
        )
    return pd.DataFrame(rows), float(np.median([row["threshold"] for row in rows]))


def main() -> None:
    frame = pd.read_csv(DATA_PATH)
    target = frame["target"].astype(int)
    base = frame.drop(columns=[column for column in DROP_COLUMNS if column in frame])
    capacity = make_capacity_features(base)

    baseline_rows, _ = evaluate(base, target, "Current tuned Gradient Boosting")
    capacity_rows, selected_threshold = evaluate(
        capacity,
        target,
        "Capacity-aware Gradient Boosting",
    )
    folds = pd.concat([baseline_rows, capacity_rows], ignore_index=True)
    metrics = (
        folds.groupby("model")
        .agg(
            f1_default_mean=("f1_default_threshold", "mean"),
            f1_default_std=("f1_default_threshold", "std"),
            f1_tuned_mean=("f1_tuned_threshold", "mean"),
            f1_tuned_std=("f1_tuned_threshold", "std"),
            average_precision_mean=("average_precision", "mean"),
            average_precision_std=("average_precision", "std"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            median_threshold=("threshold", "median"),
        )
        .reset_index()
    )

    (OUTPUT_PATH / "metrics").mkdir(parents=True, exist_ok=True)
    (OUTPUT_PATH / "models").mkdir(parents=True, exist_ok=True)
    folds.to_csv(FOLDS_PATH, index=False)
    metrics.to_csv(METRICS_PATH, index=False)

    final_model = pipeline()
    final_model.fit(capacity, target)
    joblib.dump(final_model, MODEL_PATH)
    METADATA_PATH.write_text(
        json.dumps(
            {
                "model": "Capacity-aware Gradient Boosting",
                "base_model": "Tuned Gradient Boosting",
                "threshold": selected_threshold,
                "threshold_selection": "median of nested 5-fold inner-CV thresholds",
                "capacity_features": [
                    "loan_to_avg_balance",
                    "loan_to_balance_at_loan",
                    "payment_to_avg_balance",
                    "negative_min_balance",
                ],
                "model_parameters": MODEL_PARAMS,
                "warning": "This artifact is an experimental model version; compare it with the current headline model before replacing production selection.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(metrics.to_string(index=False))
    print(f"\nSaved: {METRICS_PATH}")
    print(f"Saved: {FOLDS_PATH}")
    print(f"Saved: {MODEL_PATH}")


if __name__ == "__main__":
    main()
