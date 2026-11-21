"""Models page."""

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from components.common import (
    ACCENT,
    GOOD,
    INK,
    MUTED,
    RISK,
    RULE,
    confusion_figure,
    grid,
    heatmap_figure,
    load_features,
    load_output,
    load_scores,
    note,
    page_header,
    ref_rule,
    roc_pr_frames,
    score_config,
    scored_frame,
    section,
    show,
)

MODELS = ["Gradient Boosting", "MLP", "CBA rules"]
COLOURS = [ACCENT, GOOD, RISK]
MODEL_LABELS = {
    "DecisionTree": "Decision Tree",
    "GradientBoosting": "Gradient Boosting",
    "KNN": "KNN",
    "NaiveBayes": "Naive Bayes",
    "RandomForest": "Random Forest",
    "SVM": "SVM",
}


def curve(frame: pd.DataFrame, x: str, y: str, extra):
    return alt.Chart(frame).mark_line(strokeWidth=2).encode(
        x=alt.X(f"{x}:Q", title=x, axis=alt.Axis(format="%")),
        y=alt.Y(f"{y}:Q", title=y, axis=alt.Axis(format="%")),
        color=alt.Color(
            "Model:N",
            scale=alt.Scale(domain=MODELS, range=COLOURS),
            legend=alt.Legend(title=None, orient="bottom"),
        ),
        tooltip=[
            "Model",
            alt.Tooltip(f"{x}:Q", format=".2%"),
            alt.Tooltip(f"{y}:Q", format=".2%"),
        ],
    ) + extra


def models_page() -> None:
    features = load_features()
    cv = load_output("cv_comparison.csv")
    metrics = load_output("metrics_comparison.csv")
    tuning = load_output("tuning_results.csv")
    if cv is None or metrics is None or cv.empty or metrics.empty:
        st.error("Model comparison outputs are missing.")
        return

    page_header(
        "Models",
        f"{cv['model'].nunique()} classifiers, repeated-CV estimates each",
        "Selection rule: highest repeated-CV F1 on the unresampled baseline arm. The other "
        "columns are checks on that rule, not tie-breakers.",
    )

    controls = st.columns(3)
    metric = controls[0].selectbox("Rank by", ["F1", "Average precision", "ROC-AUC"])
    setting = controls[1].radio("Resampling", ["baseline", "smote"], horizontal=True)
    variant = controls[2].radio("Variant", ["tuned", "default"], horizontal=True)

    metric_column = {"F1": "cv_f1_mean", "Average precision": "cv_ap_mean", "ROC-AUC": "cv_auc_mean"}[metric]
    sd_column = {"F1": "cv_f1_std", "Average precision": "cv_ap_std", "ROC-AUC": "cv_auc_std"}[metric]
    selected = cv[(cv["setting"] == setting) & (cv["variant"] == variant)].copy()
    if selected.empty:
        st.warning("No runs match that combination.")
        return
    board = selected.sort_values(metric_column, ascending=False).reset_index(drop=True)
    board.insert(0, "Rank", np.arange(1, len(board) + 1))
    board["Model"] = board["model"].map(MODEL_LABELS).fillna(board["model"])
    plot = board.copy()
    plot["Low"] = plot[metric_column] - plot[sd_column]
    plot["High"] = plot[metric_column] + plot[sd_column]
    order = plot["Model"].tolist()
    low = max(0.0, float(plot["Low"].min()) - 0.04)
    high = min(1.0, float(plot["High"].max()) + 0.10)
    domain = alt.Scale(domain=[low, high])
    interval = alt.Chart(plot).mark_rule(strokeWidth=3, color=MUTED, opacity=0.55).encode(
        y=alt.Y("Model:N", sort=order),
        x=alt.X("Low:Q", title=f"{metric} (mean of 50 repeated-CV folds)", scale=domain, axis=alt.Axis(format=".2f")),
        x2="High:Q",
    )
    dots = alt.Chart(plot).mark_circle(size=90, color=ACCENT).encode(
        y=alt.Y("Model:N", sort=order),
        x=alt.X(f"{metric_column}:Q", scale=domain),
        tooltip=[
            "Model",
            alt.Tooltip(f"{metric_column}:Q", format=".3f", title=metric),
            alt.Tooltip(f"{sd_column}:Q", format=".3f", title="SD"),
        ],
    )
    labels = alt.Chart(plot).mark_text(align="left", dx=7, color=INK, fontSize=10).encode(
        y=alt.Y("Model:N", sort=order),
        x=alt.X("High:Q", scale=domain),
        text=alt.Text(f"{metric_column}:Q", format=".3f"),
    )
    show((interval + dots + labels).properties(height=max(220, len(plot) * 32)).interactive())
    st.caption(
        "Dot = mean; line = ±1 SD across 50 repeated-CV estimates. "
        "Folds are not independent, so this is descriptive spread, not a confidence interval. "
        "The axis starts near the observed range."
    )
    grid(
        board[["Rank", "Model", "cv_f1_mean", "cv_f1_std", "cv_ap_mean", "cv_auc_mean"]].rename(
            columns={
                "cv_f1_mean": "F1",
                "cv_f1_std": "F1 sd",
                "cv_ap_mean": "Avg precision",
                "cv_auc_mean": "ROC-AUC",
            }
        ),
        score_config("F1", "F1 sd", "Avg precision", "ROC-AUC"),
    )

    section("Held-out scores", "Default hyperparameters on the held-out split")
    held = metrics.copy()
    held["Run"] = held["model"] + " / " + held["setting"]
    columns = ["accuracy", "precision", "recall", "f1", "average_precision", "roc_auc"]
    labels = ["Accuracy", "Precision", "Recall", "F1", "Avg precision", "ROC-AUC"]
    show(heatmap_figure(held.set_index("Run")[columns], labels, held["Run"].tolist()))

    advanced = load_output("advanced_metrics.csv")
    if advanced is not None and len(advanced):
        section(
            "Additional models",
            "5-fold out-of-fold scores, threshold chosen from the same out-of-fold scores",
        )
        view = advanced.rename(
            columns={
                "model": "Model",
                "threshold": "Threshold",
                "accuracy": "Accuracy",
                "precision": "Precision",
                "recall": "Recall",
                "f1": "F1",
                "average_precision": "Avg precision",
                "roc_auc": "ROC-AUC",
            }
        )
        advanced_models = advanced["model"].unique().tolist()
        metrics_long = advanced.melt(
            id_vars="model",
            value_vars=["precision", "recall", "f1", "average_precision", "roc_auc"],
            var_name="Metric",
            value_name="Score",
        )
        metrics_long["Metric"] = metrics_long["Metric"].map(
            {
                "precision": "Precision",
                "recall": "Recall",
                "f1": "F1",
                "average_precision": "Avg precision",
                "roc_auc": "ROC-AUC",
            }
        )
        show(
            alt.Chart(metrics_long)
            .mark_bar()
            .encode(
                x=alt.X("Metric:N", title=None, axis=alt.Axis(labelAngle=-20)),
                xOffset=alt.XOffset("model:N", sort=advanced_models),
                y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color(
                    "model:N",
                    scale=alt.Scale(
                        domain=advanced_models,
                        range=[COLOURS[index % len(COLOURS)] for index in range(len(advanced_models))],
                    ),
                    title=None,
                ),
                tooltip=["model", "Metric", alt.Tooltip("Score:Q", format=".3f")],
            )
            .properties(height=260)
            .interactive()
        )
        grid(
            view[["Model", "Threshold", "Precision", "Recall", "F1", "Avg precision", "ROC-AUC"]],
            {
                **score_config("Precision", "Recall", "F1", "Avg precision", "ROC-AUC"),
                **score_config("Threshold", places=2),
            },
        )
        cba = advanced[advanced["model"] == "CBA rules"]
        if not cba.empty:
            cba = cba.iloc[0]
            note(
                f"The rule classifier is precise ({cba['precision']:.2f}) but catches only "
                f"{cba['recall']:.0%} of defaults, because its rules are dominated by the "
                "negative-balance condition. The MLP is a nonlinear check on the tree models; "
                "it does not beat the tuned Gradient Boosting.",
                label="Reading it",
            )

    tabs = st.tabs(["Repeated CV", "Curves", "Tuning", "Challenger"])

    with tabs[0]:
        table = cv.copy()
        table["Configuration"] = table["setting"] + " / " + table["variant"]
        grid(
            table.sort_values("cv_f1_mean", ascending=False)[
                ["model", "Configuration", "cv_f1_mean", "cv_f1_std", "cv_ap_mean", "cv_auc_mean"]
            ].rename(
                columns={
                    "model": "Model",
                    "cv_f1_mean": "F1",
                    "cv_f1_std": "F1 sd",
                    "cv_ap_mean": "Avg precision",
                    "cv_auc_mean": "ROC-AUC",
                }
            ),
            score_config("F1", "F1 sd", "Avg precision", "ROC-AUC"),
        )
        st.caption(
            f"{len(cv)} configurations, repeated-CV estimates each. "
            "Selection uses this table, not the held-out split."
        )

    with tabs[1]:
        roc, pr = roc_pr_frames()
        left, right = st.columns(2)
        diagonal = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(
            color=RULE, strokeDash=[4, 3]
        ).encode(x="x:Q", y="y:Q")
        with left:
            st.markdown("**ROC curves**")
            show(curve(roc, "False positive rate", "True positive rate", diagonal))

        base_rate = features["target"].mean()
        baseline = ref_rule(base_rate, axis="y", color=RULE)
        with right:
            st.markdown("**Precision-recall curves**")
            show(curve(pr, "Recall", "Precision", baseline))
        st.caption(
            "Out-of-fold scores from the selected model, the MLP, and the CBA rules. "
            f"The dashed line is the {base_rate:.1%} base rate."
        )

        st.markdown("**Confusion matrix, out-of-fold scores**")
        cutoff = st.slider("Cutoff", 0.05, 0.95, 0.50, 0.01, key="models_cutoff")
        show(confusion_figure(scored_frame(load_scores(), features), cutoff))

    with tabs[2]:
        if tuning is not None:
            grid(
                tuning[["model", "setting", "n_candidates", "inner_cv_f1", "search_seconds", "best_params"]].rename(
                    columns={
                        "model": "Model",
                        "setting": "Setting",
                        "n_candidates": "Candidates",
                        "inner_cv_f1": "Inner F1",
                        "search_seconds": "Seconds",
                        "best_params": "Selected parameters",
                    }
                ),
                {
                    **score_config("Inner F1"),
                    **score_config("Seconds", places=1),
                },
            )
            chart = alt.Chart(tuning).mark_bar().encode(
                x=alt.X("model:N", title=None, axis=alt.Axis(labelAngle=-20)),
                xOffset=alt.XOffset("setting:N", sort=["baseline", "smote"]),
                y=alt.Y("inner_cv_f1:Q", title="Inner CV F1"),
                color=alt.Color(
                    "setting:N",
                    scale=alt.Scale(domain=["baseline", "smote"], range=[ACCENT, MUTED]),
                    title=None,
                ),
                tooltip=[
                    "model",
                    "setting",
                    "n_candidates",
                    alt.Tooltip("inner_cv_f1:Q", format=".3f"),
                    alt.Tooltip("search_seconds:Q", format=".1f"),
                ],
            )
            show(chart.properties(height=280).interactive())

    with tabs[3]:
        comparison = load_output("capacity_model_comparison.csv")
        if comparison is None:
            st.warning("Challenger results are not available.")
        else:
            challenger = comparison.melt(
                id_vars="model",
                value_vars=[
                    "f1_default_mean",
                    "f1_tuned_mean",
                    "average_precision_mean",
                    "roc_auc_mean",
                ],
                var_name="Metric",
                value_name="Score",
            )
            challenger["Metric"] = challenger["Metric"].map(
                {
                    "f1_default_mean": "F1 at 0.50",
                    "f1_tuned_mean": "F1 tuned",
                    "average_precision_mean": "Avg precision",
                    "roc_auc_mean": "ROC-AUC",
                }
            )
            pivot = challenger.pivot(index="Metric", columns="model", values="Score").reset_index()
            current = next((name for name in pivot.columns if "Current" in name), None)
            challenger_name = next((name for name in pivot.columns if "Capacity" in name), None)
            if current is None or challenger_name is None:
                st.warning("Challenger model names are not in the comparison.")
                return
            pivot["Current"] = pivot[current]
            pivot["Challenger"] = pivot[challenger_name]
            metric_order = ["F1 at 0.50", "F1 tuned", "Avg precision", "ROC-AUC"]
            points = pivot.melt(
                id_vars="Metric",
                value_vars=["Current", "Challenger"],
                var_name="Model",
                value_name="Score",
            )
            points["Model"] = points["Model"].map(
                {"Current": "Current tuned", "Challenger": "Capacity-aware"}
            )
            domain = alt.Scale(
                domain=[
                    float(points["Score"].min()) - 0.015,
                    float(points["Score"].max()) + 0.015,
                ]
            )
            connector = (
                alt.Chart(pivot)
                .mark_rule(strokeWidth=3, color=MUTED, opacity=0.8)
                .encode(
                    y=alt.Y("Metric:N", sort=metric_order, title=None),
                    x=alt.X("Current:Q", scale=domain),
                    x2="Challenger:Q",
                )
            )
            dots = (
                alt.Chart(points)
                .mark_circle(size=90)
                .encode(
                    y=alt.Y("Metric:N", sort=metric_order, title=None),
                    x=alt.X("Score:Q", title="Score", scale=domain),
                    color=alt.Color(
                        "Model:N",
                        scale=alt.Scale(
                            domain=["Current tuned", "Capacity-aware"], range=[ACCENT, GOOD]
                        ),
                        title=None,
                    ),
                    tooltip=[
                        "Model",
                        "Metric",
                        alt.Tooltip("Score:Q", format=".3f"),
                    ],
                )
            )
            show((connector + dots).properties(height=220).interactive())
            st.caption(
                "Dots that overlap mean the two models score almost identically on that metric. "
                "The line shows the direction and size of the gap."
            )
            grid(
                comparison.rename(
                    columns={
                        "model": "Model",
                        "f1_default_mean": "F1 at 0.50",
                        "f1_tuned_mean": "F1 at tuned threshold",
                        "average_precision_mean": "Avg precision",
                        "roc_auc_mean": "ROC-AUC",
                        "median_threshold": "Threshold",
                    }
                )[
                    ["Model", "F1 at 0.50", "F1 at tuned threshold", "Avg precision", "ROC-AUC", "Threshold"]
                ],
                {
                    **score_config("F1 at 0.50", "F1 at tuned threshold", "Avg precision", "ROC-AUC"),
                    **score_config("Threshold", places=2),
                },
            )
            current_row = comparison[
                comparison["model"].str.contains("Current", case=False, na=False)
            ]
            challenger_row = comparison[
                comparison["model"].str.contains("Capacity", case=False, na=False)
            ]
            if not current_row.empty and not challenger_row.empty:
                current_row = current_row.iloc[0]
                challenger_row = challenger_row.iloc[0]
                note(
                    "The challenger adds four capacity ratios and a tuned threshold. "
                    f"It gains {challenger_row['roc_auc_mean'] - current_row['roc_auc_mean']:+.3f} "
                    f"ROC-AUC and {challenger_row['f1_tuned_mean'] - current_row['f1_tuned_mean']:+.3f} "
                    "F1 against the selected model, so it stays a challenger.",
                    label="Result",
                )

    baseline_tuned = cv[(cv["setting"] == "baseline") & (cv["variant"] == "tuned")].sort_values(
        "cv_f1_mean", ascending=False
    )
    if not baseline_tuned.empty:
        best = baseline_tuned.iloc[0]
        rival = baseline_tuned[baseline_tuned["model"] == "RandomForest"]
        if not rival.empty:
            rival = rival.iloc[0]
            note(
                f"The tuned baseline arm leads on F1 at {best['cv_f1_mean']:.3f} ± {best['cv_f1_std']:.3f}. "
                f"Random Forest trails by {best['cv_f1_mean'] - rival['cv_f1_mean']:.3f} on F1 but leads on "
                f"average precision ({rival['cv_ap_mean']:.3f} vs {best['cv_ap_mean']:.3f}) and ROC-AUC "
                f"({rival['cv_auc_mean']:.3f} vs {best['cv_auc_mean']:.3f}). The F1 gap is a fraction of the "
                "fold-to-fold spread, so the choice is a convention rather than a decisive win.",
                label="Why this model",
            )

    note(
        f"{int(features['target'].sum())} defaults in {len(features):,} loans. Every interval on "
        "this page is wide. Treat the ranking as a shortlist, not a verdict.",
        label="Limits",
        kind="risk",
    )
