"""Simulator page."""

import pandas as pd
import streamlit as st

from components.common import (
    SIM_INPUTS,
    applicant_row,
    driver_figure,
    grid,
    load_features,
    load_model,
    load_scores,
    median_row,
    model_columns,
    money,
    note,
    page_header,
    predict,
    score_config,
    score_distribution,
    section,
    show,
    stress_text,
)


def simulator_page() -> None:
    features = load_features()
    scores = load_scores()
    model = load_model()
    columns = model_columns(model, features)
    base = median_row(features, columns)

    page_header(
        "Simulator",
        "Score one application against the fitted model",
        "The model is the tuned Gradient Boosting pipeline in artifacts/models. Five inputs are "
        f"yours; the other {len(columns) - len(SIM_INPUTS)} features are held at portfolio "
        "medians and listed in full below.",
    )

    left, right = st.columns([1, 1.05], gap="large")

    with left:
        st.markdown('<div class="side-label">Application</div>', unsafe_allow_html=True)
        values = {
            "amount": st.number_input("Loan amount (CZK)", min_value=10_000, max_value=500_000, value=116_928, step=1_000),
            "duration": st.select_slider("Duration (months)", options=[6, 12, 24, 36, 48, 60], value=36),
            "balance_at_loan": st.number_input(
                "Balance at loan date (CZK)", min_value=-20_000, max_value=300_000, value=38_550, step=500
            ),
            "min_balance": st.number_input(
                "Minimum balance in the window (CZK)", min_value=-20_000, max_value=50_000, value=600, step=100
            ),
            "txn_per_month": st.number_input(
                "Transactions per month", min_value=0.0, max_value=15.0, value=5.7, step=0.1, format="%.1f"
            ),
        }
        cutoff = st.slider("Review threshold", 0.05, 0.90, 0.40, 0.01)

    row = applicant_row(base, values)
    probability = predict(model, row, columns)
    flagged = probability >= cutoff
    percentile = float((scores["prob_default"] < probability).mean())

    with right:
        st.markdown(
            f'<div class="kpi-label">Default probability</div>'
            f'<div class="prob {"risk" if flagged else "calm"}">{probability:.1%}</div>'
            f'<div class="verdict">'
            f'{"Above" if flagged else "Below"} the {cutoff:.2f} threshold. '
            f'{"Refer for manual review." if flagged else "Standard processing."}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="side-line" style="margin-top:1.1rem"><span>Position in the portfolio</span>'
            f"<span>above {percentile:.0%} of scores</span></div>"
            f'<div class="side-line"><span>PD × exposure</span><span>{money(probability * values["amount"])}</span></div>'
            f'<div class="side-line"><span>Model</span><span>tuned Gradient Boosting</span></div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "PD × exposure multiplies the probability by the loan amount. Loss given default "
            "is not modelled, so this is an upper bound."
        )

    show(score_distribution(scores, cutoff, probability))
    st.caption(
        "Your application against the out-of-fold score distribution. "
        "Red segments are observed defaults; the shaded region is flagged. "
        "The dashed line is the cutoff, the solid line is this applicant."
    )

    section("What moves the score", "One input at a time, moved to a stressed value")
    stress = {
        "amount": float(features["amount"].quantile(0.95)),
        "duration": 60.0,
        "balance_at_loan": float(features["balance_at_loan"].quantile(0.01)),
        "min_balance": float(features["min_balance"].quantile(0.01)),
        "txn_per_month": float(features["txn_per_month"].quantile(0.01)),
    }
    deltas = []
    for key in SIM_INPUTS:
        altered = dict(values)
        altered[key] = stress[key]
        deltas.append(predict(model, applicant_row(base, altered), columns) - probability)
    left, right = st.columns([1.1, 1], gap="large")
    with left:
        show(driver_figure(list(SIM_INPUTS.values()), deltas))
    with right:
        for (key, label), delta in zip(SIM_INPUTS.items(), deltas):
            short = "Activity" if key == "txn_per_month" else label
            if abs(delta) < 0.0005:
                effect = "no measurable effect"
            else:
                direction = "raises" if delta > 0 else "lowers"
                effect = f"{direction} risk by {abs(delta) * 100:.1f} pp"
            st.markdown(
                f'<div class="side-line"><span>{short} at {stress_text(key, stress[key])}</span>'
                f"<span>{effect}</span></div>",
                unsafe_allow_html=True,
            )
        st.caption(
            "Each row moves one input to a stress value (95th percentile for amount, 60 months, "
            "1st percentile for balances and activity) and holds the rest at your entries."
        )

    section(
        "What the model sees",
        f"All {len(columns)} features, applicant value against the portfolio median",
    )
    with st.expander("Open the full feature vector", expanded=False):
        vector = pd.DataFrame(
            {
                "Feature": columns,
                "Applicant": [row[column] for column in columns],
                "Portfolio median": [base[column] for column in columns],
            }
        )
        grid(
            vector,
            score_config("Applicant", "Portfolio median", places=2),
        )

    note(
        "payments is derived as amount ÷ duration, exactly as in the feature build, so it is not "
        "an independent input. The pipeline was fitted on all "
        f"{len(features):,} loans; the portfolio page uses out-of-fold scores. Use this page to "
        "compare applications, not to read a calibrated probability.",
        label="Method",
    )


# ---------------------------------------------------------------------- shell
