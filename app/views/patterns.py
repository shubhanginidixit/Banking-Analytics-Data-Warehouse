"""Patterns and segments page."""

import altair as alt
import pandas as pd
import streamlit as st

from components.common import (
    ACCENT,
    INK,
    grid,
    kpi_row,
    load_features,
    load_output,
    note,
    page_header,
    parse_itemset,
    percent,
    score_config,
    section,
    show,
)


def patterns_page() -> None:
    features = load_features()
    itemsets = load_output("pattern_itemsets.csv")
    rules = load_output("pattern_rules.csv")
    if itemsets is None or rules is None:
        st.error("Pattern outputs are missing. Run tools/pattern_mining.py.")
        return
    itemsets = itemsets.drop_duplicates(subset=["itemsets"])
    rules = rules.drop_duplicates(subset=["antecedents", "consequents"])
    if rules.empty:
        st.warning("No association rules pass the configured thresholds.")
        return

    loans = len(features)
    page_header(
        "Patterns",
        f"{len(itemsets)} itemsets, {len(rules)} rules, one worth a second look",
        "FP-Growth and Apriori over pre-loan behaviour baskets. Support, confidence and lift "
        f"are measured against all {loans} loans.",
    )

    top = rules.sort_values("lift", ascending=False).iloc[0]
    kpi_row(
        [
            ("Itemsets", f"{len(itemsets)}", "Identical across both algorithms"),
            ("Default rules", f"{len(rules)}", "Consequence is default"),
            ("Top-rule support", percent(top["support"]), f"{round(top['support'] * loans)} loans"),
            ("Top-rule lift", f"{top['lift']:.2f}", f"Against the {percent(features['target'].mean())} base rate"),
        ]
    )

    section("The rule worth a second look", "Highest lift in the rule set")
    st.markdown(
        f'<div class="note"><b>Rule</b><span style="font-family:var(--serif);font-size:1.15rem">'
        f"{parse_itemset(top['antecedents'])} → {parse_itemset(top['consequents'])}</span></div>",
        unsafe_allow_html=True,
    )
    grid(
        pd.DataFrame(
            [
                ["Antecedent support", percent(top["antecedent support"])],
                ["Consequent support", percent(top["consequent support"])],
                ["Support", percent(top["support"])],
                ["Confidence", f"{top['confidence']:.2f}"],
                ["Lift", f"{top['lift']:.2f}"],
                ["Conviction", f"{top['conviction']:.1f}"],
            ],
            columns=["Measure", "Value"],
        )
    )
    cases = round(top["support"] * loans)
    note(
        f"The rule covers {cases} loans at confidence {top['confidence']:.2f} and lift "
        f"{top['lift']:.2f}. {cases} cases is a small base: the interval around it is wide. "
        "Use it as a screening flag, not an approval rule.",
        label="Reading it",
    )

    section("All rules", f"{len(rules)} rules pass the configured thresholds")
    table = rules.copy()
    table["If"] = table["antecedents"].map(parse_itemset)
    table["Then"] = table["consequents"].map(parse_itemset)
    table["Support %"] = (table["support"] * 100).round(2)
    table["Confidence %"] = (table["confidence"] * 100).round(1)
    grid(
        table[["If", "Then", "Support %", "Confidence %", "lift"]]
        .sort_values("lift", ascending=False)
        .rename(columns={"lift": "Lift"}),
        score_config("Support %", "Lift", places=2) | score_config("Confidence %", places=1),
    )
    st.caption("Both algorithms return the same rules; duplicates are collapsed.")

    cba = load_output("cba_rules.csv")
    if cba is not None and len(cba):
        section("Classifier rules", "CBA-style rules mined from binned behaviour")
        view = cba.copy()
        view["Coverage %"] = (view["coverage"] * 100).round(1)
        view["Confidence %"] = (view["confidence"] * 100).round(1)
        view["Covered"] = (view["coverage"] * loans).round().astype(int)
        view["Label"] = view["Covered"].astype(str) + " covered"
        view["Rule"] = view["antecedents"].map(parse_itemset).str.replace("_", " ")
        bars = (
            alt.Chart(view)
            .mark_bar()
            .encode(
                y=alt.Y("Rule:N", sort="-x", title=None, axis=alt.Axis(labelLimit=300, labelOverlap=False)),
                x=alt.X(
                    "Coverage %:Q",
                    title="Coverage (%)",
                    scale=alt.Scale(domain=[0, float(view["Coverage %"].max()) * 1.4]),
                ),
                color=alt.value(ACCENT),
                tooltip=[
                    "Rule",
                    alt.Tooltip("Coverage %:Q", format=".1f"),
                    "Covered",
                    alt.Tooltip("support:Q", format=".3f"),
                    alt.Tooltip("confidence:Q", format=".3f"),
                    alt.Tooltip("lift:Q", format=".2f"),
                ],
            )
        )
        labels = alt.Chart(view).mark_text(align="left", dx=6, color=INK, fontSize=9, clip=True).encode(
            y=alt.Y("Rule:N", sort="-x", title=None),
            x="Coverage %:Q",
            text="Label:N",
        )
        show((bars + labels).properties(height=max(260, len(view) * 36)).interactive())
        grid(
            view[["antecedents", "support", "Confidence %", "lift", "Coverage %"]].rename(
                columns={
                    "antecedents": "If",
                    "support": "Support",
                    "lift": "Lift",
                }
            ),
            score_config("Support", places=3) | score_config("Lift", places=2),
        )
        st.caption(
            "Mined on the full modelling table with fixed, interpretable bins. "
            f"Confidence runs {view['confidence'].min():.2f} to {view['confidence'].max():.2f}; "
            f"lift runs {view['lift'].min():.2f} to {view['lift'].max():.2f}. "
            "The label counts the loans each rule covers."
        )

    section("Itemsets", f"Apriori and FP-Growth return the same {len(itemsets)} sets")
    threshold = st.slider("Minimum support", 0, 100, 20, step=5)
    table = itemsets.copy()
    table["Support %"] = (table["support"] * 100).round(2)
    filtered = table[table["Support %"] >= threshold].sort_values("Support %", ascending=False)
    table_view = filtered.assign(Items=filtered["itemsets"].map(parse_itemset))[["Items", "Support %"]]
    top = table_view.head(12).assign(Items=lambda x: x["Items"].str.replace("_", " "))
    show(
        alt.Chart(top)
        .mark_bar()
        .encode(
            y=alt.Y("Items:N", sort="-x", title=None, axis=alt.Axis(labelLimit=300, labelOverlap=False)),
            x=alt.X("Support %:Q", title="Support (%)"),
            color=alt.value(ACCENT),
            tooltip=["Items", alt.Tooltip("Support %:Q", format=".1f")],
        )
        .properties(height=max(260, len(top) * 28))
        .interactive()
    )
    grid(table_view)
    st.caption(
        f"{len(filtered)} of {len(itemsets)} itemsets at support ≥ {threshold}%. "
        "Support counts the share of loans that contain the set."
    )


# ------------------------------------------------------------------ simulator
