"""Loan default review dashboard entrypoint."""

import streamlit as st

from components.common import apply_theme, load_features, sidebar
from views.data_leakage import data_page
from views.data_model import data_model_page
from views.models import models_page
from views.patterns import patterns_page
from views.portfolio import portfolio_page
from views.segments import segments_page
from views.simulator import simulator_page
from views.snowflake import snowflake_page


def main() -> None:
    apply_theme()
    features = load_features()
    sidebar(features)
    st.navigation(
        [
            st.Page(portfolio_page, title="Executive overview", default=True),
            st.Page(data_page, title="Data & leakage"),
            st.Page(models_page, title="Models"),
            st.Page(patterns_page, title="Patterns"),
            st.Page(segments_page, title="Segments"),
            st.Page(simulator_page, title="Simulator"),
            st.Page(snowflake_page, title="Snowflake"),
            st.Page(data_model_page, title="Data model card"),
        ]
    ).run()


if __name__ == "__main__":
    main()
