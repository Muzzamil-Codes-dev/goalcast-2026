"""GoalCast 2026 — Streamlit dashboard entry / overview page."""
from __future__ import annotations

import streamlit as st

from _shared import get_predictor

st.set_page_config(page_title="GoalCast 2026", page_icon="⚽", layout="wide")

st.title("⚽ GoalCast 2026")
st.subheader("World Cup 2026 MLOps Predictor")
st.markdown(
    "An end-to-end ML system that forecasts match outcomes and expected goals, and "
    "simulates the FIFA World Cup 2026 with Monte Carlo. Use the pages in the sidebar."
)

predictor = get_predictor()
if predictor is None:
    st.warning("No trained model loaded yet. Run `make pipeline` to ingest data and train.")
else:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model version", predictor.version)
    c2.metric("Teams", len(predictor.teams))
    c3.metric("Test log loss", f"{predictor.b.metrics.get('log_loss', float('nan')):.3f}")
    c4.metric("Test Brier", f"{predictor.b.metrics.get('brier', float('nan')):.3f}")

    st.markdown("---")
    st.markdown(
        "**Pages**: Match Predictor · Scoreline · Group Simulator · Tournament Winner · "
        "Model Performance · Monitoring · AI Match Analyst"
    )

st.caption("Built with XGBoost, Dixon-Coles Poisson, MLflow, FastAPI, Evidently & Claude.")
