"""Scoreline probability heatmap page."""
from __future__ import annotations

import numpy as np
import plotly.express as px
import streamlit as st

from _shared import require_model

predictor = require_model()
st.title("🔢 Scoreline Probabilities")

teams = predictor.teams
c1, c2, c3 = st.columns([2, 2, 1])
home = c1.selectbox("Home", teams, index=0)
away = c2.selectbox("Away", teams, index=1)
neutral = c3.checkbox("Neutral venue", value=True)

if home == away:
    st.info("Pick two different teams.")
    st.stop()

res = predictor.predict_scoreline(home, away, neutral, top=6)
mat = np.array(res["matrix"])[:6, :6]

fig = px.imshow(
    mat, text_auto=".2f", color_continuous_scale="Blues",
    labels=dict(x=f"{away} goals", y=f"{home} goals", color="P"),
    x=list(range(6)), y=list(range(6)),
)
fig.update_layout(height=420)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Most likely scorelines")
st.table(res["top_scores"])
c1, c2, c3 = st.columns(3)
c1.metric(f"{home} win", f"{res['p_home_win']:.0%}")
c2.metric("Draw", f"{res['p_draw']:.0%}")
c3.metric(f"{away} win", f"{res['p_away_win']:.0%}")
