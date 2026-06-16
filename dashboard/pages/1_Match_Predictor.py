"""Match predictor page."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from _shared import require_model

predictor = require_model()
st.title("🎯 Match Predictor")

teams = predictor.teams
c1, c2, c3 = st.columns([2, 2, 1])
home = c1.selectbox("Home team", teams, index=0)
away = c2.selectbox("Away team", teams, index=1)
neutral = c3.checkbox("Neutral venue", value=True)

if home == away:
    st.info("Pick two different teams.")
    st.stop()

pred = predictor.predict_match(home, away, neutral)

fig = go.Figure(go.Bar(
    x=[pred["p_home_win"], pred["p_draw"], pred["p_away_win"]],
    y=[f"{home} win", "Draw", f"{away} win"],
    orientation="h",
    text=[f"{p:.0%}" for p in (pred["p_home_win"], pred["p_draw"], pred["p_away_win"])],
    marker_color=["#2563eb", "#9ca3af", "#dc2626"],
))
fig.update_layout(height=260, xaxis_tickformat=".0%", margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig, use_container_width=True)

c1, c2, c3 = st.columns(3)
c1.metric(f"{home} xG", pred["exp_home_goals"])
c2.metric(f"{away} xG", pred["exp_away_goals"])
c3.metric("Likely score", pred["likely_scoreline"])
st.caption(f"Model version {pred['model_version']}")
