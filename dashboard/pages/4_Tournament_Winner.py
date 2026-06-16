"""Tournament winner probabilities page (the hero chart)."""
from __future__ import annotations

import plotly.express as px
import streamlit as st

from _shared import require_model
from goalcast.simulation.monte_carlo import simulate_tournament

predictor = require_model()
st.title("🏆 Tournament Winner Probabilities")
st.caption("Group stage (top 2 advance) + single-elimination bracket, simulated many times.")

teams = predictor.teams
n_groups = st.selectbox("Number of groups (even)", [2, 4], index=1)
n_sims = st.slider("Simulations", 500, 5000, 1500, step=500)

per_group = 4
needed = n_groups * per_group
pool = st.multiselect("Teams in the tournament", teams, default=teams[:needed])

if len(pool) < needed:
    st.info(f"Select {needed} teams ({n_groups} groups of {per_group}).")
    st.stop()

groups = {chr(65 + i): pool[i * per_group:(i + 1) * per_group] for i in range(n_groups)}
with st.expander("Groups"):
    st.json(groups)

if st.button("Run simulation", type="primary"):
    with st.spinner("Running Monte Carlo..."):
        df = simulate_tournament(predictor.b.poisson, groups, n_sims=n_sims)
    top = df[df["p_champion"] > 0].head(15)
    fig = px.bar(top, x="p_champion", y="team", orientation="h", text_auto=".1%",
                 color="p_champion", color_continuous_scale="Tealgrn")
    fig.update_layout(height=500, yaxis=dict(autorange="reversed"), xaxis_tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)
