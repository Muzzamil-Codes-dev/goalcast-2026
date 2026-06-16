"""Group qualification simulator page."""
from __future__ import annotations

import streamlit as st

from _shared import require_model
from goalcast.simulation.monte_carlo import simulate_group

predictor = require_model()
st.title("📊 Group Simulator")
st.caption("Monte Carlo round-robin — probability each team finishes in the top 2.")

teams = predictor.teams
selected = st.multiselect("Pick 4 teams for the group", teams, default=teams[:4], max_selections=4)
n_sims = st.slider("Simulations", 1000, 20000, 5000, step=1000)

if len(selected) < 3:
    st.info("Select at least 3 teams.")
    st.stop()

with st.spinner("Simulating..."):
    df = simulate_group(predictor.b.poisson, selected, n_sims=n_sims)

df_display = df.copy()
for col in ("p_advance", "p_win_group"):
    df_display[col] = (df_display[col] * 100).round(1).astype(str) + "%"
df_display["exp_points"] = df_display["exp_points"].round(2)
st.dataframe(df_display, use_container_width=True, hide_index=True)
st.bar_chart(df.set_index("team")["p_advance"])
