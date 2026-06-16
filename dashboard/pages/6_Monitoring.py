"""Monitoring page: realized performance + data drift."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from _shared import require_model
from goalcast.features.build import FEATURE_COLUMNS
from goalcast.models.dataset import time_split
from goalcast.monitoring.drift import data_drift, realized_performance

require_model()
st.title("🔍 Monitoring & Drift")

st.subheader("Realized performance (predicted vs actual)")
perf = realized_performance()
if perf.get("n_scored", 0) == 0:
    st.info(perf.get("message", "No settled predictions yet. Log predictions, then settle results."))
else:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Scored", perf["n_scored"])
    c2.metric("Log loss", f"{perf['log_loss']:.3f}")
    c3.metric("Brier", f"{perf['brier']:.3f}")
    c4.metric("Accuracy", f"{perf['accuracy']:.1%}")

st.subheader("Data drift (train vs recent)")
try:
    split = time_split()
    drift = data_drift(split.train, split.test, FEATURE_COLUMNS)
    psi_df = pd.DataFrame(
        [{"feature": k, "psi": v, "drifted": v > 0.2} for k, v in drift["psi"].items()]
    ).sort_values("psi", ascending=False)
    st.dataframe(psi_df, use_container_width=True, hide_index=True)
    st.caption("PSI > 0.2 indicates meaningful drift. Evidently HTML report saved to docs/ when available.")
except Exception as exc:  # noqa: BLE001
    st.info(f"Drift check unavailable: {exc}")
