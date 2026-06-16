"""Model performance page: metrics, run history, calibration."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from _shared import require_model
from goalcast.config import ROOT
from goalcast.data.db import ModelRun, get_session

predictor = require_model()
st.title("📈 Model Performance")

m = predictor.b.metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Log loss", f"{m.get('log_loss', float('nan')):.3f}")
c2.metric("Brier", f"{m.get('brier', float('nan')):.3f}")
c3.metric("Accuracy", f"{m.get('accuracy', float('nan')):.1%}")
c4.metric("Goals MAE", f"{m.get('mae_goals', float('nan')):.2f}")
st.caption("Lower log loss / Brier is better. Accuracy is secondary for probabilistic forecasts.")

st.subheader("Calibration")
cal = ROOT / "docs" / "screenshots" / "calibration.png"
if cal.exists():
    st.image(str(cal))
else:
    st.info("Run `python -m goalcast.models.evaluate` to generate the calibration plot.")

st.subheader("Model run history (registry)")
session = get_session()
runs = session.query(ModelRun).order_by(ModelRun.trained_at.desc()).all()
session.close()
if runs:
    st.dataframe(pd.DataFrame([{
        "version": r.version, "stage": r.stage, "log_loss": r.log_loss,
        "brier": r.brier, "accuracy": r.accuracy, "mae_goals": r.mae_goals,
    } for r in runs]), use_container_width=True, hide_index=True)
else:
    st.info("No runs recorded yet.")
