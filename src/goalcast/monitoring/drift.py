"""Monitoring: realized performance (predicted vs actual) + data drift.

Closes the ML feedback loop — the part most portfolio projects skip.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from goalcast.config import ROOT
from goalcast.data.db import Prediction, get_session
from goalcast.logging_conf import get_logger

log = get_logger(__name__)
REPORT_PATH = ROOT / "docs" / "screenshots" / "drift_report.html"
_OUTCOME_IDX = {"home_win": 0, "draw": 1, "away_win": 2}


def realized_performance() -> dict:
    """Log loss / Brier / accuracy on predictions whose real result is now known."""
    session = get_session()
    rows = session.query(Prediction).filter(Prediction.actual_outcome.isnot(None)).all()
    session.close()
    if not rows:
        return {"n_scored": 0, "message": "No settled predictions yet."}

    probs = np.array([[r.p_home_win, r.p_draw, r.p_away_win] for r in rows])
    probs = np.clip(probs, 1e-9, 1.0)
    probs /= probs.sum(axis=1, keepdims=True)
    y = np.array([_OUTCOME_IDX[r.actual_outcome] for r in rows])
    onehot = np.eye(3)[y]

    log_loss = float(-np.mean(np.log(probs[np.arange(len(y)), y])))
    brier = float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))
    accuracy = float(np.mean(probs.argmax(axis=1) == y))
    return {"n_scored": len(rows), "log_loss": log_loss, "brier": brier, "accuracy": accuracy}


def _psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index — lightweight drift metric (no deps)."""
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    if len(edges) < 2:
        return 0.0
    ref_pct = np.histogram(reference, bins=edges)[0] / len(reference) + 1e-6
    cur_pct = np.histogram(current, bins=edges)[0] / max(len(current), 1) + 1e-6
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def data_drift(reference: pd.DataFrame, current: pd.DataFrame, columns: list[str]) -> dict:
    """Per-feature PSI plus an Evidently HTML report when available."""
    psi = {c: round(_psi(reference[c].to_numpy(), current[c].to_numpy()), 4) for c in columns}
    drifted = [c for c, v in psi.items() if v > 0.2]

    try:
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference[columns], current_data=current[columns])
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        report.save_html(str(REPORT_PATH))
        log.info("Saved Evidently report -> %s", REPORT_PATH)
    except Exception as exc:  # noqa: BLE001 - evidently optional
        log.info("Evidently report skipped (%s); PSI fallback used.", exc)

    return {"psi": psi, "drifted_features": drifted, "report_path": str(REPORT_PATH)}
