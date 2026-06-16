"""Shared API dependencies: cached predictor + prediction logging."""
from __future__ import annotations

from functools import lru_cache

from fastapi import HTTPException

from goalcast.api.predictor import Predictor
from goalcast.data.db import Prediction, get_session, init_db
from goalcast.logging_conf import get_logger
from goalcast.models.bundle import ModelBundle

log = get_logger(__name__)


@lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    if not ModelBundle.exists():
        raise HTTPException(
            status_code=503,
            detail="No trained model found. Run `make pipeline` (or `python -m goalcast.models.train`).",
        )
    return Predictor.load()


def log_prediction(pred: dict, model_name: str = "goalcast-wdl") -> None:
    """Persist every prediction so monitoring can later compare against real results."""
    try:
        init_db()
        session = get_session()
        session.add(Prediction(
            home_team=pred["home_team"], away_team=pred["away_team"],
            model_name=model_name, model_version=pred.get("model_version", "unknown"),
            p_home_win=pred["p_home_win"], p_draw=pred["p_draw"], p_away_win=pred["p_away_win"],
            exp_home_goals=pred.get("exp_home_goals"), exp_away_goals=pred.get("exp_away_goals"),
        ))
        session.commit()
        session.close()
    except Exception as exc:  # noqa: BLE001 - logging must never break serving
        log.warning("Could not log prediction: %s", exc)
