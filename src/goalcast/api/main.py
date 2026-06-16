"""GoalCast 2026 FastAPI service."""
from __future__ import annotations

from fastapi import Depends, FastAPI

from goalcast import __version__
from goalcast.api.deps import get_predictor, log_prediction
from goalcast.api.predictor import Predictor
from goalcast.api.schemas import (
    ExplainRequest,
    GroupRequest,
    MatchPrediction,
    MatchRequest,
    TournamentRequest,
)
from goalcast.models.registry import get_production
from goalcast.simulation.monte_carlo import simulate_group, simulate_tournament

app = FastAPI(
    title="GoalCast 2026 API",
    description="Forecasts and simulates the FIFA World Cup 2026.",
    version=__version__,
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.get("/model/info")
def model_info(predictor: Predictor = Depends(get_predictor)) -> dict:
    prod = get_production()
    return {
        "model_version": predictor.version,
        "n_teams": len(predictor.teams),
        "metrics": predictor.b.metrics,
        "production_run": prod.version if prod else None,
    }


@app.get("/teams")
def teams(predictor: Predictor = Depends(get_predictor)) -> dict:
    return {"teams": predictor.teams}


@app.post("/predict/match", response_model=MatchPrediction)
def predict_match(req: MatchRequest, predictor: Predictor = Depends(get_predictor)) -> dict:
    pred = predictor.predict_match(req.home_team, req.away_team, req.neutral)
    log_prediction(pred)
    return pred


@app.post("/predict/scoreline")
def predict_scoreline(req: MatchRequest, predictor: Predictor = Depends(get_predictor)) -> dict:
    return predictor.predict_scoreline(req.home_team, req.away_team, req.neutral)


@app.post("/simulate/group")
def sim_group(req: GroupRequest, predictor: Predictor = Depends(get_predictor)) -> dict:
    df = simulate_group(predictor.b.poisson, req.teams, n_sims=req.n_sims)
    return {"results": df.to_dict(orient="records")}


@app.post("/simulate/tournament")
def sim_tournament(req: TournamentRequest, predictor: Predictor = Depends(get_predictor)) -> dict:
    df = simulate_tournament(predictor.b.poisson, req.groups, n_sims=req.n_sims)
    return {"results": df.to_dict(orient="records")}


@app.post("/explain")
def explain(req: ExplainRequest, predictor: Predictor = Depends(get_predictor)) -> dict:
    from goalcast.llm.analyst import explain_prediction

    return explain_prediction(predictor, req.home_team, req.away_team, req.neutral)


@app.get("/metrics")
def metrics() -> dict:
    from goalcast.monitoring.drift import realized_performance

    return realized_performance()
