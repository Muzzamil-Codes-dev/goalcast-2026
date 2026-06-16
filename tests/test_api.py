"""API tests using FastAPI's TestClient with a dependency-overridden predictor."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from goalcast.api.deps import get_predictor
from goalcast.api.main import app


@pytest.fixture()
def client(predictor):
    app.dependency_overrides[get_predictor] = lambda: predictor
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health():
    assert TestClient(app).get("/health").json()["status"] == "ok"


def test_predict_match(client, predictor):
    home, away = predictor.teams[0], predictor.teams[1]
    r = client.post("/predict/match", json={"home_team": home, "away_team": away, "neutral": True})
    assert r.status_code == 200
    body = r.json()
    assert abs(body["p_home_win"] + body["p_draw"] + body["p_away_win"] - 1.0) < 1e-6


def test_predict_scoreline(client, predictor):
    r = client.post("/predict/scoreline",
                    json={"home_team": predictor.teams[0], "away_team": predictor.teams[1]})
    assert r.status_code == 200
    assert "top_scores" in r.json()


def test_simulate_group(client, predictor):
    r = client.post("/simulate/group", json={"teams": predictor.teams[:4], "n_sims": 200})
    assert r.status_code == 200
    assert len(r.json()["results"]) == 4
