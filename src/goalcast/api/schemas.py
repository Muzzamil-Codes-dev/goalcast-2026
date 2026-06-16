"""Pydantic request/response models for the API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class MatchRequest(BaseModel):
    home_team: str
    away_team: str
    neutral: bool = False


class MatchPrediction(BaseModel):
    home_team: str
    away_team: str
    p_home_win: float
    p_draw: float
    p_away_win: float
    exp_home_goals: float
    exp_away_goals: float
    likely_scoreline: str
    model_version: str


class GroupRequest(BaseModel):
    teams: list[str] = Field(min_length=3, max_length=6)
    n_sims: int = 3000


class TournamentRequest(BaseModel):
    groups: dict[str, list[str]]
    n_sims: int = 1000


class ExplainRequest(BaseModel):
    home_team: str
    away_team: str
    neutral: bool = False
