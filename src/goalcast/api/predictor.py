"""Serving-time predictor: builds live feature rows from the bundle and scores matchups."""
from __future__ import annotations

import numpy as np
import pandas as pd

from goalcast.features.build import HOSTS_2026
from goalcast.features.elo import BASE_RATING
from goalcast.models.bundle import ModelBundle

_DEFAULT_SNAP = {"form": 1.3, "gf": 1.3, "ga": 1.3}


class Predictor:
    def __init__(self, bundle: ModelBundle) -> None:
        self.b = bundle

    @classmethod
    def load(cls) -> Predictor:
        return cls(ModelBundle.load())

    @property
    def teams(self) -> list[str]:
        return self.b.teams

    @property
    def version(self) -> str:
        return self.b.version

    def _snap(self, team: str) -> dict[str, float]:
        return self.b.team_snapshot.get(team, _DEFAULT_SNAP)

    def _feature_row(self, home: str, away: str, neutral: bool) -> pd.DataFrame:
        hs, as_ = self._snap(home), self._snap(away)
        home_elo = self.b.elo_ratings.get(home, BASE_RATING)
        away_elo = self.b.elo_ratings.get(away, BASE_RATING)
        adv = 0.0 if neutral else 1.0
        if home in HOSTS_2026:
            adv += 0.5
        row = {
            "elo_diff": home_elo - away_elo, "home_elo": home_elo, "away_elo": away_elo,
            "home_form": hs["form"], "away_form": as_["form"],
            "home_gf": hs["gf"], "home_ga": hs["ga"], "away_gf": as_["gf"], "away_ga": as_["ga"],
            "home_rest_days": 7.0, "away_rest_days": 7.0, "home_advantage": adv,
        }
        return pd.DataFrame([row])[self.b.feature_columns]

    def predict_match(self, home: str, away: str, neutral: bool = False) -> dict:
        proba = self.b.classifier.predict_proba(self._feature_row(home, away, neutral))[0]
        lam_h, lam_a = self.b.poisson.expected_goals(home, away, neutral)
        score = self.b.poisson.most_likely_score(home, away, neutral)
        return {
            "home_team": home, "away_team": away,
            "p_home_win": float(proba[0]), "p_draw": float(proba[1]), "p_away_win": float(proba[2]),
            "exp_home_goals": round(lam_h, 2), "exp_away_goals": round(lam_a, 2),
            "likely_scoreline": f"{score[0]}-{score[1]}",
            "model_version": self.b.version,
        }

    def predict_scoreline(self, home: str, away: str, neutral: bool = False, top: int = 5) -> dict:
        mat = self.b.poisson.scoreline_matrix(home, away, neutral)
        ranked = sorted(
            ((float(mat[i, j]), i, j) for i in range(mat.shape[0]) for j in range(mat.shape[1])),
            reverse=True,
        )
        top_scores = [{"score": f"{i}-{j}", "prob": p} for p, i, j in ranked[:top]]
        return {
            "home_team": home, "away_team": away,
            "matrix": np.round(mat, 4).tolist(),
            "top_scores": top_scores,
            **self.b.poisson.outcome_probs(home, away, neutral),
        }
