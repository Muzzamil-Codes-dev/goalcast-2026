"""Simplified Dixon-Coles bivariate Poisson goals model.

Estimates per-team attack & defense strengths plus home advantage via Poisson
regression, then builds a scoreline probability matrix with a low-score correction.
One model yields: expected goals, most-likely scoreline, W/D/L probabilities, and the
sampling distribution for the Monte Carlo tournament simulation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor

MAX_GOALS = 10
DEFAULT_RHO = -0.08  # Dixon-Coles low-score dependence parameter


@dataclass
class PoissonGoals:
    teams: list[str] = field(default_factory=list)
    attack: dict[str, float] = field(default_factory=dict)
    defense: dict[str, float] = field(default_factory=dict)
    intercept: float = 0.0
    home_coef: float = 0.0
    rho: float = DEFAULT_RHO

    def fit(self, df: pd.DataFrame) -> PoissonGoals:
        played = df.dropna(subset=["home_score", "away_score"]).copy()
        self.teams = sorted(pd.unique(played[["home_team", "away_team"]].values.ravel()))
        idx = {t: i for i, t in enumerate(self.teams)}
        n = len(self.teams)

        # Two rows per match: each team's scoring event.
        rows, goals = [], []
        for r in played.itertuples(index=False):
            neutral = bool(getattr(r, "neutral", False))
            # home scoring
            rows.append((idx[r.home_team], idx[r.away_team], 0.0 if neutral else 1.0))
            goals.append(r.home_score)
            # away scoring
            rows.append((idx[r.away_team], idx[r.home_team], 0.0))
            goals.append(r.away_score)

        X = np.zeros((len(rows), 2 * n + 1))
        for k, (att, deff, home) in enumerate(rows):
            X[k, att] = 1.0
            X[k, n + deff] = 1.0
            X[k, 2 * n] = home
        y = np.asarray(goals, dtype=float)

        model = PoissonRegressor(alpha=1e-4, max_iter=400)
        model.fit(X, y)
        coef = model.coef_
        self.intercept = float(model.intercept_)
        self.attack = {t: float(coef[idx[t]]) for t in self.teams}
        self.defense = {t: float(coef[n + idx[t]]) for t in self.teams}
        self.home_coef = float(coef[2 * n])
        return self

    def _mean(self, team: str) -> float:
        # fallback to average strength for unseen teams
        return self.attack.get(team, float(np.mean(list(self.attack.values()) or [0.0])))

    def expected_goals(self, home: str, away: str, neutral: bool = False) -> tuple[float, float]:
        home_flag = 0.0 if neutral else 1.0
        lam_h = np.exp(
            self.intercept + self._mean(home)
            + self.defense.get(away, 0.0) + self.home_coef * home_flag
        )
        lam_a = np.exp(self.intercept + self._mean(away) + self.defense.get(home, 0.0))
        return float(lam_h), float(lam_a)

    def _dc_tau(self, i: int, j: int, lam_h: float, lam_a: float) -> float:
        rho = self.rho
        if i == 0 and j == 0:
            return 1 - lam_h * lam_a * rho
        if i == 0 and j == 1:
            return 1 + lam_h * rho
        if i == 1 and j == 0:
            return 1 + lam_a * rho
        if i == 1 and j == 1:
            return 1 - rho
        return 1.0

    def scoreline_matrix(self, home: str, away: str, neutral: bool = False) -> np.ndarray:
        lam_h, lam_a = self.expected_goals(home, away, neutral)
        ph = poisson.pmf(np.arange(MAX_GOALS + 1), lam_h)
        pa = poisson.pmf(np.arange(MAX_GOALS + 1), lam_a)
        mat = np.outer(ph, pa)
        for i in range(2):
            for j in range(2):
                mat[i, j] *= self._dc_tau(i, j, lam_h, lam_a)
        return mat / mat.sum()

    def outcome_probs(self, home: str, away: str, neutral: bool = False) -> dict[str, float]:
        mat = self.scoreline_matrix(home, away, neutral)
        return {
            "p_home_win": float(np.tril(mat, -1).sum()),
            "p_draw": float(np.trace(mat)),
            "p_away_win": float(np.triu(mat, 1).sum()),
        }

    def most_likely_score(self, home: str, away: str, neutral: bool = False) -> tuple[int, int]:
        mat = self.scoreline_matrix(home, away, neutral)
        i, j = np.unravel_index(int(mat.argmax()), mat.shape)
        return int(i), int(j)
