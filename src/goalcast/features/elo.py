"""Chronological Elo rating engine for international football.

Point-in-time safe: ratings are updated AFTER each match, so the rating used as a
feature for match N reflects only matches < N. No leakage.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

BASE_RATING = 1500.0
HOME_ADVANTAGE = 65.0

TOURNAMENT_WEIGHT = {
    "FIFA World Cup": 60.0,
    "FIFA World Cup qualification": 40.0,
    "UEFA Euro": 50.0,
    "Copa América": 50.0,
    "Friendly": 20.0,
}
DEFAULT_WEIGHT = 30.0


@dataclass
class EloEngine:
    ratings: dict[str, float] = field(default_factory=dict)

    def get(self, team: str) -> float:
        return self.ratings.setdefault(team, BASE_RATING)

    @staticmethod
    def expected(rating_a: float, rating_b: float) -> float:
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    @staticmethod
    def _margin_factor(goal_diff: int) -> float:
        if goal_diff <= 1:
            return 1.0
        if goal_diff == 2:
            return 1.5
        return (11 + goal_diff) / 8.0  # diminishing boost for blowouts

    def update(
        self,
        home: str,
        away: str,
        home_goals: int,
        away_goals: int,
        tournament: str = "Friendly",
        neutral: bool = False,
    ) -> tuple[float, float]:
        """Apply one result. Returns the PRE-match ratings (leakage-safe feature values)."""
        r_home, r_away = self.get(home), self.get(away)
        pre_home, pre_away = r_home, r_away

        adj = 0.0 if neutral else HOME_ADVANTAGE
        exp_home = self.expected(r_home + adj, r_away)
        score_home = 1.0 if home_goals > away_goals else 0.5 if home_goals == away_goals else 0.0

        k = TOURNAMENT_WEIGHT.get(tournament, DEFAULT_WEIGHT)
        k *= self._margin_factor(abs(home_goals - away_goals))
        delta = k * (score_home - exp_home)

        self.ratings[home] = r_home + delta
        self.ratings[away] = r_away - delta
        return pre_home, pre_away


def add_elo_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add home_elo, away_elo, elo_diff to a chronologically-sorted match frame."""
    df = df.sort_values("match_date").reset_index(drop=True)
    engine = EloEngine()
    home_elos, away_elos = [], []
    for r in df.itertuples(index=False):
        played = pd.notna(r.home_score) and pd.notna(r.away_score)
        if played:
            pre_h, pre_a = engine.update(
                r.home_team, r.away_team, int(r.home_score), int(r.away_score),
                getattr(r, "tournament", "Friendly"), bool(getattr(r, "neutral", False)),
            )
        else:
            pre_h, pre_a = engine.get(r.home_team), engine.get(r.away_team)
        home_elos.append(pre_h)
        away_elos.append(pre_a)
    df["home_elo"] = home_elos
    df["away_elo"] = away_elos
    df["elo_diff"] = df["home_elo"] - df["away_elo"]
    return df


def current_ratings(df: pd.DataFrame) -> dict[str, float]:
    """Return the latest Elo rating per team after replaying all played matches."""
    engine = EloEngine()
    for r in df.sort_values("match_date").itertuples(index=False):
        if pd.notna(r.home_score) and pd.notna(r.away_score):
            engine.update(
                r.home_team, r.away_team, int(r.home_score), int(r.away_score),
                getattr(r, "tournament", "Friendly"), bool(getattr(r, "neutral", False)),
            )
    return dict(engine.ratings)
