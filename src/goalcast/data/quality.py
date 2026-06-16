"""Lightweight data-quality checks run after ingestion."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class QualityReport:
    n_rows: int
    n_duplicates: int
    n_missing_scores: int
    n_invalid_scores: int
    n_self_matches: int
    n_unique_teams: int

    @property
    def passed(self) -> bool:
        return self.n_invalid_scores == 0 and self.n_self_matches == 0

    def as_dict(self) -> dict:
        return self.__dict__ | {"passed": self.passed}


def check(df: pd.DataFrame) -> QualityReport:
    played = df.dropna(subset=["home_score", "away_score"])
    invalid = played[
        (played["home_score"] < 0)
        | (played["away_score"] < 0)
        | (played["home_score"] > 40)
        | (played["away_score"] > 40)
    ]
    dup_cols = ["match_date", "home_team", "away_team"]
    teams = pd.unique(df[["home_team", "away_team"]].values.ravel())
    return QualityReport(
        n_rows=len(df),
        n_duplicates=int(df.duplicated(subset=dup_cols).sum()),
        n_missing_scores=int(df[["home_score", "away_score"]].isna().any(axis=1).sum()),
        n_invalid_scores=len(invalid),
        n_self_matches=int((df["home_team"] == df["away_team"]).sum()),
        n_unique_teams=len(teams),
    )
