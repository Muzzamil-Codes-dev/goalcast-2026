"""Tests for feature engineering — correctness and leakage safety."""
from __future__ import annotations

import pandas as pd

from goalcast.features.build import FEATURE_COLUMNS
from goalcast.features.elo import EloEngine, add_elo_features


def test_elo_winner_gains_points():
    eng = EloEngine()
    eng.update("A", "B", 3, 0, "Friendly", neutral=True)
    assert eng.get("A") > 1500 > eng.get("B")


def test_elo_expected_symmetry():
    assert abs(EloEngine.expected(1500, 1500) - 0.5) < 1e-9


def test_elo_features_are_pre_match():
    # First match for both teams must use the base rating (no leakage from its own result).
    df = pd.DataFrame({
        "match_date": pd.to_datetime(["2020-01-01", "2020-02-01"]),
        "home_team": ["A", "A"], "away_team": ["B", "B"],
        "home_score": [5, 0], "away_score": [0, 0],
        "tournament": ["Friendly", "Friendly"], "neutral": [True, True],
    })
    out = add_elo_features(df)
    assert out.iloc[0]["home_elo"] == 1500
    # second match should reflect the first result
    assert out.iloc[1]["home_elo"] > 1500


def test_build_features_has_no_nans(features: pd.DataFrame):
    assert set(FEATURE_COLUMNS).issubset(features.columns)
    assert features[FEATURE_COLUMNS].isna().sum().sum() == 0
    assert features["outcome"].isin([0, 1, 2]).all()
