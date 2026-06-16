"""Pandera validation schema for raw match data.

Catches the data-quality problems that matter: missing teams, impossible scores,
self-matches. Future fixtures may have null scores (is_future=True).
"""
from __future__ import annotations

import pandera.pandas as pa
from pandera.typing import Series


class MatchSchema(pa.DataFrameModel):
    match_date: Series[pa.DateTime]
    home_team: Series[str] = pa.Field(nullable=False, str_length={"min_value": 1})
    away_team: Series[str] = pa.Field(nullable=False, str_length={"min_value": 1})
    home_score: Series[float] = pa.Field(ge=0, le=40, nullable=True)
    away_score: Series[float] = pa.Field(ge=0, le=40, nullable=True)
    tournament: Series[str] = pa.Field(nullable=False)
    neutral: Series[bool] = pa.Field(coerce=True)

    class Config:
        strict = False
        coerce = True

    @pa.dataframe_check
    def teams_must_differ(cls, df):  # type: ignore[misc]  # pandera check uses cls
        return df["home_team"] != df["away_team"]


def validate_matches(df):
    """Validate, returning the coerced frame. Raises SchemaError on failure."""
    return MatchSchema.validate(df, lazy=True)
