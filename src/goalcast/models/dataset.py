"""Load the feature table and produce a time-based train/test split (no leakage)."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from goalcast.config import DATA_DIR, settings
from goalcast.features.build import FEATURE_COLUMNS, build_features


@dataclass
class SplitData:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    train: pd.DataFrame
    test: pd.DataFrame
    feature_columns: list[str]


def load_feature_table() -> pd.DataFrame:
    path = DATA_DIR / "features" / "features.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return build_features()


def time_split(df: pd.DataFrame | None = None, split_date: str | None = None) -> SplitData:
    df = load_feature_table() if df is None else df
    split_date = split_date or settings.test_split_date
    df = df.sort_values("match_date")
    cut = pd.Timestamp(split_date)
    train = df[df["match_date"] < cut]
    test = df[df["match_date"] >= cut]
    return SplitData(
        X_train=train[FEATURE_COLUMNS], y_train=train["outcome"],
        X_test=test[FEATURE_COLUMNS], y_test=test["outcome"],
        train=train, test=test, feature_columns=FEATURE_COLUMNS,
    )
