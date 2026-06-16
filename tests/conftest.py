"""Shared fixtures: a small, fast trained bundle for tests."""
from __future__ import annotations

import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

from goalcast.api.predictor import Predictor
from goalcast.data.ingest import _generate_synthetic
from goalcast.features.build import FEATURE_COLUMNS, build_features
from goalcast.features.elo import current_ratings
from goalcast.models.bundle import ModelBundle, build_team_snapshot
from goalcast.models.poisson import PoissonGoals


@pytest.fixture(scope="session")
def synthetic_matches() -> pd.DataFrame:
    df = _generate_synthetic(seed=1)
    keep = sorted(pd.unique(df[["home_team", "away_team"]].values.ravel()))[:8]
    df = df[df["home_team"].isin(keep) & df["away_team"].isin(keep)].reset_index(drop=True)
    return df


@pytest.fixture(scope="session")
def features(synthetic_matches: pd.DataFrame) -> pd.DataFrame:
    return build_features(synthetic_matches.copy())


@pytest.fixture(scope="session")
def bundle(features: pd.DataFrame) -> ModelBundle:
    params = {
        "max_depth": 3, "n_estimators": 40, "learning_rate": 0.1,
        "objective": "multi:softprob", "num_class": 3, "tree_method": "hist",
    }
    clf = CalibratedClassifierCV(XGBClassifier(**params), method="isotonic", cv=2)
    clf.fit(features[FEATURE_COLUMNS], features["outcome"])
    poisson = PoissonGoals().fit(features)
    return ModelBundle(
        version="test", feature_columns=FEATURE_COLUMNS, classifier=clf, poisson=poisson,
        elo_ratings=current_ratings(features), team_snapshot=build_team_snapshot(features),
        teams=sorted(poisson.teams), metrics={"log_loss": 1.0},
    )


@pytest.fixture(scope="session")
def predictor(bundle: ModelBundle) -> Predictor:
    return Predictor(bundle)
