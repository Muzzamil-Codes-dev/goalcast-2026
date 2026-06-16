"""Serializable model bundle — everything the API needs to score any matchup offline."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from goalcast.config import ARTIFACT_DIR
from goalcast.models.poisson import PoissonGoals

BUNDLE_PATH = ARTIFACT_DIR / "model_bundle.joblib"


@dataclass
class ModelBundle:
    version: str
    feature_columns: list[str]
    classifier: Any  # CalibratedClassifierCV
    poisson: PoissonGoals
    elo_ratings: dict[str, float]
    team_snapshot: dict[str, dict[str, float]]
    teams: list[str]
    metrics: dict[str, float] = field(default_factory=dict)

    def save(self, path: Path | None = None) -> Path:
        path = path or BUNDLE_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: Path | None = None) -> ModelBundle:
        return joblib.load(path or BUNDLE_PATH)

    @staticmethod
    def exists(path: Path | None = None) -> bool:
        return (path or BUNDLE_PATH).exists()


def build_team_snapshot(feat: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Latest known rolling form/goals per team, for building live feature rows."""
    snap: dict[str, dict[str, float]] = {}
    feat = feat.sort_values("match_date")
    for side, opp in (("home", "away"), ("away", "home")):  # noqa: B007
        for r in feat.itertuples(index=False):
            team = getattr(r, f"{side}_team")
            snap[team] = {
                "form": float(getattr(r, f"{side}_form")),
                "gf": float(getattr(r, f"{side}_gf")),
                "ga": float(getattr(r, f"{side}_ga")),
            }
    return snap
