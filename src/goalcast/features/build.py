"""Assemble the model-ready feature table: raw matches -> features -> parquet/DB."""
from __future__ import annotations

import pandas as pd

from goalcast.config import DATA_DIR, settings
from goalcast.data.db import Match, get_session
from goalcast.features.elo import add_elo_features
from goalcast.features.form import add_form_features
from goalcast.logging_conf import get_logger

log = get_logger(__name__)

# 2026 co-hosts get a home-style boost even on "neutral" ground.
HOSTS_2026 = {"USA", "Canada", "Mexico"}

FEATURE_COLUMNS = [
    "elo_diff", "home_elo", "away_elo",
    "home_form", "away_form",
    "home_gf", "home_ga", "away_gf", "away_ga",
    "home_rest_days", "away_rest_days",
    "home_advantage",
]

# Outcome encoding for the W/D/L classifier.
OUTCOME_LABELS = {0: "home_win", 1: "draw", 2: "away_win"}


def _outcome(home_score: float, away_score: float) -> int:
    if home_score > away_score:
        return 0
    if home_score == away_score:
        return 1
    return 2


def load_matches() -> pd.DataFrame:
    """Prefer the raw parquet; fall back to the DB."""
    raw_path = DATA_DIR / "raw" / "matches.parquet"
    if raw_path.exists():
        return pd.read_parquet(raw_path)
    session = get_session()
    rows = session.query(Match).all()
    session.close()
    return pd.DataFrame([{
        "match_date": r.match_date, "home_team": r.home_team, "away_team": r.away_team,
        "home_score": r.home_score, "away_score": r.away_score,
        "tournament": r.tournament, "neutral": r.neutral,
    } for r in rows])


def build_features(df: pd.DataFrame | None = None) -> pd.DataFrame:
    if df is None:
        df = load_matches()
    df["match_date"] = pd.to_datetime(df["match_date"])
    df = add_elo_features(df)
    df = add_form_features(df, window=settings.form_window)

    df["home_advantage"] = (~df["neutral"].astype(bool)).astype(float)
    df.loc[df["home_team"].isin(HOSTS_2026), "home_advantage"] += 0.5

    played = df.dropna(subset=["home_score", "away_score"]).copy()
    played["outcome"] = [
        _outcome(h, a) for h, a in zip(played["home_score"], played["away_score"], strict=True)
    ]
    played = played.dropna(subset=FEATURE_COLUMNS)
    log.info("Built features for %d played matches", len(played))
    return played


def main() -> None:
    settings.ensure_dirs()
    feats = build_features()
    out = DATA_DIR / "features" / "features.parquet"
    feats.to_parquet(out, index=False)
    log.info("Wrote feature table -> %s", out)


if __name__ == "__main__":
    main()
