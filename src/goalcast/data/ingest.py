"""Ingestion: raw match data -> validation -> quality check -> database.

Source priority:
  1. Public martj42 international results CSV (if reachable).
  2. Deterministic synthetic generator (offline / CI), based on latent team strengths
     so downstream models still learn a real signal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from goalcast.config import DATA_DIR, settings
from goalcast.data.db import Match, Team, get_session, init_db
from goalcast.data.quality import check
from goalcast.data.schema import validate_matches
from goalcast.logging_conf import get_logger

log = get_logger(__name__)

PUBLIC_CSV = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

# Latent strengths for synthetic generation (attack power, ~1.0 = average).
_SYNTH_TEAMS = {
    "Brazil": 1.9, "France": 1.85, "Argentina": 1.8, "England": 1.7, "Spain": 1.7,
    "Germany": 1.65, "Portugal": 1.6, "Netherlands": 1.55, "Belgium": 1.5, "Italy": 1.5,
    "Croatia": 1.4, "Uruguay": 1.4, "Mexico": 1.3, "USA": 1.3, "Colombia": 1.3,
    "Senegal": 1.25, "Japan": 1.25, "Morocco": 1.2, "Switzerland": 1.2, "Denmark": 1.2,
    "South Korea": 1.15, "Poland": 1.1, "Serbia": 1.1, "Ecuador": 1.05, "Ghana": 1.0,
    "Canada": 1.0, "Australia": 0.95, "Tunisia": 0.9, "Iran": 0.95, "Nigeria": 1.1,
    "Qatar": 0.8, "Saudi Arabia": 0.85,
}


def _download_public() -> pd.DataFrame | None:
    try:
        import requests

        log.info("Trying public dataset: %s", PUBLIC_CSV)
        r = requests.get(PUBLIC_CSV, timeout=15)
        r.raise_for_status()
        from io import StringIO

        df = pd.read_csv(StringIO(r.text))
        df = df.rename(columns={"date": "match_date"})
        return df[["match_date", "home_team", "away_team", "home_score",
                   "away_score", "tournament", "neutral"]]
    except Exception as exc:  # noqa: BLE001 - any failure -> fall back offline
        log.warning("Public source unavailable (%s); using synthetic data.", exc)
        return None


def _generate_synthetic(seed: int = 42, n_per_pair_year: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    teams = list(_SYNTH_TEAMS)
    rows = []
    for year in range(2010, 2026):
        for i, home in enumerate(teams):
            for away in teams[i + 1 :]:
                for _ in range(n_per_pair_year):
                    neutral = bool(rng.random() < 0.4)
                    home_adv = 0.0 if neutral else 0.35
                    lam_h = max(0.15, _SYNTH_TEAMS[home] + home_adv - 0.5 * _SYNTH_TEAMS[away] + 0.5)
                    lam_a = max(0.15, _SYNTH_TEAMS[away] - 0.5 * _SYNTH_TEAMS[home] + 0.5)
                    hs, as_ = int(rng.poisson(lam_h)), int(rng.poisson(lam_a))
                    month = int(rng.integers(1, 13))
                    rows.append({
                        "match_date": f"{year}-{month:02d}-15",
                        "home_team": home, "away_team": away,
                        "home_score": hs, "away_score": as_,
                        "tournament": rng.choice(
                            ["Friendly", "FIFA World Cup qualification", "FIFA World Cup"],
                            p=[0.5, 0.4, 0.1]),
                        "neutral": neutral,
                    })
    df = pd.DataFrame(rows)
    return df.sort_values("match_date").reset_index(drop=True)


def load_raw() -> pd.DataFrame:
    df = _download_public()
    if df is None:
        df = _generate_synthetic()
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    df = df.dropna(subset=["match_date", "home_team", "away_team"])
    df["neutral"] = df["neutral"].astype(bool)
    return df.reset_index(drop=True)


def ingest() -> pd.DataFrame:
    settings.ensure_dirs()
    df = load_raw()
    log.info("Loaded %d raw matches", len(df))

    validate_matches(df)
    log.info("Pandera validation passed")

    report = check(df)
    log.info("Quality: %s", report.as_dict())
    if not report.passed:
        raise ValueError(f"Data quality check failed: {report.as_dict()}")

    raw_path = DATA_DIR / "raw" / "matches.parquet"
    df.to_parquet(raw_path, index=False)

    init_db()
    session = get_session()
    session.query(Match).delete()
    session.query(Team).delete()
    for name in sorted(pd.unique(df[["home_team", "away_team"]].values.ravel())):
        session.add(Team(name=str(name)))
    for r in df.itertuples(index=False):
        session.add(Match(
            match_date=r.match_date.date(), home_team=r.home_team, away_team=r.away_team,
            home_score=int(r.home_score) if pd.notna(r.home_score) else None,
            away_score=int(r.away_score) if pd.notna(r.away_score) else None,
            tournament=r.tournament, neutral=bool(r.neutral),
            is_future=bool(pd.isna(r.home_score)),
        ))
    session.commit()
    session.close()
    log.info("Wrote %d matches to DB (%s)", len(df), settings.database_url)
    return df


if __name__ == "__main__":
    ingest()
