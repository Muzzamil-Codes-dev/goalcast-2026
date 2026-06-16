"""Rolling form & goal features, computed point-in-time (shifted, no leakage)."""
from __future__ import annotations

import pandas as pd


def add_form_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Add rolling points/goals for home & away teams using only prior matches.

    Builds a long per-team view, computes shifted rolling stats, then merges back
    onto the home and away sides by (match_idx, team).
    """
    df = df.sort_values("match_date").reset_index(drop=True)
    df["match_idx"] = df.index

    def long_view(side: str) -> pd.DataFrame:
        opp = "away" if side == "home" else "home"
        gf = df[f"{side}_score"]
        ga = df[f"{opp}_score"]
        points = (gf > ga).astype(float) * 3 + (gf == ga).astype(float)
        return pd.DataFrame({
            "match_idx": df["match_idx"],
            "match_date": df["match_date"],
            "team": df[f"{side}_team"],
            "points": points,
            "gf": gf,
            "ga": ga,
        })

    long = pd.concat([long_view("home"), long_view("away")], ignore_index=True)
    long = long.sort_values(["team", "match_date", "match_idx"])

    grp = long.groupby("team", group_keys=False)
    long["form"] = grp["points"].apply(lambda s: s.shift().rolling(window, min_periods=1).mean())
    long["roll_gf"] = grp["gf"].apply(lambda s: s.shift().rolling(window, min_periods=1).mean())
    long["roll_ga"] = grp["ga"].apply(lambda s: s.shift().rolling(window, min_periods=1).mean())
    long["last_date"] = grp["match_date"].shift()

    cols = ["match_idx", "team", "form", "roll_gf", "roll_ga", "last_date"]
    home_keys = df[["match_idx", "home_team"]].rename(columns={"home_team": "team"})
    away_keys = df[["match_idx", "away_team"]].rename(columns={"away_team": "team"})
    hf = home_keys.merge(long[cols], on=["match_idx", "team"], how="left")
    af = away_keys.merge(long[cols], on=["match_idx", "team"], how="left")

    df["home_form"] = hf["form"].to_numpy()
    df["home_gf"] = hf["roll_gf"].to_numpy()
    df["home_ga"] = hf["roll_ga"].to_numpy()
    df["away_form"] = af["form"].to_numpy()
    df["away_gf"] = af["roll_gf"].to_numpy()
    df["away_ga"] = af["roll_ga"].to_numpy()

    home_rest = (df["match_date"] - pd.to_datetime(hf["last_date"].to_numpy())).dt.days
    away_rest = (df["match_date"] - pd.to_datetime(af["last_date"].to_numpy())).dt.days
    df["home_rest_days"] = home_rest.fillna(30).clip(0, 365)
    df["away_rest_days"] = away_rest.fillna(30).clip(0, 365)

    return df.drop(columns=["match_idx"])
