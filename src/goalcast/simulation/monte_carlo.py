"""Monte Carlo tournament simulation driven by the Poisson goals model."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from goalcast.models.poisson import PoissonGoals


def simulate_match(
    poisson: PoissonGoals, home: str, away: str, neutral: bool, rng: np.random.Generator
) -> tuple[int, int]:
    lam_h, lam_a = poisson.expected_goals(home, away, neutral)
    return int(rng.poisson(lam_h)), int(rng.poisson(lam_a))


def _knockout_winner(
    poisson: PoissonGoals, a: str, b: str, rng: np.random.Generator
) -> str:
    gh, ga = simulate_match(poisson, a, b, neutral=True, rng=rng)
    if gh > ga:
        return a
    if ga > gh:
        return b
    # extra time / penalties -> weight by attacking strength
    lam_a, lam_b = poisson.expected_goals(a, b, neutral=True)
    return a if rng.random() < lam_a / (lam_a + lam_b) else b


def simulate_group(
    poisson: PoissonGoals, teams: list[str], n_sims: int = 5000, seed: int = 0
) -> pd.DataFrame:
    """Round-robin group. Returns P(advance, top 2), P(win group), expected points."""
    rng = np.random.default_rng(seed)
    advance: defaultdict[str, int] = defaultdict(int)
    won_group: defaultdict[str, int] = defaultdict(int)
    points_total: defaultdict[str, float] = defaultdict(float)

    for _ in range(n_sims):
        pts = dict.fromkeys(teams, 0)
        gd = dict.fromkeys(teams, 0)
        for i, h in enumerate(teams):
            for a in teams[i + 1 :]:
                gh, ga = simulate_match(poisson, h, a, neutral=True, rng=rng)
                gd[h] += gh - ga
                gd[a] += ga - gh
                if gh > ga:
                    pts[h] += 3
                elif ga > gh:
                    pts[a] += 3
                else:
                    pts[h] += 1
                    pts[a] += 1
        ranking = sorted(teams, key=lambda t: (pts[t], gd[t], rng.random()), reverse=True)
        for t in teams:
            points_total[t] += pts[t]
        won_group[ranking[0]] += 1
        for t in ranking[:2]:
            advance[t] += 1

    return (
        pd.DataFrame({
            "team": teams,
            "p_advance": [advance[t] / n_sims for t in teams],
            "p_win_group": [won_group[t] / n_sims for t in teams],
            "exp_points": [points_total[t] / n_sims for t in teams],
        })
        .sort_values("p_advance", ascending=False)
        .reset_index(drop=True)
    )


def simulate_tournament(
    poisson: PoissonGoals, groups: dict[str, list[str]], n_sims: int = 2000, seed: int = 0
) -> pd.DataFrame:
    """Group stage (top 2 advance) -> single-elimination bracket. Returns champion probs.

    Requires an even number of equal-size groups; pairs winners vs runners-up across
    adjacent groups, then runs single elimination.
    """
    rng = np.random.default_rng(seed)
    group_names = list(groups)
    if len(group_names) % 2 != 0:
        raise ValueError("Need an even number of groups for the bracket.")

    champions: defaultdict[str, int] = defaultdict(int)
    finalists: defaultdict[str, int] = defaultdict(int)
    all_teams = [t for g in groups.values() for t in g]

    for _ in range(n_sims):
        winners, runners = {}, {}
        for g, teams in groups.items():
            res = simulate_group(poisson, teams, n_sims=1, seed=int(rng.integers(1_000_000_000)))
            winners[g] = res.iloc[0]["team"]
            runners[g] = res.iloc[1]["team"]

        # seed bracket: A1-B2, B1-A2, C1-D2, D1-C2, ...
        bracket: list[str] = []
        for i in range(0, len(group_names), 2):
            g1, g2 = group_names[i], group_names[i + 1]
            bracket += [winners[g1], runners[g2], winners[g2], runners[g1]]

        round_teams = bracket
        final_two: list[str] = []
        while len(round_teams) > 1:
            nxt = [
                _knockout_winner(poisson, round_teams[k], round_teams[k + 1], rng)
                for k in range(0, len(round_teams), 2)
            ]
            if len(round_teams) == 2:
                final_two = round_teams
            round_teams = nxt
        for t in final_two:
            finalists[t] += 1
        champions[round_teams[0]] += 1

    return (
        pd.DataFrame({
            "team": all_teams,
            "p_champion": [champions[t] / n_sims for t in all_teams],
            "p_finalist": [finalists[t] / n_sims for t in all_teams],
        })
        .sort_values("p_champion", ascending=False)
        .reset_index(drop=True)
    )
