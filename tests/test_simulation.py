"""Tests for the Monte Carlo simulation."""
from __future__ import annotations

from goalcast.simulation.monte_carlo import simulate_group, simulate_tournament


def test_group_probabilities_valid(bundle):
    teams = bundle.teams[:4]
    df = simulate_group(bundle.poisson, teams, n_sims=500, seed=0)
    assert len(df) == 4
    assert df["p_advance"].between(0, 1).all()
    # exactly two teams advance each sim -> probabilities sum to ~2
    assert abs(df["p_advance"].sum() - 2.0) < 0.05


def test_tournament_champion_probs_sum_to_one(bundle):
    groups = {"A": bundle.teams[:4], "B": bundle.teams[4:8]}
    df = simulate_tournament(bundle.poisson, groups, n_sims=300, seed=0)
    assert abs(df["p_champion"].sum() - 1.0) < 1e-6
    assert df["p_champion"].between(0, 1).all()
