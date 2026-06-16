"""Tests for the classifier, Poisson model, and predictor."""
from __future__ import annotations


def test_classifier_probs_sum_to_one(predictor):
    pred = predictor.predict_match("Brazil", "Qatar", neutral=True) if \
        "Brazil" in predictor.teams else predictor.predict_match(
            predictor.teams[0], predictor.teams[1], neutral=True)
    total = pred["p_home_win"] + pred["p_draw"] + pred["p_away_win"]
    assert abs(total - 1.0) < 1e-6


def test_poisson_expected_goals_positive(bundle):
    a, b = bundle.teams[0], bundle.teams[1]
    lam_h, lam_a = bundle.poisson.expected_goals(a, b)
    assert lam_h > 0 and lam_a > 0


def test_scoreline_matrix_normalised(bundle):
    a, b = bundle.teams[0], bundle.teams[1]
    mat = bundle.poisson.scoreline_matrix(a, b)
    assert abs(mat.sum() - 1.0) < 1e-6
    probs = bundle.poisson.outcome_probs(a, b)
    assert abs(sum(probs.values()) - 1.0) < 1e-6


def test_stronger_team_more_likely_to_win(bundle):
    # team[0] has highest synthetic strength among the kept teams
    strong, weak = bundle.teams[0], bundle.teams[-1]
    p_strong = bundle.poisson.outcome_probs(strong, weak)["p_home_win"]
    p_weak = bundle.poisson.outcome_probs(weak, strong)["p_home_win"]
    assert p_strong > p_weak


def test_predict_scoreline_top_scores(predictor):
    res = predictor.predict_scoreline(predictor.teams[0], predictor.teams[1], neutral=True)
    probs = [s["prob"] for s in res["top_scores"]]
    assert len(probs) > 0
    assert probs == sorted(probs, reverse=True)  # ranked most-likely first
    assert all(0 <= p <= 1 for p in probs)
