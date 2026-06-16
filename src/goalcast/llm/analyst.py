"""AI Match Analyst: grounded, plain-English explanations of a prediction.

Combines (1) the model's own outputs, (2) instance-level feature drivers, and
(3) retrieved knowledge-base snippets. Uses Claude when ANTHROPIC_API_KEY is set;
otherwise a deterministic template (so the feature always works offline).
"""
from __future__ import annotations

from goalcast.api.predictor import Predictor
from goalcast.config import settings
from goalcast.features.elo import BASE_RATING
from goalcast.llm.rag import KnowledgeBase

_KB = KnowledgeBase()


def _drivers(predictor: Predictor, home: str, away: str, neutral: bool) -> list[str]:
    b = predictor.b
    home_elo = b.elo_ratings.get(home, BASE_RATING)
    away_elo = b.elo_ratings.get(away, BASE_RATING)
    hs, as_ = predictor._snap(home), predictor._snap(away)
    lam_h, lam_a = b.poisson.expected_goals(home, away, neutral)

    drivers = []
    gap = home_elo - away_elo
    favored = home if gap >= 0 else away
    drivers.append(f"Elo rating gap: {favored} is rated {abs(gap):.0f} points higher.")
    if abs(hs["form"] - as_["form"]) > 0.3:
        better = home if hs["form"] > as_["form"] else away
        drivers.append(f"Recent form favours {better} ({hs['form']:.1f} vs {as_['form']:.1f} pts/game).")
    drivers.append(f"Expected goals: {home} {lam_h:.1f} – {lam_a:.1f} {away}.")
    if not neutral:
        drivers.append(f"{home} has home advantage.")
    return drivers


def explain_prediction(predictor: Predictor, home: str, away: str, neutral: bool = False) -> dict:
    pred = predictor.predict_match(home, away, neutral)
    drivers = _drivers(predictor, home, away, neutral)
    snippets = _KB.retrieve(f"Elo form expected goals explain {home} {away}", k=2)
    sources = sorted({c.source for c in snippets})

    facts = (
        f"Match: {home} vs {away} ({'neutral' if neutral else 'home: ' + home}).\n"
        f"Model probabilities: {home} win {pred['p_home_win']:.0%}, "
        f"draw {pred['p_draw']:.0%}, {away} win {pred['p_away_win']:.0%}.\n"
        f"Expected goals: {pred['exp_home_goals']} – {pred['exp_away_goals']}. "
        f"Most likely score: {pred['likely_scoreline']}.\n"
        f"Key drivers: {' '.join(drivers)}\n"
        f"Background: {' '.join(c.text for c in snippets)}"
    )

    explanation, generated_by = _maybe_claude(facts), "claude"
    if explanation is None:
        explanation, generated_by = _template(home, away, pred, drivers), "template"

    return {
        "home_team": home, "away_team": away, "prediction": pred,
        "drivers": drivers, "sources": sources,
        "explanation": explanation, "generated_by": generated_by,
    }


def _maybe_claude(facts: str) -> str | None:
    if not settings.anthropic_api_key:
        return None
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        msg = client.messages.create(
            model=settings.llm_model,
            max_tokens=350,
            system=(
                "You are a football analyst. Explain the prediction in 3-4 sentences using "
                "ONLY the facts provided. Do not invent statistics. Be concise and concrete."
            ),
            messages=[{"role": "user", "content": facts}],
        )
        # content is a union of block types; only text blocks carry `.text`.
        return getattr(msg.content[0], "text", None)
    except Exception:  # noqa: BLE001 - any LLM failure falls back to template
        return None


def _template(home: str, away: str, pred: dict, drivers: list[str]) -> str:
    probs = {home: pred["p_home_win"], "a draw": pred["p_draw"], away: pred["p_away_win"]}
    top = max(probs, key=lambda k: probs[k])
    return (
        f"The model favours {top} ({probs[top]:.0%}) in {home} vs {away}. "
        f"It projects an expected scoreline of about {pred['exp_home_goals']}–"
        f"{pred['exp_away_goals']}, with {pred['likely_scoreline']} the single most likely result. "
        + " ".join(drivers)
    )
