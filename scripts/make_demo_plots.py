"""Generate the README demo plots — tournament winner odds + a scoreline heatmap —
to match the committed ``docs/screenshots/calibration.png``.

These three images make up the README "Demo" table. ``calibration.png`` is written
by ``goalcast.models.evaluate``; this script writes the other two from the trained
model bundle, so the whole table is reproducible rather than hand-pasted.

Run after ``make pipeline`` (needs ``models/artifacts/model_bundle.joblib``)::

    python scripts/make_demo_plots.py
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

from goalcast.api.predictor import Predictor  # noqa: E402
from goalcast.config import ROOT  # noqa: E402
from goalcast.simulation.monte_carlo import simulate_tournament  # noqa: E402

PLOT_DIR = ROOT / "docs" / "screenshots"
TEAL = "#1b9e8f"

# A credible 16-team demo field — 2026 hosts (USA/Mexico/Canada) + top contenders —
# so the bracket reads like a real World Cup rather than an alphabetical slice.
DEMO_FIELD = [
    "Argentina", "France", "Brazil", "England", "Spain", "Germany", "Portugal",
    "Netherlands", "Belgium", "Croatia", "Uruguay", "Morocco", "Japan",
    "United States", "Mexico", "Canada",
]


def winner_plot(df, path) -> None:
    top = df[df["p_champion"] > 0].head(10).iloc[::-1]  # reverse: highest bar on top
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.barh(top["team"], top["p_champion"], color=TEAL)
    ax.bar_label(bars, labels=[f"{v:.1%}" for v in top["p_champion"]], padding=3, fontsize=9)
    ax.set(xlabel="P(champion)", title="Tournament winner odds (Monte Carlo)")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.margins(x=0.16)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def scoreline_plot(poisson, home: str, away: str, path, maxg: int = 5) -> None:
    mat = poisson.scoreline_matrix(home, away, neutral=True)[: maxg + 1, : maxg + 1]
    fig, ax = plt.subplots(figsize=(5.6, 5))
    im = ax.imshow(mat, cmap="Greens", origin="upper")
    hi = mat.max()
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(
                j, i, f"{mat[i, j]:.0%}", ha="center", va="center", fontsize=8,
                color="white" if mat[i, j] > hi * 0.6 else "#222",
            )
    ax.set(
        xticks=range(maxg + 1), yticks=range(maxg + 1),
        xlabel=f"{away} goals", ylabel=f"{home} goals",
        title=f"Scoreline probabilities\n{home} vs {away}",
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    pred = Predictor.load()
    teams = pred.teams

    per_group = 4
    # Prefer the curated marquee field; fall back to whatever's available.
    pool = [t for t in DEMO_FIELD if t in teams]
    if len(pool) < 16:
        pool = (pool + [t for t in teams if t not in pool])[:16]
    n_groups = 4 if len(pool) >= 16 else 2
    pool = pool[: n_groups * per_group]
    groups = {chr(65 + i): pool[i * per_group:(i + 1) * per_group] for i in range(n_groups)}

    df = simulate_tournament(pred.b.poisson, groups, n_sims=2000, seed=42)
    winner_plot(df, PLOT_DIR / "winner.png")

    home, away = df.iloc[0]["team"], df.iloc[1]["team"]
    scoreline_plot(pred.b.poisson, home, away, PLOT_DIR / "scoreline.png")

    print(f"Saved winner.png + scoreline.png to {PLOT_DIR}")
    print(f"Projected final used for the scoreline heatmap: {home} vs {away}")


if __name__ == "__main__":
    main()
