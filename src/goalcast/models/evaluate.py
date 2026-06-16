"""Evaluation: probabilistic metrics + calibration/confusion plots.

For a forecasting problem the probabilities matter more than the hard label, so I score
them with log loss, Brier score and calibration curves rather than accuracy on its own.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from goalcast.config import ROOT
from goalcast.logging_conf import get_logger

log = get_logger(__name__)
PLOT_DIR = ROOT / "docs" / "screenshots"


def multiclass_brier(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Mean over the 3 one-vs-rest Brier scores."""
    y_true = np.asarray(y_true)
    return float(
        np.mean([brier_score_loss((y_true == k).astype(int), proba[:, k]) for k in range(3)])
    )


def classification_metrics(y_true, proba) -> dict[str, float]:
    preds = proba.argmax(axis=1)
    return {
        "log_loss": float(log_loss(y_true, proba, labels=[0, 1, 2])),
        "brier": multiclass_brier(y_true, proba),
        "accuracy": float(accuracy_score(y_true, preds)),
    }


def goals_mae(poisson_model, test: pd.DataFrame) -> dict[str, float]:
    errs_h, errs_a = [], []
    for r in test.itertuples(index=False):
        lam_h, lam_a = poisson_model.expected_goals(
            r.home_team, r.away_team, bool(getattr(r, "neutral", False))
        )
        errs_h.append(lam_h - r.home_score)
        errs_a.append(lam_a - r.away_score)
    errs = np.array(errs_h + errs_a)
    return {"mae_goals": float(np.mean(np.abs(errs))), "rmse_goals": float(np.sqrt(np.mean(errs**2)))}


def calibration_plot(y_true, proba, path=None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve

    y_true = np.asarray(y_true)
    fig, ax = plt.subplots(figsize=(5, 5))
    for k, name in enumerate(["home win", "draw", "away win"]):
        frac_pos, mean_pred = calibration_curve((y_true == k).astype(int), proba[:, k], n_bins=10)
        ax.plot(mean_pred, frac_pos, marker="o", label=name)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set(xlabel="Predicted probability", ylabel="Observed frequency", title="Calibration")
    ax.legend()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(path or PLOT_DIR / "calibration.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Honest held-out evaluation: train on the train split, score the test split.

    (The production bundle is refit on ALL data for serving, so it must NOT be scored on
    the test rows it has seen — we retrain a clean model here for unbiased metrics/plots.)
    """
    from goalcast.models.classifier import train_classifier
    from goalcast.models.dataset import time_split
    from goalcast.models.poisson import PoissonGoals

    split = time_split()
    clf = train_classifier(split.X_train, split.y_train)
    poisson = PoissonGoals().fit(split.train)
    proba = clf.predict_proba(split.X_test)
    metrics = classification_metrics(split.y_test, proba)
    metrics |= goals_mae(poisson, split.test)
    log.info("Held-out test metrics: %s", metrics)
    calibration_plot(split.y_test, proba)
    log.info("Saved calibration plot to docs/screenshots/")


if __name__ == "__main__":
    main()
