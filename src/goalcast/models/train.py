"""Training orchestrator with MLflow tracking.

Flow: features -> time split -> baselines (lift check) -> calibrated XGBoost + Poisson
-> log to MLflow -> refit on all data -> save bundle -> record run in DB.
"""
from __future__ import annotations

import uuid

import mlflow

from goalcast.config import settings
from goalcast.data.db import ModelRun, get_session, init_db
from goalcast.features.build import build_features
from goalcast.features.elo import current_ratings
from goalcast.logging_conf import get_logger
from goalcast.models.baseline import EloBaseline, MajorityBaseline
from goalcast.models.bundle import ModelBundle, build_team_snapshot
from goalcast.models.classifier import train_classifier
from goalcast.models.dataset import time_split
from goalcast.models.evaluate import classification_metrics, goals_mae
from goalcast.models.poisson import PoissonGoals

log = get_logger(__name__)


def train() -> ModelBundle:
    settings.ensure_dirs()
    feats = build_features()
    split = time_split(feats)
    log.info("Train=%d  Test=%d matches", len(split.train), len(split.test))

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("goalcast")
    version = uuid.uuid4().hex[:8]

    with mlflow.start_run(run_name=f"train-{version}"):
        # --- Baselines (the bar to beat) ---
        for name, model in (("majority", MajorityBaseline()), ("elo_only", EloBaseline())):
            model.fit(split.X_train, split.y_train)
            m = classification_metrics(split.y_test, model.predict_proba(split.X_test))
            log.info("baseline[%s]: %s", name, m)
            mlflow.log_metrics({f"{name}_{k}": v for k, v in m.items()})

        # --- Main classifier + Poisson goals ---
        clf = train_classifier(split.X_train, split.y_train)
        poisson_eval = PoissonGoals().fit(split.train)

        proba = clf.predict_proba(split.X_test)
        metrics = classification_metrics(split.y_test, proba)
        metrics |= goals_mae(poisson_eval, split.test)
        log.info("XGBoost(calibrated) test: %s", metrics)

        mlflow.log_params({"model": "xgboost+isotonic", "n_features": len(split.feature_columns)})
        mlflow.log_metrics(metrics)
        try:
            mlflow.sklearn.log_model(clf, "model")
        except Exception as exc:  # noqa: BLE001 - file store can't register; bundle is source of truth
            log.warning("MLflow model log skipped: %s", exc)

    # --- Refit on ALL data for serving, then bundle ---
    clf_full = train_classifier(feats[split.feature_columns], feats["outcome"])
    poisson_full = PoissonGoals().fit(feats)
    bundle = ModelBundle(
        version=version,
        feature_columns=split.feature_columns,
        classifier=clf_full,
        poisson=poisson_full,
        elo_ratings=current_ratings(feats),
        team_snapshot=build_team_snapshot(feats),
        teams=sorted(poisson_full.teams),
        metrics=metrics,
    )
    path = bundle.save()
    log.info("Saved bundle %s -> %s", version, path)

    _record_run(version, metrics)
    return bundle


def _record_run(version: str, metrics: dict) -> None:
    init_db()
    session = get_session()
    session.merge(ModelRun(
        run_id=version, model_name=settings.model_name, version=version, stage="staging",
        log_loss=metrics.get("log_loss"), brier=metrics.get("brier"),
        accuracy=metrics.get("accuracy"), mae_goals=metrics.get("mae_goals"),
    ))
    session.commit()
    session.close()


if __name__ == "__main__":
    train()
