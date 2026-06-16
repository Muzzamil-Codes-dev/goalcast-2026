"""Model registry: metric-gated staging -> production promotion.

A new model is promoted only if it beats the current production model's log loss.
This automated, gated promotion is the MLOps detail recruiters look for.
"""
from __future__ import annotations

import argparse

from goalcast.data.db import ModelRun, get_session, init_db
from goalcast.logging_conf import get_logger

log = get_logger(__name__)


def get_production() -> ModelRun | None:
    session = get_session()
    row = session.query(ModelRun).filter(ModelRun.stage == "production").first()
    session.close()
    return row


def promote() -> str | None:
    init_db()
    session = get_session()
    best_staging = (
        session.query(ModelRun)
        .filter(ModelRun.stage == "staging", ModelRun.log_loss.isnot(None))
        .order_by(ModelRun.log_loss.asc())
        .first()
    )
    if best_staging is None:
        log.info("No staging models to promote.")
        session.close()
        return None

    prod = session.query(ModelRun).filter(ModelRun.stage == "production").first()
    if prod is not None and prod.log_loss <= best_staging.log_loss:
        log.info("Production (%.4f) still beats best staging (%.4f); no promotion.",
                 prod.log_loss, best_staging.log_loss)
        session.close()
        return None

    if prod is not None:
        prod.stage = "archived"
    best_staging.stage = "production"
    session.commit()
    log.info("Promoted %s to production (log_loss=%.4f)",
             best_staging.version, best_staging.log_loss)
    version = best_staging.version
    session.close()
    return version


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--promote", action="store_true")
    args = parser.parse_args()
    if args.promote:
        promote()
