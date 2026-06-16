"""Calibrated XGBoost classifier for win/draw/loss.

XGBoost tends to be overconfident out of the box, so I wrap it in isotonic calibration to
keep the probabilities (and the Brier score) honest.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

DEFAULT_PARAMS = {
    "max_depth": 4,
    "n_estimators": 350,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "tree_method": "hist",
}


def build_classifier(params: dict | None = None) -> CalibratedClassifierCV:
    base = XGBClassifier(**(params or DEFAULT_PARAMS))
    return CalibratedClassifierCV(base, method="isotonic", cv=3)


def train_classifier(
    X: pd.DataFrame, y: pd.Series, params: dict | None = None
) -> CalibratedClassifierCV:
    model = build_classifier(params)
    model.fit(X, y)
    return model


def predict_proba(model: CalibratedClassifierCV, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)
