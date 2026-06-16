"""Baseline models. The bar XGBoost must clear before we celebrate."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


class MajorityBaseline:
    """Predicts the class base-rates seen in training for every match."""

    def __init__(self) -> None:
        self.rates_: np.ndarray | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> MajorityBaseline:
        counts = pd.Series(y).value_counts(normalize=True).sort_index()
        self.rates_ = np.array([counts.get(i, 0.0) for i in range(3)])
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        assert self.rates_ is not None
        return np.tile(self.rates_, (len(X), 1))


class EloBaseline:
    """Logistic regression on a single feature: elo_diff. Surprisingly strong."""

    def __init__(self) -> None:
        self.model = LogisticRegression(max_iter=1000)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> EloBaseline:
        self.model.fit(X[["elo_diff"]], y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X[["elo_diff"]])
