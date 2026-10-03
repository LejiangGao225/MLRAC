"""
mrmr_selector.py
=================
MRMRSelector
------------
Implements Section 3.1 ("Feature Engineering and mRMR Selection"),
Eq. (1)-(3), Fig. 2 of the manuscript: minimum Redundancy Maximum
Relevance feature selection.
"""

import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import GradientBoostingRegressor

from .model_evaluator import ModelEvaluator


class MRMRSelector:
    """
    Minimum Redundancy Maximum Relevance feature selection.

    fit():       ranks candidate predictors and chooses the subset size
                 that minimizes validation RMSE of a probe GBM model.
    transform(): returns only the selected columns.
    """

    def __init__(self, max_features=None, seed=42):
        self.max_features = max_features
        self.seed = seed
        self.order_ = None
        self.relevance_ = None
        self.selected_idx_ = None
        self.selected_names_ = None
        self.validation_curve_ = None
        self._mi_cache = {}

    def _pairwise_mi(self, X, i, j):
        key = (min(i, j), max(i, j))
        if key not in self._mi_cache:
            self._mi_cache[key] = mutual_info_regression(
                X[:, [i]], X[:, j], random_state=self.seed)[0]
        return self._mi_cache[key]

    def _mrmr_rank(self, X, y):
        n_features = X.shape[1]
        relevance = mutual_info_regression(X, y, random_state=self.seed)   # D(S,Y) basis
        selected = [int(np.argmax(relevance))]
        remaining = [i for i in range(n_features) if i != selected[0]]

        while remaining:
            scores = []
            for f in remaining:
                redundancy = np.mean([self._pairwise_mi(X, f, s) for s in selected])  # Eq.(2)
                scores.append(relevance[f] - redundancy)                              # Eq.(3)
            best_local = int(np.argmax(scores))
            selected.append(remaining.pop(best_local))

        return selected, relevance

    def fit(self, X_train, y_train, X_val, y_val, feature_names):
        self._mi_cache = {}
        max_features = self.max_features or X_train.shape[1]
        order, relevance = self._mrmr_rank(X_train, y_train)
        self.order_, self.relevance_ = order, relevance

        best_k, best_rmse, history = 1, np.inf, []
        for k in range(1, max_features + 1):
            idx = order[:k]
            probe = GradientBoostingRegressor(
                n_estimators=150, max_depth=3, learning_rate=0.05,
                random_state=self.seed)
            probe.fit(X_train[:, idx], y_train)
            rmse = ModelEvaluator.rmse(y_val, probe.predict(X_val[:, idx]))
            history.append(rmse)
            if rmse < best_rmse - 1e-6:
                best_rmse, best_k = rmse, k

        self.selected_idx_ = order[:best_k]
        self.selected_names_ = [feature_names[i] for i in self.selected_idx_]
        self.validation_curve_ = history
        return self

    def transform(self, X):
        return X[:, self.selected_idx_]

    def fit_transform(self, X_train, y_train, X_val, y_val, feature_names):
        self.fit(X_train, y_train, X_val, y_val, feature_names)
        return self.transform(X_train)