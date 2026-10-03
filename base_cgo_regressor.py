"""
base_cgo_regressor.py
========================
BaseCGOTunedRegressor
------------------------
Implements the shared CGO hyperparameter-tuning logic described in
Section 3.2 ("Chaos Game Optimizer"), Eq. (4):

    Fitness(theta) = mean RMSE over 5-fold cross-validation
                     within the training subset.

Concrete learners (GBM, SVR, TabNet) subclass this and only need to
define `bounds` and `_build_model(theta)`.
"""

import numpy as np
from sklearn.model_selection import KFold

from .chaos_game_optimizer import ChaosGameOptimizer
from .model_evaluator import ModelEvaluator


class BaseCGOTunedRegressor:
    """
    Shared logic for a regressor whose hyperparameters are tuned
    by the Chaos Game Optimizer using mean k-fold cross-validation
    RMSE as the fitness function (Eq. 4).
    """

    bounds = []  # overridden by subclasses

    def __init__(self, cv_folds=5, cgo_pop=12, cgo_iter=20, seed=42):
        self.cv_folds = cv_folds
        self.cgo_pop = cgo_pop
        self.cgo_iter = cgo_iter
        self.seed = seed
        self.best_theta_ = None
        self.best_cv_rmse_ = None
        self.convergence_ = None
        self.model_ = None

    def _build_model(self, theta):
        """Must be implemented by subclasses: returns a fitted-capable model
        instance built from the hyperparameter vector `theta`."""
        raise NotImplementedError

    def _fitness(self, theta, X, y):
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.seed)
        rmses = []
        for tr_idx, va_idx in kf.split(X):
            try:
                m = self._build_model(theta)
                m.fit(X[tr_idx], y[tr_idx])
                pred = m.predict(X[va_idx])
                rmses.append(ModelEvaluator.rmse(y[va_idx], pred))
            except Exception:
                rmses.append(1e6)
        return float(np.mean(rmses))

    def tune(self, X, y):
        """Runs CGO to find the best hyperparameter configuration (Eq. 4)."""
        optimizer = ChaosGameOptimizer(
            fitness_func=lambda theta: self._fitness(theta, X, y),
            bounds=self.bounds, n_pop=self.cgo_pop, n_iter=self.cgo_iter,
            seed=self.seed)
        best_theta, best_fit, hist_best, hist_mean = optimizer.optimize()
        self.best_theta_ = best_theta
        self.best_cv_rmse_ = best_fit
        self.convergence_ = {'best': hist_best, 'mean': hist_mean}
        return self

    def fit(self, X, y, X_val=None, y_val=None):
        if self.best_theta_ is None:
            self.tune(X, y)
        self.model_ = self._build_model(self.best_theta_)
        self.model_.fit(X, y, X_val, y_val)
        return self

    def predict(self, X):
        return self.model_.predict(X)