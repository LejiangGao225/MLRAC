"""
gbm_regressor.py
==================
GBMRegressor
-------------
Implements Section 3.3 ("Gradient Boosting Machine"), Eq. (9)-(10)
of the manuscript. Hyperparameters (number of estimators, maximum
tree depth, learning rate) are optimized via the Chaos Game Optimizer.
"""

from sklearn.ensemble import GradientBoostingRegressor

from .base_cgo_regressor import BaseCGOTunedRegressor
from .sklearn_adapter import SklearnRegressorAdapter


class GBMRegressor(BaseCGOTunedRegressor):
    """Gradient Boosting Machine with CGO-tuned n_estimators, max_depth, learning_rate."""

    bounds = [(50, 400), (2, 8), (0.01, 0.3)]

    def _build_model(self, theta):
        n_estimators = int(round(theta[0]))
        max_depth = int(round(theta[1]))
        lr = float(theta[2])
        estimator = GradientBoostingRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=lr, subsample=0.9, random_state=self.seed)
        return SklearnRegressorAdapter(estimator)