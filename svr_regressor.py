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
		
		

"""
svr_regressor.py
==================
SVRRegressor
-------------
Implements Section 3.4 ("Support Vector Regression"), Eq. (11)-(14)
of the manuscript. The penalty parameter C, epsilon-insensitive
tolerance, and RBF kernel coefficient gamma are optimized via CGO.
Predictors are standardized using parameters fit exclusively on the
training data (as described in the manuscript).
"""

from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .base_cgo_regressor import BaseCGOTunedRegressor
from .sklearn_adapter import SklearnRegressorAdapter


class SVRRegressor(BaseCGOTunedRegressor):
    """RBF-kernel Support Vector Regression with CGO-tuned C, epsilon, gamma."""

    bounds = [(0.1, 100.0), (0.001, 2.0), (0.001, 1.0)]

    def _build_model(self, theta):
        C, eps, gamma = [float(v) for v in theta]
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svr', SVR(C=C, epsilon=eps, gamma=gamma, kernel='rbf'))
        ])
        return SklearnRegressorAdapter(pipeline)