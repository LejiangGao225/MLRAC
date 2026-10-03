"""
ensemble_model.py
===================
EnsembleModel
--------------
Implements Section 3.2 ("Chaos Game Optimizer"), Eq. (5)-(6) of the
manuscript: combines fitted GBM, SVR, and TabNet predictions using
CGO-optimized nonnegative weights constrained to sum to one.
"""

import numpy as np

from .chaos_game_optimizer import ChaosGameOptimizer
from .model_evaluator import ModelEvaluator
from .utils import softmax


class EnsembleModel:
    """
    y_MLRAC = w_GBM * yhat_GBM + w_SVR * yhat_SVR + w_TabNet * yhat_TabNet
    subject to w_GBM + w_SVR + w_TabNet = 1, w_m >= 0   (Eq. 5-6)
    """

    def __init__(self, models: dict, cgo_pop=15, cgo_iter=30, seed=42):
        self.models = models          # name -> fitted model instance
        self.cgo_pop = cgo_pop
        self.cgo_iter = cgo_iter
        self.seed = seed
        self.weights_ = None
        self.convergence_ = None

    def _predict_all(self, X):
        return {name: m.predict(X) for name, m in self.models.items()}

    def optimize_weights(self, X_val, y_val):
        """Finds nonnegative, sum-to-one weights minimizing validation RMSE."""
        preds = self._predict_all(X_val)
        names = list(preds.keys())
        P = np.column_stack([preds[n] for n in names])

        def fitness(theta):
            w = softmax(theta)
            return ModelEvaluator.rmse(y_val, P @ w)

        bounds = [(-5, 5)] * len(names)
        optimizer = ChaosGameOptimizer(fitness, bounds, n_pop=self.cgo_pop,
                                        n_iter=self.cgo_iter, seed=self.seed)
        best_theta, best_fit, hist_best, hist_mean = optimizer.optimize()
        self.weights_ = dict(zip(names, softmax(best_theta)))
        self.convergence_ = {'best': hist_best, 'mean': hist_mean}
        return self

    def predict_from_preds(self, preds_dict):
        return sum(self.weights_[name] * preds_dict[name] for name in self.weights_)

    def predict(self, X):
        if self.weights_ is None:
            raise RuntimeError("Call optimize_weights() before predict().")
        return self.predict_from_preds(self._predict_all(X))