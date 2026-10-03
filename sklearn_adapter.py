"""
sklearn_adapter.py
====================
SklearnRegressorAdapter
-------------------------
Adapts any scikit-learn estimator (GBM, SVR pipelines, etc.) to the
uniform `fit(X, y, X_val=None, y_val=None)` / `predict(X)` interface
used throughout MLRAC.
"""


class SklearnRegressorAdapter:
    """Wraps a scikit-learn estimator so it exposes a validation-aware fit()."""

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y, X_val=None, y_val=None):
        # Standard scikit-learn estimators do not use a validation set
        # during fitting; it is accepted here only for interface uniformity.
        self.estimator.fit(X, y)
        return self

    def predict(self, X):
        return self.estimator.predict(X)