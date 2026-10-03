"""
tabnet_wrapper.py
===================
TabNetWrapper
--------------
Implements Section 3.5 ("TabNet"), Eq. (15)-(25) of the manuscript:
a deep tabular-learning architecture with sequential attention and
intrinsic interpretability.

Wraps the real TabNet (pytorch-tabnet) if installed; otherwise falls
back to an MLP, which approximates TabNet's nonlinear representation
role (without the attention-mask interpretability) so the full
pipeline still runs end-to-end.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

try:
    from pytorch_tabnet.tab_model import TabNetRegressor as _RealTabNet
    TABNET_AVAILABLE = True
except Exception:
    TABNET_AVAILABLE = False


class TabNetWrapper:
    """
    Thin wrapper exposing a uniform fit/predict interface over either
    the genuine TabNet architecture or an MLP fallback.
    """

    def __init__(self, n_steps=3, n_d=16, lr=0.005, lambda_sparse=1e-4,
                 seed=42, max_epochs=80, patience=15):
        self.n_steps = int(round(n_steps))
        self.n_d = int(round(n_d))
        self.lr = float(lr)
        self.lambda_sparse = float(lambda_sparse)
        self.seed = seed
        self.max_epochs = max_epochs
        self.patience = patience
        self.scaler = StandardScaler()
        self.model = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        Xtr = self.scaler.fit_transform(X_train)
        y_train = np.asarray(y_train, dtype=float)

        if TABNET_AVAILABLE:
            eval_set = None
            if X_val is not None:
                Xv = self.scaler.transform(X_val)
                eval_set = [(Xv.astype(np.float32),
                             np.asarray(y_val, float).reshape(-1, 1).astype(np.float32))]
            self.model = _RealTabNet(
                n_d=self.n_d, n_a=self.n_d, n_steps=self.n_steps,
                lambda_sparse=self.lambda_sparse,
                optimizer_params=dict(lr=self.lr), seed=self.seed, verbose=0)
            self.model.fit(
                Xtr.astype(np.float32), y_train.reshape(-1, 1).astype(np.float32),
                eval_set=eval_set, max_epochs=self.max_epochs, patience=self.patience,
                batch_size=min(64, len(Xtr)), virtual_batch_size=min(32, len(Xtr)))
        else:
            hidden = max(8, self.n_d)
            self.model = MLPRegressor(
                hidden_layer_sizes=(hidden, hidden),
                learning_rate_init=self.lr, alpha=self.lambda_sparse,
                max_iter=self.max_epochs * 5, random_state=self.seed,
                early_stopping=True, n_iter_no_change=self.patience)
            self.model.fit(Xtr, y_train)
        return self

    def predict(self, X):
        Xs = self.scaler.transform(X)
        if TABNET_AVAILABLE:
            pred = self.model.predict(Xs.astype(np.float32))
        else:
            pred = self.model.predict(Xs)
        return np.asarray(pred).ravel()