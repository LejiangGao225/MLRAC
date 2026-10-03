"""
pipeline.py
=============
MLRACPipeline
--------------
Implements Section 3 ("Method") and the workflow shown in Fig. 1 of
the manuscript: orchestrates the complete MLRAC procedure from data
splitting through bootstrap uncertainty analysis.
"""

import numpy as np
from sklearn.model_selection import train_test_split

from .mrmr_selector import MRMRSelector
from .gbm_regressor import GBMRegressor
from .svr_regressor import SVRRegressor
from .tabnet_regressor import TabNetRegressorMLRAC
from .ensemble_model import EnsembleModel
from .model_evaluator import ModelEvaluator
from .bootstrap_analyzer import BootstrapAnalyzer


class MLRACPipeline:
    """
    Orchestrates the complete MLRAC workflow:
    split -> mRMR -> CGO-tuned GBM/SVR/TabNet -> CGO ensemble weighting
    -> evaluation -> bootstrap uncertainty analysis.
    """

    def __init__(self, response_name='Response', unit='unit',
                 cgo_pop=12, cgo_iter=20, cv_folds=5, seed=42, verbose=True):
        self.response_name = response_name
        self.unit = unit
        self.cgo_pop = cgo_pop
        self.cgo_iter = cgo_iter
        self.cv_folds = cv_folds
        self.seed = seed
        self.verbose = verbose

        self.selector_ = None
        self.models_ = {}
        self.ensemble_ = None
        self.results_ = {}

    def run(self, X, y, feature_names):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        # ---- 1. 70 / 15 / 15 split -----------------------------------------
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.30, random_state=self.seed)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, random_state=self.seed)

        # ---- 2. mRMR feature selection (fit on training data only) --------
        self.selector_ = MRMRSelector(seed=self.seed)
        self.selector_.fit(X_train, y_train, X_val, y_val, feature_names)
        if self.verbose:
            print(f"\n[{self.response_name}] mRMR-selected features: "
                  f"{self.selector_.selected_names_}")

        Xtr = self.selector_.transform(X_train)
        Xva = self.selector_.transform(X_val)
        Xte = self.selector_.transform(X_test)

        # ---- 3. CGO-tuned constituent learners -----------------------------
        gbm = GBMRegressor(cv_folds=self.cv_folds, cgo_pop=self.cgo_pop,
                            cgo_iter=self.cgo_iter, seed=self.seed)
        svr = SVRRegressor(cv_folds=self.cv_folds, cgo_pop=self.cgo_pop,
                            cgo_iter=self.cgo_iter, seed=self.seed)
        tabnet = TabNetRegressorMLRAC(cv_folds=self.cv_folds,
                                       cgo_pop=max(5, self.cgo_pop // 2),
                                       cgo_iter=max(8, self.cgo_iter // 2),
                                       seed=self.seed)

        gbm.tune(Xtr, y_train)
        svr.tune(Xtr, y_train)
        tabnet.tune(Xtr, y_train)

        if self.verbose:
            print(f"[{self.response_name}] Best CV-RMSE -> "
                  f"GBM={gbm.best_cv_rmse_:.3f}  SVR={svr.best_cv_rmse_:.3f}  "
                  f"TabNet={tabnet.best_cv_rmse_:.3f}")

        gbm.fit(Xtr, y_train)
        svr.fit(Xtr, y_train)
        tabnet.fit(Xtr, y_train, Xva, y_val)
        self.models_ = {'GBM': gbm, 'SVR': svr, 'TabNet': tabnet}

        # ---- 4. CGO ensemble-weight optimization (on validation subset) ---
        self.ensemble_ = EnsembleModel(self.models_, cgo_pop=self.cgo_pop,
                                        cgo_iter=self.cgo_iter, seed=self.seed)
        self.ensemble_.optimize_weights(Xva, y_val)
        if self.verbose:
            w_str = ", ".join(f"{k}={v:.3f}" for k, v in self.ensemble_.weights_.items())
            print(f"[{self.response_name}] Ensemble weights: {w_str}")

        # ---- 5. Predictions & metrics ---------------------------------------
        y_pred_train = self.ensemble_.predict(Xtr)
        y_pred_val = self.ensemble_.predict(Xva)
        y_pred_test = self.ensemble_.predict(Xte)

        m_train = ModelEvaluator.compute_all(y_train, y_pred_train)
        m_val = ModelEvaluator.compute_all(y_val, y_pred_val)
        m_test = ModelEvaluator.compute_all(y_test, y_pred_test)
        obj = ModelEvaluator.obj_function(y_train, y_pred_train, y_test, y_pred_test,
                                           m_train['R2'], m_test['R2'])

        if self.verbose:
            ModelEvaluator.print_table(self.response_name, self.unit,
                                        m_train, m_val, m_test, obj)

        # ---- 6. Bootstrap uncertainty on the locked test subset -------------
        bootstrap = BootstrapAnalyzer(n_boot=1000, seed=self.seed).analyze(
            y_test, y_pred_test)
        if self.verbose:
            BootstrapAnalyzer.print_summary(self.response_name, bootstrap)

        self.results_ = dict(
            selected_features=self.selector_.selected_names_,
            weights=self.ensemble_.weights_,
            metrics_train=m_train, metrics_val=m_val, metrics_test=m_test,
            obj=obj, bootstrap=bootstrap,
            y_test=y_test, y_pred_test=y_pred_test,
        )
        return self.results_