"""
bootstrap_analyzer.py
========================
BootstrapAnalyzer
-------------------
Implements Section 5.4 ("Uncertainty Analysis") of the manuscript:
1,000x bootstrap resampling to quantify the stability of RMSE, MAE,
and R2 for a fixed, already-trained model.
"""

import numpy as np

from .model_evaluator import ModelEvaluator


class BootstrapAnalyzer:
    """1,000x bootstrap resampling to quantify uncertainty of RMSE, MAE, R2."""

    def __init__(self, n_boot=1000, seed=42):
        self.n_boot = n_boot
        self.seed = seed

    def analyze(self, y_true, y_pred):
        rng = np.random.default_rng(self.seed)
        y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
        n = len(y_true)
        store = {'RMSE': [], 'MAE': [], 'R2': []}

        for _ in range(self.n_boot):
            idx = rng.integers(0, n, n)
            yt, yp = y_true[idx], y_pred[idx]
            store['RMSE'].append(ModelEvaluator.rmse(yt, yp))
            store['MAE'].append(ModelEvaluator.mae(yt, yp))
            store['R2'].append(ModelEvaluator.r2(yt, yp))

        summary = {}
        for k, v in store.items():
            v = np.asarray(v)
            summary[k] = dict(mean=float(v.mean()), sd=float(v.std()),
                               ci_lower=float(np.percentile(v, 2.5)),
                               ci_upper=float(np.percentile(v, 97.5)))
        return summary

    @staticmethod
    def print_summary(name, summary):
        print(f"\n[{name}] Bootstrap (1000 resamples):")
        for k, v in summary.items():
            print(f"  {k}: mean={v['mean']:.3f}  sd={v['sd']:.3f}  "
                  f"95% CI=[{v['ci_lower']:.3f}, {v['ci_upper']:.3f}]")


