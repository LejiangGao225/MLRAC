"""
model_evaluator.py
===================
ModelEvaluator
--------------
Implements Section 5 ("Evaluations") of the manuscript:
R2, RMSE, MAE, VAF, OBJ, A20, Theil's U, and the scatter index (SI),
Eq. (27)-(34).
"""

import numpy as np


class ModelEvaluator:
    """Static collection of all performance metrics used in the manuscript."""

    @staticmethod
    def rmse(y_true, y_pred):
        """Eq. (28): root mean square error."""
        y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    @staticmethod
    def mae(y_true, y_pred):
        """Eq. (29): mean absolute error."""
        y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
        return float(np.mean(np.abs(y_true - y_pred)))

    @staticmethod
    def r2(y_true, y_pred):
        """Eq. (27): squared Pearson correlation coefficient."""
        y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
        if np.std(y_true) == 0 or np.std(y_pred) == 0:
            return 0.0
        r = np.corrcoef(y_true, y_pred)[0, 1]
        return float(r ** 2)

    @staticmethod
    def vaf(y_true, y_pred):
        """Eq. (30): Variance Accounted For (%)."""
        y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
        denom = np.var(y_true)
        if denom == 0:
            return 0.0
        return float((1 - np.var(y_true - y_pred) / denom) * 100)

    @staticmethod
    def a20(y_true, y_pred):
        """Eq. (32): percentage of predictions within 20% relative error."""
        y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
        mask = y_true != 0
        rel_err = np.abs((y_pred[mask] - y_true[mask]) / y_true[mask])
        return float(100 * np.mean(rel_err <= 0.20))

    @staticmethod
    def theil_u(y_true, y_pred):
        """Eq. (33): Theil's inequality coefficient."""
        y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
        num = np.sqrt(np.mean((y_true - y_pred) ** 2))
        den = np.sqrt(np.mean(y_true ** 2)) + np.sqrt(np.mean(y_pred ** 2))
        return float(num / den) if den != 0 else 0.0

    @staticmethod
    def scatter_index(y_true, y_pred):
        """Eq. (34): scatter index (RMSE normalized by observed mean)."""
        y_true = np.asarray(y_true, float)
        ybar = np.mean(y_true)
        return float(ModelEvaluator.rmse(y_true, y_pred) / ybar) if ybar != 0 else 0.0

    @staticmethod
    def obj_function(y_train, pred_train, y_test, pred_test, r2_train, r2_test):
        """Eq. (31): combined training/testing objective function (lower=better)."""
        n_train, n_test = len(y_train), len(y_test)
        mae_test = ModelEvaluator.mae(y_test, pred_test)
        rmse_train = ModelEvaluator.rmse(y_train, pred_train)
        rmse_test = ModelEvaluator.rmse(y_test, pred_test)

        term1 = ((n_train - n_test) / (n_train + n_test)) * \
            (mae_test + rmse_train) / (1 + r2_train)
        term2 = ((2 * n_train) / (n_test + n_train)) * \
            (rmse_test - mae_test) / (1 + r2_test)
        return float(term1 + term2)

    @classmethod
    def compute_all(cls, y_true, y_pred):
        """Convenience method returning all split-specific metrics at once."""
        return dict(
            R2=cls.r2(y_true, y_pred),
            RMSE=cls.rmse(y_true, y_pred),
            MAE=cls.mae(y_true, y_pred),
            VAF=cls.vaf(y_true, y_pred),
            A20=cls.a20(y_true, y_pred),
            U=cls.theil_u(y_true, y_pred),
            SI=cls.scatter_index(y_true, y_pred),
        )

    @staticmethod
    def print_table(name, unit, m_train, m_val, m_test, obj):
        """Prints a formatted train/validation/test performance table."""
        print(f"\n=== {name} performance summary ===")
        header = (f"{'Split':<12}{'R2':>8}{'RMSE':>10}{'MAE':>10}"
                  f"{'VAF(%)':>10}{'A20(%)':>8}{'U':>8}{'SI':>8}")
        print(header)
        for split_name, m in [('Train', m_train), ('Validation', m_val), ('Test', m_test)]:
            print(f"{split_name:<12}{m['R2']:>8.3f}{m['RMSE']:>10.3f}{m['MAE']:>10.3f}"
                  f"{m['VAF']:>10.2f}{m['A20']:>8.1f}{m['U']:>8.3f}{m['SI']:>8.3f}")
        print(f"OBJ index (lower=better): {obj:.4f}")
        print(f"(RMSE, MAE reported in {unit})")