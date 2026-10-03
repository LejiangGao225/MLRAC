"""
utils.py
========
Small shared helper functions used across MLRAC modules.
"""

import numpy as np


def softmax(x):
    """Numerically stable softmax, used to enforce the ensemble-weight
    constraints of Eq. (6): weights are nonnegative and sum to one."""
    e = np.exp(np.asarray(x) - np.max(x))
    return e / e.sum()