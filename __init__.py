"""
MLRAC package
=============
Hybrid machine-learning framework for predicting Recycled Aggregate
Concrete (RAC) compressive strength and slump, as described in:

"Predictive Modeling of Recycled Aggregate Concrete Mechanical
Performance Through Hybrid Machine Learning Techniques" (Han & Gao)

Each method of the paper is implemented as a distinct class in its
own module. This file re-exports them for convenient imports, e.g.:

    from mlrac import MLRACPipeline, DatasetLoader
"""

from .utils import softmax
from .dataset_loader import DatasetLoader
from .model_evaluator import ModelEvaluator
from .mrmr_selector import MRMRSelector
from .chaos_game_optimizer import ChaosGameOptimizer
from .sklearn_adapter import SklearnRegressorAdapter
from .base_cgo_regressor import BaseCGOTunedRegressor
from .gbm_regressor import GBMRegressor
from .svr_regressor import SVRRegressor
from .tabnet_wrapper import TabNetWrapper
from .tabnet_regressor import TabNetRegressorMLRAC
from .ensemble_model import EnsembleModel
from .bootstrap_analyzer import BootstrapAnalyzer
from .pipeline import MLRACPipeline

__all__ = [
    "softmax",
    "DatasetLoader",
    "ModelEvaluator",
    "MRMRSelector",
    "ChaosGameOptimizer",
    "SklearnRegressorAdapter",
    "BaseCGOTunedRegressor",
    "GBMRegressor",
    "SVRRegressor",
    "TabNetWrapper",
    "TabNetRegressorMLRAC",
    "EnsembleModel",
    "BootstrapAnalyzer",
    "MLRACPipeline",
]