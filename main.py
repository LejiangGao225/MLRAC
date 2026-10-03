"""
main.py
=========
Entry point demonstrating the full MLRAC workflow for both
compressive-strength and slump prediction, using the `mlrac` package.
"""

import numpy as np

from mlrac import DatasetLoader, MLRACPipeline


def main():
    SEED = 42
    np.random.seed(SEED)

    print("=" * 78)
    print(" MLRAC - Hybrid ML framework for Recycled Aggregate Concrete (RAC)")
    print(" mRMR + Chaos Game Optimizer + GBM/SVR/TabNet ensemble")
    print("=" * 78)

    # ---------------- Compressive strength model -----------------------------
    loader_cs = DatasetLoader(seed=SEED)
    X_cs, y_cs, feat_cs = loader_cs.generate_synthetic(n=210, task='strength')

    pipeline_cs = MLRACPipeline(response_name='Compressive Strength', unit='MPa',
                                 cgo_pop=12, cgo_iter=20, cv_folds=5, seed=SEED)
    pipeline_cs.run(X_cs, y_cs, feat_cs)

    # ---------------- Slump model ---------------------------------------------
    loader_sl = DatasetLoader(seed=SEED + 1)
    X_sl, y_sl, feat_sl = loader_sl.generate_synthetic(n=210, task='slump')

    pipeline_sl = MLRACPipeline(response_name='Slump', unit='mm',
                                 cgo_pop=12, cgo_iter=20, cv_folds=5, seed=SEED)
    pipeline_sl.run(X_sl, y_sl, feat_sl)

    print("\nDone. Replace DatasetLoader.generate_synthetic() with "
          "DatasetLoader.load_csv(path, target_col) to run MLRAC on  "
          "RAC records.")


if __name__ == "__main__":
    main()