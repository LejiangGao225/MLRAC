"""
dataset_loader.py
=================
DatasetLoader
-------------
Implements Section 4 ("Material initialization") of the manuscript:
acquisition of the RAC datasets.

Use `load_csv()` for real experimental records, or
`generate_synthetic()` for a demonstration dataset calibrated
approximately to the descriptive statistics reported in Table 3.
"""

import numpy as np
import pandas as pd


class DatasetLoader:
    """Handles acquisition of the compressive-strength / slump RAC datasets."""

    FEATURE_NAMES = ['Cement', 'FlyAsh', 'GBS', 'Water',
                      'Sand', 'CoarseAggregate', 'Superplasticizer']

    def __init__(self, seed=42):
        self.seed = seed

    def load_csv(self, csv_path, target_col, feature_cols=None):
        """Load a real experimental dataset from a CSV file."""
        df = pd.read_csv(csv_path)
        feature_cols = feature_cols or [c for c in df.columns if c != target_col]
        X = df[feature_cols].values.astype(float)
        y = df[target_col].values.astype(float)
        return X, y, feature_cols

    def generate_synthetic(self, n=210, task='strength'):
        """
        Synthetic RAC mixture generator calibrated approximately to the
        descriptive statistics reported in Table 3 of the manuscript.
        Replace with `load_csv()` for real experimental records.
        """
        rng = np.random.default_rng(self.seed)

        if task == 'strength':
            cement = rng.normal(375.7, 81.0, n).clip(180, 663)
            water = rng.normal(183.2, 23.6, n).clip(135, 304)
            sand = rng.normal(646.5, 84.7, n).clip(435, 952)
            coarse = rng.normal(1166.9, 94.0, n).clip(870, 1427)
            fly_ash = np.where(rng.random(n) < 0.75, 0.0,
                                rng.gamma(2.0, 40, n)).clip(0, 170)
            gbs = np.where(rng.random(n) < 0.97, 0.0,
                            rng.gamma(1.5, 15, n)).clip(0, 81)
            sp = np.where(rng.random(n) < 0.55, 0.0,
                           rng.gamma(2.0, 1.5, n)).clip(0, 8.6)

            noise = rng.normal(0, 3.0, n)
            y = (0.030 * cement - 0.060 * water + 0.020 * fly_ash
                 + 0.015 * gbs + 2.30 * sp - 0.004 * coarse
                 + 0.003 * sand + 27.0 + noise).clip(16.5, 70.5)

        else:  # slump
            cement = rng.normal(370.3, 75.3, n).clip(210, 650)
            water = rng.normal(188.0, 23.2, n).clip(107, 264)
            sand = rng.normal(652.7, 86.4, n).clip(444, 952)
            coarse = rng.normal(1156.6, 89.0, n).clip(826, 1342)
            fly_ash = np.where(rng.random(n) < 0.70, 0.0,
                                rng.gamma(2.0, 45, n)).clip(0, 192)
            gbs = np.where(rng.random(n) < 0.98, 0.0,
                            rng.gamma(1.4, 12, n)).clip(0, 56.5)
            sp = np.where(rng.random(n) < 0.65, 0.0,
                           rng.gamma(2.0, 1.3, n)).clip(0, 10.15)

            noise = rng.normal(0, 14.0, n)
            y = (0.70 * fly_ash + 13.0 * sp - 0.10 * water
                 - 0.06 * cement + 0.03 * sand + 85.0 + noise).clip(0.0, 250.0)

        X = np.column_stack([cement, fly_ash, gbs, water, sand, coarse, sp])
        return X, y, list(self.FEATURE_NAMES)