"""
chaos_game_optimizer.py
=========================
ChaosGameOptimizer
-------------------
Implements Section 3.2.1 ("Solution Generation"), Eq. (7)-(8) of the
manuscript: a metaheuristic optimizer using logistic-map chaotic
initialization and a chaos-driven exploration/exploitation update rule.
"""

import numpy as np


class ChaosGameOptimizer:
    """
    Metaheuristic optimizer used to tune model hyperparameters (Eq. 4)
    and to determine ensemble weights (Eq. 5-6) throughout MLRAC.
    """

    def __init__(self, fitness_func, bounds, n_pop=20, n_iter=40, seed=None):
        self.fitness_func = fitness_func
        self.bounds = np.asarray(bounds, dtype=float)   # shape (dim, 2)
        self.dim = len(bounds)
        self.n_pop = n_pop
        self.n_iter = n_iter
        self.rng = np.random.default_rng(seed)

    def _logistic_sequence(self, n):
        """Logistic-map chaotic sequence, Eq. (7), r = 4 (fully chaotic regime)."""
        x = float(self.rng.random())
        if x in (0.0, 0.25, 0.5, 0.75, 1.0):
            x += 1e-6
        seq = np.empty(n)
        for i in range(n):
            x = 4.0 * x * (1.0 - x)
            seq[i] = x
        return seq

    def _init_population(self):
        """Chaotic initialization spread across the permitted search space."""
        pop = np.zeros((self.n_pop, self.dim))
        for d in range(self.dim):
            chaos_seq = self._logistic_sequence(self.n_pop)
            lb, ub = self.bounds[d]
            pop[:, d] = lb + chaos_seq * (ub - lb)
        return pop

    def optimize(self):
        """Runs the CGO search and returns the best solution found."""
        pop = self._init_population()
        fitness = np.array([self.fitness_func(ind) for ind in pop])
        best_idx = int(np.argmin(fitness))
        GB, GB_fit = pop[best_idx].copy(), fitness[best_idx]

        hist_best, hist_mean = [], []

        for t in range(self.n_iter):
            alpha = 1.0 - t / max(1, self.n_iter)   # decreasing step size
            lb, ub = self.bounds[:, 0], self.bounds[:, 1]
            for i in range(self.n_pop):
                chaos_val = self._logistic_sequence(1)[0]
                pull = alpha * chaos_val * (GB - pop[i])                 # Eq. (8)
                jitter = (1 - alpha) * (self.rng.random(self.dim) - 0.5) \
                    * (ub - lb) * 0.02
                candidate = np.clip(pop[i] + pull + jitter, lb, ub)
                cand_fit = self.fitness_func(candidate)
                if cand_fit < fitness[i]:
                    pop[i], fitness[i] = candidate, cand_fit

            best_idx = int(np.argmin(fitness))
            if fitness[best_idx] < GB_fit:
                GB_fit, GB = fitness[best_idx], pop[best_idx].copy()

            hist_best.append(GB_fit)
            hist_mean.append(float(fitness.mean()))

        return GB, GB_fit, hist_best, hist_mean