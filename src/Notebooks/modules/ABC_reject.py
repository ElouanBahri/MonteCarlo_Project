import random

import numpy as np
from modules.Gibbs_sampler import run_gibbs


def sufficient_statistics(grid):
    """Computes the sufficient statistics S(x) and S_2(x)."""
    S_x = np.sum(grid)
    S_x_2 = sum(
        (grid[i, j] == grid[(i + 1) % len(grid), j])
        + (grid[i, j] == grid[i, (j + 1) % len(grid)])
        for i in range(len(grid))
        for j in range(len(grid))
    )
    return S_x, S_x_2


def abc_reject(obs_stats, prior_alpha, prior_beta, n, epsilon, num_samples=1000):
    """ABC-Rejection algorithm to estimate alpha and beta."""
    accepted_params = []

    for _ in range(num_samples):
        alpha, beta = np.random.uniform(*prior_alpha), np.random.uniform(*prior_beta)
        sim_grid = run_gibbs(n, alpha, beta, steps=1000)
        sim_stats = sufficient_statistics(sim_grid)
        distance = abs(obs_stats[0] - sim_stats[0]) + abs(obs_stats[1] - sim_stats[1])
        if distance < epsilon:
            accepted_params.append((alpha, beta, distance))
    return np.array(accepted_params)
