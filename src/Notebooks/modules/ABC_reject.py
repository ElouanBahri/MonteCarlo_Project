import random

import numpy as np
from modules.Gibbs_sampler import run_gibbs


# MSE
def mse_distance(s1, s2):
    return np.mean((s1 - s2) ** 2)


# MAE
def mae_distance(s1, s2):
    return np.mean(np.abs(s1 - s2))


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
        sim_grid = run_gibbs(n, alpha, beta, steps=1)
        sim_stats = sufficient_statistics(sim_grid)
        distance = abs(obs_stats[0] - sim_stats[0]) + abs(obs_stats[1] - sim_stats[1])
        print(alpha, beta, distance)
        if distance < epsilon:
            accepted_params.append((alpha, beta, distance))
    return np.array(accepted_params)


def abc_reject_while(obs_stats, prior_alpha, prior_beta, n, epsilon):
    """ABC-Rejection algorithm to estimate alpha and beta."""

    while True:
        alpha, beta = np.random.uniform(*prior_alpha), np.random.uniform(*prior_beta)
        sim_grid = run_gibbs(n, alpha, beta, steps=1000)
        sim_stats = sufficient_statistics(sim_grid)
        distance1 = mae_distance(np.array(obs_stats), np.array(sim_stats))
        distance2 = mse_distance(np.array(obs_stats), np.array(sim_stats))

        print(distance1, distance2, alpha, beta)
        if distance2 < epsilon:
            return (alpha, beta, distance2)
