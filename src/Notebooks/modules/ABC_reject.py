import numpy as np
import random

def ising_model(n, alpha, beta, iterations=10000):
    """Simulates the Ising model using Gibbs sampling."""
    grid = np.random.choice([0, 1], size=(n, n))  # Initialize random grid
    for _ in range(iterations):
        i, j = np.random.randint(0, n, size=2)
        neighbors = grid[(i-1) % n, j] + grid[(i+1) % n, j] + grid[i, (j-1) % n] + grid[i, (j+1) % n]
        prob_1 = np.exp(alpha + beta * neighbors) / (1 + np.exp(alpha + beta * neighbors))
        grid[i, j] = 1 if np.random.rand() < prob_1 else 0
    return grid

def sufficient_statistics(grid):
    """Computes the sufficient statistics S(x) and S'(x)."""
    S_x = np.sum(grid)
    S_x_2= sum(
        (grid[i, j] == grid[(i+1) % len(grid), j]) +
        (grid[i, j] == grid[i, (j+1) % len(grid)])
        for i in range(len(grid)) for j in range(len(grid))
    )
    return S_x, S_x_2

def abc_reject(obs_stats, prior_alpha, prior_beta, n, epsilon, num_samples=1000):
    """ABC-Rejection algorithm to estimate alpha and beta."""
    accepted_params = []
    for _ in range(num_samples):
        alpha, beta = np.random.uniform(*prior_alpha), np.random.uniform(*prior_beta)
        sim_grid = ising_model(n, alpha, beta)
        sim_stats = sufficient_statistics(sim_grid)
        distance = abs(obs_stats[0] - sim_stats[0]) + abs(obs_stats[1] - sim_stats[1])
        if distance < epsilon:
            accepted_params.append((alpha, beta))
    return np.array(accepted_params)


