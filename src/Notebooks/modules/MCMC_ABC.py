import numpy as np
from modules.ABC_reject import sufficient_statistics
from modules.Gibbs_sampler import run_gibbs


# Define the distance function (L2 norm)
def distance_function(summary_simulated, summary_observed):
    return np.sum((np.array(summary_simulated) - np.array(summary_observed)) ** 2)


# Markov kernel q(theta' | theta) - Propose new values from normal distribution centered around current theta
def markov_kernel(theta_current, step_size=0.1):
    alpha_new = np.random.normal(theta_current[0], step_size)
    beta_new = np.random.normal(theta_current[1], step_size)
    return (alpha_new, beta_new)


# Define the MCMC-ABC algorithm (Algorithm 3)
def mcmc_abc_algorithm(N, epsilon, observed_data, n_spins, step_size=0.1):
    # Initial parameters (theta(0)) from prior distribution (uniform for simplicity)
    theta_current = (np.random.uniform(0, 1), np.random.uniform(0, 1))  # (alpha, beta)

    # Initial simulation based on theta(0)
    z_current = run_gibbs(n_spins, theta_current[0], theta_current[1])

    # Compute summary statistics for observed data
    summary_observed = sufficient_statistics(observed_data)

    # Initialize list to store samples
    alpha_samples = []
    beta_samples = []

    # MCMC Loop
    for t in range(1, N + 1):
        # Step 1: Propose new parameters from the Markov kernel
        theta_proposed = markov_kernel(theta_current, step_size)

        # Step 2: Simulate data based on the proposed theta
        z_proposed = run_gibbs(n_spins, theta_proposed[0], theta_proposed[1])

        # Step 3: Compute summary statistics for the proposed data
        summary_simulated = sufficient_statistics(z_proposed)

        # Step 4: Compute the distance between summary statistics
        dist = distance_function(summary_simulated, summary_observed)

        # Step 5: Compute acceptance probability
        u = np.random.rand()
        likelihood_ratio = (
            1  # Here we assume the prior is uniform and the proposal is symmetric
        )
        acceptance_prob = min(1, (1 / (1 + np.exp(dist))) * likelihood_ratio)

        # Step 6: Accept or reject based on acceptance probability and distance
        if u <= acceptance_prob and dist <= epsilon:
            theta_current = theta_proposed
            z_current = z_proposed

            # Store the accepted parameters
            alpha_samples.append(theta_current[0])
            beta_samples.append(theta_current[1])

    return alpha_samples, beta_samples
