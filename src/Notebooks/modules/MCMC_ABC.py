import numpy as np
from modules.ABC_reject import abc_reject, mae_distance, sufficient_statistics
from modules.Gibbs_sampler import run_gibbs


# Markov kernel q(theta' | theta) - Propose new values from normal distribution centered around current theta
def markov_kernel(theta_current, step_size):
    alpha_current, beta_current = theta_current

    # Propose new parameters by adding normal noise
    alpha_proposed = alpha_current + np.random.normal(0, step_size)
    beta_proposed = beta_current + np.random.normal(0, step_size)

    # Optionally, constrain to your prior support, e.g. [–1, 1] for alpha, [0, 2] for beta
    alpha_proposed = np.clip(alpha_proposed, -1, 1)
    beta_proposed = np.clip(beta_proposed, 0, 2)

    return (alpha_proposed, beta_proposed)


# Define the MCMC-ABC algorithm (Algorithm 3)
def mcmc_abc_algorithm(N, epsilon_1, observed_data, n_spins, step_size=0.1):

    summary_observed = sufficient_statistics(observed_data)

    theta_current = abc_reject(
        summary_observed,
        prior_alpha=(-1, 1),
        prior_beta=(0, 2),
        n=n_spins,
        epsilon=epsilon_1,
        num_samples=1000,
    )

    theta_current = np.mean(theta_current, axis=0)[0], np.mean(theta_current, axis=0)[1]

    print("Initizialisation Done")
    print("theta zero : ", theta_current)

    # Initialize list to store samples
    alpha_samples = []
    beta_samples = []

    for t in range(1, N + 1):
        theta_proposed = markov_kernel(theta_current, step_size)
        z_proposed = run_gibbs(n_spins, theta_proposed[0], theta_proposed[1], steps=1)
        summary_simulated = sufficient_statistics(z_proposed)
        dist = mae_distance(np.array(summary_simulated), np.array(summary_observed))

        print(theta_proposed, dist)

        if dist <= epsilon_1:
            theta_current = theta_proposed
            alpha_samples.append(theta_current[0])
            beta_samples.append(theta_current[1])

    return alpha_samples, beta_samples
