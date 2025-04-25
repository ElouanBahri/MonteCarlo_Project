import numpy as np
from modules.ABC_reject import sufficient_statistics, abc_reject,mae_distance
from modules.Gibbs_sampler import run_gibbs




# Markov kernel q(theta' | theta) - Propose new values from normal distribution centered around current theta
def markov_kernel(theta_current, step_size=0.1):
    alpha_new = np.random.normal(theta_current[0], step_size)
    beta_new = np.random.normal(theta_current[1], step_size)
    return (alpha_new, beta_new)




# Define the MCMC-ABC algorithm (Algorithm 3)
def mcmc_abc_algorithm(N, epsilon_1, observed_data, epsilon_algo_2, n_spins, step_size=0.1):

    
    theta_current = abc_reject(
        observed_data,
        prior_alpha=(-1, 1),
        prior_beta=(0, 2),
        n=n_spins,
        epsilon=3,
        num_samples=1000,
    )

   
    print('Initizialisation Done')
    # Compute summary statistics for observed data
    summary_observed = sufficient_statistics(observed_data)

    # Initialize list to store samples
    alpha_samples = []
    beta_samples = []


    for t in range(1, N + 1):
        theta_proposed = markov_kernel(theta_current, step_size)
        z_proposed = run_gibbs(n_spins, theta_proposed[0], theta_proposed[1], steps=1)
        summary_simulated = sufficient_statistics(z_proposed)
        dist = mae_distance(np.array(summary_simulated), np.array(summary_observed))

        if dist <= epsilon_1:
            theta_current = theta_proposed
            alpha_samples.append(theta_current[0])
            beta_samples.append(theta_current[1])

    return alpha_samples, beta_samples

