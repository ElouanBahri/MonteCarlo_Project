import numpy as np

import matplotlib.pyplot as plt

def initialize_lattice(L):
    """Initialize an LxL lattice with random spins 0 or 1."""
    return np.random.choice([0, 1], size=(L, L))

def gibbs_step(lattice, alpha, beta):
    """Perform one full Gibbs sampling step on the Ising model."""
    L = lattice.shape[0]
    for i in range(L):
        for j in range(L):
            # Sum of neighboring spins
            S_i = (lattice[(i-1) % L, j] + lattice[(i+1) % L, j] +
                   lattice[i, (j-1) % L] + lattice[i, (j+1) % L])
            
            # Compute conditional probability
            p_1 = np.exp(alpha + beta * S_i) / (1 + np.exp(alpha + beta * S_i))
            
            # Sample new spin
            lattice[i, j] = 1 if np.random.rand() < p_1 else 0

def run_gibbs(L, alpha, beta, steps):
    """Run the Gibbs sampler for a given number of steps."""
    lattice = initialize_lattice(L)
    for _ in range(steps):
        gibbs_step(lattice, alpha, beta)
    return lattice

def plot_lattice(lattice, title="Ising Model Configuration"):
    plt.imshow(lattice, cmap='gray', interpolation='nearest')
    plt.title(title)
    plt.show()

# Parameters
L = 20  # Lattice size
alpha = 0  # No external field
betas = [0.1, 0.5, 1.5, 2.5]  # Different beta values

# Run and visualize for different beta values
for beta in betas:
    final_lattice = run_gibbs(L, alpha, beta, steps=1000)
    plot_lattice(final_lattice, title=f"Beta = {beta}")
