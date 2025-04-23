import matplotlib.pyplot as plt
import numpy as np


def initialize_lattice(L):
    """Initialize an LxL lattice with random spins 0 or 1."""
    return np.random.choice([0, 1], size=(L, L))


def gibbs_step(lattice, alpha, beta):
    """Perform one full Gibbs sampling step on the Ising model."""
    L = lattice.shape[0]
    for i in range(L):
        for j in range(L):
            # The use of %L permit to avoid boundaries problems (we have -1%L = L-1)
            S_i = (
                lattice[(i - 1) % L, j]
                + lattice[(i + 1) % L, j]
                + lattice[i, (j - 1) % L]
                + lattice[i, (j + 1) % L]
            )

            # Compute conditional probability
            p_1 = np.exp(alpha + beta * S_i) / (
                np.exp(beta * (4 - S_i)) + np.exp(alpha + beta * S_i)
            )

            # Sample new spin
            lattice[i, j] = 1 if np.random.rand() < p_1 else 0


def run_gibbs(L, alpha, beta, steps):
    """Run the Gibbs sampler for a given number of steps."""
    lattice = initialize_lattice(L)
    for _ in range(steps):
        gibbs_step(lattice, alpha, beta)
    return lattice


def plot_lattice(lattice, title="Ising Model Configuration"):
    plt.imshow(lattice, cmap="gray", interpolation="nearest")
    plt.title(title)
    plt.show()
