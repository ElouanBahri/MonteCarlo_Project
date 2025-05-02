
# 🧊 Monte Carlo Simulation: Parameter Estimation in the Ising Model using ABC

This project explores **Monte Carlo methods** for parameter estimation in the **Ising model**, a fundamental model in statistical physics used to represent spin systems on a 2D grid. The goal is to estimate the interaction parameters:  
- \( \alpha \): external field  
- \( \beta \): coupling strength  

The estimation is based on simulated data, using **likelihood-free Bayesian inference** techniques.

---

## 🔧 Methods

### ✅ Gibbs Sampler
Used to simulate samples from the Ising distribution by iteratively updating each spin based on its neighbors. This forms the backbone for generating both observed and simulated datasets.

### ✅ ABC-Reject (Approximate Bayesian Computation)
A simple rejection-based ABC method:
- Draw parameters from the prior.
- Simulate a grid using the Gibbs sampler.
- Accept the parameter if the distance between simulated and observed summary statistics is below a threshold \( \epsilon \).

### ✅ MCMC-ABC (Markov Chain Monte Carlo ABC)
A more efficient, chain-based version of ABC:
- Propose new parameters using a symmetric Markov kernel.
- Accept or reject based on ABC distance criterion.
- Enables better exploration of parameter space and faster convergence.

---

## 📁 Project Structure

This project uses **Python == 3.11**.

## 1. Installation

### 1.1. Virtual environment
```bash
conda env create -f src/environment/conda_dependencies.yml
conda activate MC_project_env
```

### 1.2. Dev guidelines

1. To update your environment, make sure to run :
```bash
pip install -r src/environment/requirements.txt
```

2. To format your code, you can run :
```bash
invoke format
```

