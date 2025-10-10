import numpy as np
import matplotlib.pyplot as plt


lambdas = [1, 2, 5, 10]
n_samples = 1000

data_fixed = {lam: np.random.poisson(lam, n_samples) for lam in lambdas}

#poisson distribution randomized
random_lambdas = np.random.choice(lambdas, n_samples, replace=True)
data_randomized = np.array([np.random.poisson(lam) for lam in random_lambdas])

# histograms 
plt.figure(figsize=(12, 8))

for i, lam in enumerate(lambdas):
    plt.subplot(3, 2, i + 1)
    plt.hist(data_fixed[lam], bins=range(0, max(data_fixed[lam])+2), alpha=0.7, density=True)
    plt.title(f"Poisson(λ={lam})")

#distribution randomized
plt.subplot(3, 2, 5)
plt.hist(data_randomized, bins=range(0, max(data_randomized)+2), color='blue', alpha=0.7, density=True)
plt.title("Randomized Poisson (λ ∈ {1,2,5,10})")

plt.tight_layout()
plt.show()