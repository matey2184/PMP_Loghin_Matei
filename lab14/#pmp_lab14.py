#pmp_lab14
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az

data = pd.read_csv('date_colesterol.csv')
x = data['Ore_Exercitii'].values
y = data['Colesterol'].values
x_scaled = (x - x.mean()) / x.std()
x2_scaled = x_scaled**2

models = {}
traces = {}

for k in [3, 4, 5]:
    with pm.Model() as model:
        w = pm.Dirichlet('w', a=np.ones(k))
        
        alpha = pm.Normal('alpha', mu=200, sigma=50, shape=k)
        beta = pm.Normal('beta', mu=0, sigma=20, shape=k)
        gamma = pm.Normal('gamma', mu=0, sigma=20, shape=k)
        sigma = pm.HalfNormal('sigma', sigma=20, shape=k)
        
        mu = alpha + beta * x_scaled[:, None] + gamma * x2_scaled[:, None]
        
        components = pm.Normal.dist(mu=mu, sigma=sigma)
        likelihood = pm.Mixture('likelihood', w=w, comp_dists=components, observed=y)
        
        trace = pm.sample(2000, tune=1000, target_accept=0.9, return_inferencedata=True)
        pm.compute_log_likelihood(trace)
        
        models[k] = model
        traces[k] = trace

for k in [3, 4, 5]:
    print(f"\n--- Results for K={k} ---")
    summary = az.summary(traces[k], var_names=['w', 'alpha', 'beta', 'gamma', 'sigma'])
    print(summary)

comparison = az.compare(traces, ic="loo")
print("\n--- Model Comparison (LOO) ---")
print(comparison)

comparison_waic = az.compare(traces, ic="waic")
print("\n--- Model Comparison (WAIC) ---")
print(comparison_waic)

best_k = comparison.index[0]
print(f"\nBest model: K={best_k}")