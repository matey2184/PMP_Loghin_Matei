#pmp_lab10
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt


publicity_data = np.array([1.5, 2.0, 2.3, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0])
sales_data = np.array([5.2, 6.8, 7.5, 8.0, 9.0, 10.2, 11.5, 12.0, 13.5, 14.0, 15.0, 15.5, 16.2, 17.0, 18.0, 18.5, 19.5, 20.0, 21.0, 22.0])
publicity_mean = publicity_data.mean()
publicity_std = publicity_data.std()
publicity_normalized = (publicity_data - publicity_mean) / publicity_std

with pm.Model() as linear_model:
    alpha = pm.Normal("alpha", mu=sales_data.mean(), sigma=10)
    beta = pm.Normal("beta", mu=0, sigma=5)
    sigma = pm.HalfNormal("sigma", sigma=1)
    mu = alpha + beta * publicity_normalized
    sales = pm.Normal("sales", mu=mu, sigma=sigma, observed=sales_data)
    idata = pm.sample(2000, tune=1000, random_seed=42, return_inferencedata=True)
print("a) posterior mean estimates for Regression Coefficients")
posterior_summary = az.summary(idata, var_names=["alpha", "beta", "sigma"])
print(posterior_summary[['mean']])
#b) 
print("\nb)94% Highest density interval for regression coefficients ---")
hdi_summary = az.hdi(idata, var_names=["alpha", "beta", "sigma"], hdi_prob=0.94)
print(hdi_summary)
#c) 
print("\nc)predictive intervals for new advertising expenses")
new_publicity_original = np.array([1.5, 6.0, 12.0])
new_publicity_normalized = (new_publicity_original - publicity_mean) / publicity_std
with linear_model:
    mu_pred = alpha + beta * new_publicity_normalized
    sales_pred = pm.Normal("sales_pred", mu=mu_pred, sigma=sigma, shape=len(new_publicity_normalized))
    ppc = pm.sample_posterior_predictive(idata, predictions=sales_pred, extend_inferencedata=False, random_seed=42)
sales_predictions = ppc['sales_pred']

print(f"\nNew Publicity Expenses (Original Scale): {new_publicity_original}")

for i, pub_val in enumerate(new_publicity_original):
    pred_mean = sales_predictions[:, i].mean()
    pred_hdi = az.hdi(sales_predictions[:, i], hdi_prob=0.94)
    lower = pred_hdi.sel(hdi='lower').item()
    upper = pred_hdi.sel(hdi='higher').item()

    print(f"Publicity = ${pub_val:.1f}k:")
    print(f"Predicted Revenue (Mean): ${pred_mean:.2f}k")
    print(f"94% Predictive Interval (HDI): [${lower:.2f}k, ${upper:.2f}k]")

