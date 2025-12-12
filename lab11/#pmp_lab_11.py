import pandas as pd
import numpy as np
import pymc as pm
import arviz as az

df = pd.read_csv("Prices.csv")
df['ln_HardDrive'] = np.log(df['HardDrive'])
y = df['Price'].values
x1 = df['Speed'].values
x2 = df['ln_HardDrive'].values
premium_codes = pd.Categorical(df['Premium'], categories=['no', 'yes']).codes
premium_status = premium_codes 

if __name__ == '__main__':
    print("starting bayesian analysis")
    #a)
    print("defining and sampling the bayesian regression model")
    with pm.Model() as linear_regression_model:
        alpha = pm.Normal("alpha", mu=0, sigma=100)
        beta1 = pm.Normal("beta1", mu=0, sigma=100)
        beta2 = pm.Normal("beta2", mu=0, sigma=100)
        sigma = pm.HalfNormal("sigma", sigma=10)

        mu = alpha + beta1 * x1 + beta2 * x2
        y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

        idata = pm.sample(2000, tune=1000, random_seed=42, return_inferencedata=True)

    #b)
    hdi_95 = az.hdi(idata, var_names=['beta1', 'beta2'], hdi_prob=0.95)
    print("\nb) 95% HDI for parameters beta1 and beta2:")
    print(hdi_95)

    #c)
    beta1_hdi = hdi_95['beta1'].to_numpy()
    beta2_hdi = hdi_95['beta2'].to_numpy()
    print("\nc):")
    print(f"beta1 (Speed) 95% HDI: {beta1_hdi};since 0 is not in the interval, Speed is a useful predictor")
    print(f"beta2 (ln_HardDrive) 95% HDI: {beta2_hdi};since 0 is not in the interval, ln(HardDrive) is a useful predictor")

    #d) 
    x1_new = 33  # Speed in MHz
    x2_new = np.log(540) # ln(HardDrive) in MB
    mu_posterior_draws = idata.posterior.stack(samples=["chain", "draw"])

    mu_new_draws = (mu_posterior_draws["alpha"] + mu_posterior_draws["beta1"] * x1_new + mu_posterior_draws["beta2"] * x2_new)
    hdi_mu_90 = az.hdi(mu_new_draws.to_numpy(), hdi_prob=0.90)
    print("\nd)90% HDI for expected sale price (mu) at 33 MHz, 540 MB HD:")
    print(hdi_mu_90)

   
    #e)
    sigma_draws = mu_posterior_draws["sigma"]
    y_new_draws = pm.Normal.dist(mu=mu_new_draws, sigma=sigma_draws).eval()
    hdi_y_90 = az.hdi(y_new_draws.to_numpy().flatten(), hdi_prob=0.90)
    print("\ne)90% HDI prediction interval for sale price (y) at 33 MHz, 540 MB HD:")
    print(hdi_y_90)


    print("\nBonus:")
    with pm.Model() as bonus_model:
        alpha = pm.Normal("alpha", mu=0, sigma=100)
        beta1 = pm.Normal("beta1", mu=0, sigma=100)
        beta2 = pm.Normal("beta2", mu=0, sigma=100)
        premium_effect = pm.Normal("premium_effect", mu=0, sigma=100) 
        sigma = pm.HalfNormal("sigma", sigma=10)

        mu_bonus = alpha + beta1 * x1 + beta2 * x2 + premium_effect * premium_status
        y_obs_bonus = pm.Normal("y_obs_bonus", mu=mu_bonus, sigma=sigma, observed=y)
        idata_bonus = pm.sample(2000, tune=1000, random_seed=42, return_inferencedata=True)

    premium_summary = az.summary(idata_bonus, var_names=['premium_effect'], hdi_prob=0.95)
    premium_hdi = az.hdi(idata_bonus, var_names=['premium_effect'], hdi_prob=0.95)

    print("\nbonus: 95% HDI for premium effect:")
    print(premium_hdi)

   