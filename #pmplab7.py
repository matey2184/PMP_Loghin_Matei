import numpy as np
import pymc as pm
import arviz as az
import pandas as pd

def run_analysis():
    data = np.array([56, 60, 58, 55, 57, 59, 61, 56, 58, 60])
    n = len(data)
    sample_mean = np.mean(data)
    sample_std_dev = np.std(data, ddof=1)

    print("Frequentist Estimates (Part C)")
    print(f"Sample Mean (μ_freq): {sample_mean:.4f} dB")
    print(f"Sample Std. Dev. (σ_freq): {sample_std_dev:.4f} dB\n")

    # a) 
    x = sample_mean  

    print(f"Prior Selection")
    print(f"For μ ~ N(x, 10^2), x is set to the sample mean: {x}\n")

    with pm.Model() as noise_model_a:
        mu_a = pm.Normal("mu_a", mu=x, sigma=10)
        sigma_a = pm.HalfNormal("sigma_a", sigma=10)
        Y_obs = pm.Normal("Y_obs_a", mu=mu_a, sigma=sigma_a, observed=data)

        print("Sampling Weak Prior Model")
        idata_a = pm.sample(draws=3000, tune=1500, chains=1, cores=1, random_seed=42)
        print("Sampling complete.\n")

    # b)
    summary_a = az.summary(idata_a, hdi_prob=0.95)
    mu_a_mean = summary_a.loc["mu_a", "mean"]
    sigma_a_mean = summary_a.loc["sigma_a", "mean"]

    print("Bayesian Estimates (Weak Prior)")
    print("Posterior Summary:")
    print(summary_a[["mean", "hdi_2.5%", "hdi_97.5%"]])
    print(f"\nBayesian Mean Estimate (μ_bayesian_a): {mu_a_mean:.4f} dB")
    print(f"Bayesian Std. Dev. Estimate (σ_bayesian_a): {sigma_a_mean:.4f} dB\n")

    # d) 
    with pm.Model() as noise_model_d:
        mu_d = pm.Normal("mu_d", mu=50, sigma=1)
        sigma_d = pm.HalfNormal("sigma_d", sigma=10)

        Y_obs = pm.Normal("Y_obs_d", mu=mu_d, sigma=sigma_d, observed=data)
        
        print("Sampling Strong Prior Model...")
        idata_d = pm.sample(draws=3000, tune=1500, chains=1, cores=1, random_seed=42)
        print("Sampling complete.\n")

    summary_d = az.summary(idata_d, hdi_prob=0.95)
    mu_d_mean = summary_d.loc["mu_d", "mean"]
    sigma_d_mean = summary_d.loc["sigma_d", "mean"]

    print("Bayesian Estimates (Strong Prior)")
    print("Posterior Summary:")
    print(summary_d[["mean", "hdi_2.5%", "hdi_97.5%"]])
    print(f"\nBayesian Mean Estimate (μ_bayesian_d): {mu_d_mean:.4f} dB")
    print(f"Bayesian Std. Dev. Estimate (σ_bayesian_d): {sigma_d_mean:.4f} dB\n")

    # Comparison and Discussion
    comparison_data = {
        "Estimator": ["Frequentist (Sample)", "Bayesian (Weak Prior)", "Bayesian (Strong Prior)"],
        "Mean (μ) Estimate (dB)": [sample_mean, mu_a_mean, mu_d_mean],
        "Std. Dev. (σ) Estimate (dB)": [sample_std_dev, sigma_a_mean, sigma_d_mean]
    }

    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.set_index("Estimator")
    comparison_df = comparison_df.round(4)

    print("\n[Comparison Table of Point Estimates]")
    print(comparison_df)

    print("\n[Discussion: Differences in Estimates]")
    print("Comparing Frequentist vs. Weak Prior Bayesian:")
    print(f"Mean (μ): The estimates are nearly identical (~{sample_mean:.2f}).") 
          #This is because the weak prior (N(58, 100)) is centered on the data mean 
          # and has high variance (low precision), 
          # allowing the data (likelihood) to dominate the posterior."
    print(f"Std Dev (σ): The Bayesian estimate (~{sigma_a_mean:.2f}) is slightly higher than the frequentist sample standard deviation (~{sample_std_dev:.2f}).")
           #This is typical due to the Half-Normal prior, which acts as a mild regularizer, 
           #resulting in a more stable, slightly larger estimate."

    print("\n2. Comparing Strong Prior Bayesian vs. Others")
    print(f"Mean (μ): The strong prior Bayesian estimate (~{mu_d_mean:.2f}) is significantly lower than the frequentist/weak prior estimate (~{sample_mean:.2f}).")
    #The prior μ ~ N(50, 1^2) is highly informative (very low variance/high precision) 
    # and is mis-centered (50) relative to the data (58). 
    print(f"Std Dev (σ): The estimates for σ remain similar (~{sigma_d_mean:.2f}), as the prior for σ was kept weak in both Bayesian models.")

if __name__ == '__main__':
    run_analysis()