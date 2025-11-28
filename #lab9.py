#lab9
import pymc as pm
import arviz as az
import numpy as np
import matplotlib.pyplot as plt

az.style.use("arviz-darkgrid")


scenarios = [
    {'Y': 0, 'theta': 0.2, 'label': r'$Y=0, \theta=0.2$'},
    {'Y': 5, 'theta': 0.2, 'label': r'$Y=5, \theta=0.2$'},
    {'Y': 10, 'theta': 0.2, 'label': r'$Y=10, \theta=0.2$'},
    {'Y': 0, 'theta': 0.5, 'label': r'$Y=0, \theta=0.5$'},
    {'Y': 5, 'theta': 0.5, 'label': r'$Y=5, \theta=0.5$'},
    {'Y': 10, 'theta': 0.5, 'label': r'$Y=10, \theta=0.5$'}
]


RANDOM_SEED = 42
N_CHAINS = 4
N_SAMPLES = 4000
N_TUNING = 2000

idata_posteriors = {}
idata_predictives = {}

print("Starting MCMC Sampling for all scenarios...")

for scenario in scenarios:
    Y = scenario['Y']
    theta = scenario['theta']
    label = scenario['label']
    
    
    initial_n = max(10, Y)

    with pm.Model() as model:
        n = pm.Poisson("n", mu=10, initval=initial_n)
        
        # Likelihood for Y: Binomial(n, \theta) 
        # The likelihood is naturally 0 if n < Y, constraining the posterior
        Y_obs = pm.Binomial("Y_obs", n=n, p=theta, observed=Y)

        # Sample the posterior distribution P(n | Y, \theta)
        idata = pm.sample(
            draws=N_SAMPLES,
            tune=N_TUNING,
            chains=N_CHAINS,
            step=pm.Metropolis(),
            random_seed=RANDOM_SEED,
            return_inferencedata=True,
            progressbar=False
        )
        idata_posteriors[label] = idata
        
        
        Y_star = pm.Binomial("Y_star", n=n, p=theta, shape=1)
        
       
        # We need to pass the idata from the sampling step to use its posterior samples
        idata_ppc = pm.sample_posterior_predictive(
            idata, 
            var_names=['Y_star'], 
            random_seed=RANDOM_SEED, 
            predictions=True 
        )
        idata_predictives[label] = idata_ppc

print("Sampling and Posterior Predictive Checks complete.")

# Combine all posterior samples for visualization
combined_idata_n = az.concat(
    list(idata_posteriors.values()), 
    dim="scenario", 
    coords={"scenario": list(idata_posteriors.keys())}
)

fig_a, ax_a = plt.subplots(3, 2, figsize=(12, 10))
ax_a = ax_a.flatten()

# Plot posterior for n for all scenarios
for i, label in enumerate(idata_posteriors.keys()):

    scenario_idata = combined_idata_n.sel(scenario=label)
    
    n_samples = scenario_idata.posterior['n'].values.flatten()
    
    ax_a[i].hist(n_samples, bins=np.arange(np.min(n_samples), np.max(n_samples) + 2) - 0.5, density=True, color='skyblue', edgecolor='black', alpha=0.7)
    
    mean_n = np.mean(n_samples)
    hdi_n = az.hdi(n_samples, hdi_prob=0.94)

    ax_a[i].set_title(f"{label} | Mean $n$: {mean_n:.1f}", fontsize=12)
    ax_a[i].set_xlabel("Number of Customers ($n$)", fontsize=10)
    ax_a[i].set_ylabel("Probability Density", fontsize=10)
    ax_a[i].axvline(10, color='r', linestyle='--', label='Prior Mean (10)')
    ax_a[i].legend()

fig_a.suptitle("Part a) Posterior Distribution of Number of Customers ($n$)", fontsize=16, y=1.02)
fig_a.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig("posterior_n_all_scenarios.png")
plt.close(fig_a)
print("Plot for Part a) saved as posterior_n_all_scenarios.png")

fig_c, ax_c = plt.subplots(3, 2, figsize=(12, 10))
ax_c = ax_c.flatten()

for i, label in enumerate(idata_predictives.keys()):
    y_star_samples = idata_predictives[label].posterior_predictive['Y_star'].values.flatten()
    
    az.plot_dist(
        y_star_samples, 
        ax=ax_c[i], 
        kind='hist', 
        hist_kwargs={'bins': np.arange(np.min(y_star_samples), np.max(y_star_samples) + 2) - 0.5, 'edgecolor': 'white'},
        color='mediumseagreen'
    )
    
    mean_y_star = np.mean(y_star_samples)

    ax_c[i].set_title(f"{label} | Mean $Y^*$: {mean_y_star:.1f}", fontsize=12)
    ax_c[i].set_xlabel("Future Number of Buyers ($Y^*$)", fontsize=10)
    ax_c[i].axvline(mean_y_star, color='k', linestyle=':', label='Mean $Y^*$')
    ax_c[i].legend()

fig_c.suptitle("Part c) Predictive Posterior Distribution for Future Buyers ($Y^*$)", fontsize=16, y=1.02)
fig_c.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig("predictive_posterior_y_star.png")
plt.close(fig_c)
print("Plot for Part c) saved as predictive_posterior_y_star.png")