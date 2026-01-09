#pmp_lab13
import pymc as pm
import arviz as az
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(42)

def generate_data(n_points=33, order=2):
    
    x = np.linspace(-1, 4, n_points)
    if order == 2:
        y = 2 + 1.5*x - 0.8*x**2 + np.random.normal(0, 1.5, n_points)
    elif order == 5:
        y = 1 + 2*x - 0.5*x**2 + 0.1*x**3 - 0.05*x**4 + 0.01*x**5 + np.random.normal(0, 1.5, n_points)
    else:
        y = 2 + x + np.random.normal(0, 2, n_points)
    return x, y

def run_polynomial_model(x, y, order, beta_sd=10):
    x_p = np.array([x**i for i in range(1, order + 1)])
    x_c = x_p - x_p.mean(axis=1, keepdims=True)
    
    with pm.Model() as model:
        alpha = pm.Normal('alpha', mu=y.mean(), sigma=10)
        if isinstance(beta_sd, np.ndarray):
            beta = pm.Normal('beta', mu=0, sigma=beta_sd, shape=order)
        else:
            beta = pm.Normal('beta', mu=0, sigma=beta_sd, shape=order)
            
        epsilon = pm.HalfNormal('epsilon', sigma=5)
        mu = alpha + pm.math.dot(beta, x_c)
        
        y_pred = pm.Normal('y_pred', mu=mu, sigma=epsilon, observed=y)
        
        trace = pm.sample(1000, tune=1000, return_inferencedata=True, progressbar=False)
        return model, trace

print("1.a (Order=5, sd=10)...")
x, y = generate_data(n_points=33, order=5)
model_5, trace_5 = run_polynomial_model(x, y, order=5, beta_sd=10)

plt.figure(figsize=(10, 6))
plt.scatter(x, y, label='Date observate')
az.plot_posterior_predictive_simulated(az.sample_posterior_predictive(trace_5, model=model_5), ax=plt.gca())
plt.title("model polinomial Ordin 5 (beta_sd=10)")
plt.legend()
plt.show()

print("1.b (sd=100 și sd custom)...")
_, trace_5_sd100 = run_polynomial_model(x, y, order=5, beta_sd=100)
sd_custom = np.array([10, 0.1, 0.1, 0.1, 0.1])
_, trace_5_custom = run_polynomial_model(x, y, order=5, beta_sd=sd_custom)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
axes[0].scatter(x, y, alpha=0.5)
az.plot_hdi(x, trace_5_sd100.posterior['y_pred'], ax=axes[0], color='red')
axes[0].set_title("Ordin 5, beta_sd=100 (Prior slab)")

axes[1].scatter(x, y, alpha=0.5)
az.plot_hdi(x, trace_5_custom.posterior['y_pred'], ax=axes[1], color='green')
axes[1].set_title("Ordin 5, beta_sd=[10, 0.1, 0.1, 0.1, 0.1] (Regularizat)")
plt.show()



#ex 2
x_500, y_500 = generate_data(n_points=500, order=5)
_, trace_500 = run_polynomial_model(x_500, y_500, order=5, beta_sd=10)

plt.figure(figsize=(10, 6))
plt.scatter(x_500, y_500, alpha=0.3, label='Date observate (N=500)')
plt.title("Model Ordin 5 cu 500 puncte")
plt.show()


#ex 3
m1, t1 = run_polynomial_model(x, y, order=1)
m2, t2 = run_polynomial_model(x, y, order=2)
m3, t3 = run_polynomial_model(x, y, order=3)

comparison_dict = {
    "Linear (ord 1)": t1,
    "Quadratic (ord 2)": t2,
    "Cubic (ord 3)": t3
}

comp_waic = az.compare(comparison_dict, ic="waic")
comp_loo = az.compare(comparison_dict, ic="loo")

print("\nComparatie WAIC:")
print(comp_waic)
print("\nComparatie LOO:")
print(comp_loo)

az.plot_compare(comp_waic)
plt.title("Comparatie WAIC: Modelele Liniar, Quadratic, Cubic")
plt.show()

#Concluzie Ex 3: Modelul cu cel mai mic scor WAIC/LOO (cel mai din stanga în grafic)
#este considerat cel mai echilibrat intre complexitate si potrivire)