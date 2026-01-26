#pmp_examen
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import arviz as az

df = pd.read_csv("bike_daily.csv")

vars = ["temp_c", "humidity", "wind_kph"]

for v in vars:
    plt.figure()
    plt.scatter(df[v], df["rentals"])
    plt.xlabel(v)
    plt.ylabel("rentals")
    plt.show()

print(df[["rentals","temp_c","humidity","wind_kph"]].corr())

cont = ["temp_c","humidity","wind_kph"]
df_std = df.copy()

for c in cont + ["rentals"]:
    df_std[c] = (df[c] - df[c].mean()) / df[c].std()


X = df_std[["temp_c","humidity","wind_kph"]].values
y = df_std["rentals"].values

with pm.Model() as model:
    beta = pm.Normal("beta", 0, 1, shape=3)
    alpha = pm.Normal("alpha", 0, 1)
    sigma = pm.HalfNormal("sigma", 1)
    mu = alpha + pm.math.dot(X, beta)
    y_obs = pm.Normal("y_obs", mu, sigma, observed=y)
    trace = pm.sample(1500, tune=1000, chains=4, target_accept=0.9)

with model:
    ppc = pm.sample_posterior_predictive(trace)

plt.scatter(df["temp_c"], df["rentals"])
plt.plot(df["temp_c"], ppc["y_obs"].mean(axis=0))
plt.show()

az.waic(trace)

Q = df["rentals"].quantile(0.75)
df["high"] = (df["rentals"] > Q).astype(int)

Xl = df_std[["temp_c","humidity","wind_kph"]].values
yl = df["high"].values
