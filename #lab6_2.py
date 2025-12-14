#lab6_2
import numpy as np
from scipy.stats import gamma

k = 180
N = 10

alpha_0 = 1
beta_0 = 0
alpha_post = alpha_0 + k
beta_post = beta_0 + N
print(f"alpha_post: {alpha_post}")
print(f"beta_post: {beta_post}")

lambda_mode = (alpha_post - 1) / beta_post
print(f"lambda_mode: {lambda_mode}")
confidence_level = 0.94
lower_percentile = (1 - confidence_level) / 2
upper_percentile = 1 - lower_percentile
lambda_HDI_lower = gamma.ppf(lower_percentile, a=alpha_post, scale=1/beta_post)
lambda_HDI_upper = gamma.ppf(upper_percentile, a=alpha_post, scale=1/beta_post)
print(f"lambda_HDI_lower: {lambda_HDI_lower}")
print(f"lambda_HDI_upper: {lambda_HDI_upper}")