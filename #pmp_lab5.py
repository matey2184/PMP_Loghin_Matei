#pmp_lab5
import numpy as np
from hmmlearn import hmm

n_components = 3
n_features = 4
pi = np.array([1/3, 1/3, 1/3])
A = np.array([
    [0.0, 0.5, 0.5],   
    [0.5, 0.25, 0.25],  
    [0.5, 0.25, 0.25]   
])

B = np.array([
    [0.1, 0.2, 0.4, 0.3],     
    [0.15, 0.25, 0.5, 0.1],   
    [0.2, 0.3, 0.4, 0.1]      
])
X_obs = np.array([0, 0, 2, 1, 1, 2, 1, 1, 3, 1, 1]).reshape(-1, 1)

model = hmm.MultinomialHMM(n_components=n_components, n_iter=1)
model.startprob_ = pi
model.transmat_ = A
model.emissionprob_ = B

log_prob_O = model.score(X_obs) 
prob_O = np.exp(log_prob_O)
log_prob_Q_star_O, Q_star_indices = model.decode(X_obs)
prob_Q_star_O = np.exp(log_prob_Q_star_O)
state_map = {0: "Difficult (D)", 1: "Medium (M)", 2: "Easy (E)"}
Q_star_sequence = [state_map[i] for i in Q_star_indices]

print("results using hmmlearn")
print("-" * 35)

print("\nb)")
print(f"log probability P(O) = {log_prob_O:.10f}")
print(f"probability P(O)     = {prob_O:.10e}")

print("\nc)Viterbi")
print(f"observed G=grades indices: {X_obs.flatten().tolist()}")
print(f"most probable state sequence Q*: {Q_star_indices.tolist()}")
print(f"state sequence names:    {Q_star_sequence}")
print(f"\nLog Joint Probability P(Q*, O) = {log_prob_Q_star_O:.10f}")
print(f"probability P(Q*, O)= {prob_Q_star_O:.10e}")