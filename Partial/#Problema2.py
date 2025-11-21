#Problema2
import numpy as np

states = ['W','R','S']  
observations = ['L','M','H']

state_to_index = {state: i for i, state in enumerate(states)}
obs_to_index = {obs: i for i, obs in enumerate(observations)}
pi = np.array([0.4, 0.3, 0.3])
A = np.array([[0.6, 0.3, 0.1],[0.2, 0.7, 0.1],[0.3, 0.2, 0.5]])
B = np.array([[0.1, 0.7,  0.2],[0.05, 0.25,0.7],[0.8, 0.15, 0.05]])
def forward_algorithm(O, pi, A, B):
    N = len(pi)
    T = len(O)
    alpha = np.zeros((T, N))
    first_obs_index = obs_to_index[O[0]]
    alpha[0, :] = pi*B[:, first_obs_index]
    for t in range(1, T):
        obs_index = obs_to_index[O[t]]
        for j in range(N):
            prev_sum = np.sum(alpha[t-1, :] * A[:, j])
            alpha[t, j] = prev_sum*B[j, obs_index]
    probability = np.sum(alpha[T-1, :])
    return probability, alpha

observation_sequence_b = ['M','H','L']
prob_b, alpha_matrix_b = forward_algorithm(observation_sequence_b, pi, A, B)
print(f"(2.b) P({observation_sequence_b})")
print(f"{prob_b}")
print()

#c)
def viterbi_algorithm(O, pi, A, B):
    N=len(pi)
    T=len(O)
    delta= np.zeros((T, N))
    psi= np.zeros((T, N), dtype=int)  
    first_obs_index= obs_to_index[O[0]]
    delta[0, :]= pi * B[:, first_obs_index]
    for t in range(1, T):
        obs_index= obs_to_index[O[t]]
        for j in range(N):
            max_val = delta[t-1, :] * A[:, j]
            delta[t, j] = np.max(max_val) * B[j, obs_index]
            psi[t, j] = np.argmax(max_val) 

    Q_T_index= np.argmax(delta[T-1, :])
    Q_path = [0] * T
    Q_path[T-1]= Q_T_index
    for t in range(T-2, -1, -1):
        Q_path[t] = psi[t+1,Q_path[t + 1]]
        
    state_path= [states[i] for i in Q_path]
    return state_path, delta

observation_sequence_c = ['M','H','L']
path_c, delta_matrix_c = viterbi_algorithm(observation_sequence_c, pi, A, B)

print(f"most likely path for {observation_sequence_c}")
print(f"{path_c}")
print()

#d)
def generate_sequence(T, pi, A, B):
    N = len(pi)
    O_sequence = []
    Q_sequence = []
    current_state_index= np.random.choice(N, p=pi) 
    for _ in range(T):
        obs_probs= B[current_state_index, :]
        obs_index= np.random.choice(len(observations), p=obs_probs)
        O_sequence.append(observations[obs_index])
        Q_sequence.append(states[current_state_index])
        if _ < T - 1:
            transition_probs = A[current_state_index, :]
            current_state_index = np.random.choice(N, p=transition_probs)
            
    return O_sequence

sequence_to_match = ['M','H','L']
num_simulations =10000
match_count = 0

for _ in range(num_simulations):
    generated_sequence = generate_sequence(len(sequence_to_match), pi, A, B)
    if generated_sequence == sequence_to_match:
        match_count += 1

empirical_probability = match_count / num_simulations

print(f"exact prob (Forward Alg.): {prob_b}")
print(f"empirical prob. (10k simulations): {empirical_probability}")
print(f"difference: {abs(prob_b - empirical_probability)}")