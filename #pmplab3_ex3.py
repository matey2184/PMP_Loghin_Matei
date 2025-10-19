#pmplab3_ex3
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from scipy.stats import binom

# probabilitatile monedelor 
P_H_P0 = 0.5   
P_H_P1 = 4/7   


NUM_SIMULATIONS = 10000
victories = {'P0': 0, 'P1': 0}

for _ in range(NUM_SIMULATIONS):
    starter = np.random.choice(['P0', 'P1'])
    n = np.random.randint(1, 7) 
    rounds = 2 * n
    second_player = 'P1' if starter == 'P0' else 'P0'
    p_head = P_H_P0 if second_player == 'P0' else P_H_P1

    m = np.random.binomial(rounds, p_head)
    if n >= m:
        victories[starter] += 1
    else:
        victories[second_player] += 1

total_wins = victories['P0'] + victories['P1']
P0_win_chance = victories['P0'] / total_wins
P1_win_chance = victories['P1'] / total_wins

print("--- 1. estimarea sansei de castig prin simulare ---")
print(f"nr de simulari: {NUM_SIMULATIONS}")
print(f"victorii P0: {victories['P0']} ({P0_win_chance:.4f})")
print(f"victorii P1: {victories['P1']} ({P1_win_chance:.4f})")
if P0_win_chance > P1_win_chance:
    print("**P0** are sanse mai mari de castig.")
else:
    print("**P1** are sanse mai mari de castig.")
S_states = ['P0', 'P1']     
R_states = [str(n) for n in range(1, 7)] 
M_states = [str(m) for m in range(0, 13)] 

#  S -> M, R -> M
model = DiscreteBayesianNetwork([
    ('S', 'M'),
    ('R', 'M')
])

cpd_s = TabularCPD(variable='S', variable_card=2,
                   values=[[0.5], [0.5]],
                   state_names={'S': S_states})
cpd_r = TabularCPD(variable='R', variable_card=6,
                   values=[[1/6] * 6],
                   state_names={'R': R_states})

cpd_m_values = np.zeros((13, 2, 6))

for i_S, starter in enumerate(S_states):
    second_player = 'P1' if starter == 'P0' else 'P0'
    p_head = P_H_P0 if second_player == 'P0' else P_H_P1

    for i_R, n_str in enumerate(R_states):
        n = int(n_str)
        num_flips = 2 * n
        for m in range(num_flips + 1):
            prob = binom.pmf(m, num_flips, p_head)
            if m < 13: 
                cpd_m_values[m, i_S, i_R] = prob

cpd_m = TabularCPD(variable='M', variable_card=13,
                   values=cpd_m_values,
                   evidence=['S', 'R'],
                   evidence_card=[2, 6],
                   state_names={'M': M_states, 'S': S_states, 'R': R_states})

model.add_cpds(cpd_s, cpd_r, cpd_m)
print("\n--- 2. reteaua Bayesiana (pgmpy) ---")
print("modelul definit cu variabilele S (Starter), R (Roll), M (Heads Count).")
print("P(M | S, R) foloseste o distributie binomiala bazata pe probabilitatea monedei celui de-al doilea jucator.")

inference = VariableElimination(model)
evidence = {'M': '1'}
result = inference.query(variables=['S'], evidence=evidence, joint=False)

P_S0_given_M1 = result['S'].values[0] # P(S=P0 | M=1)
P_S1_given_M1 = result['S'].values[1] # P(S=P1 | M=1)

print("\n--- 3. probabilitatea jucatorului de Start, stiind că M=1 ---")
print(f"P(Starter=P0 | Heads=1): {P_S0_given_M1:.4f}")
print(f"P(Starter=P1 | Heads=1): {P_S1_given_M1:.4f}")

if P_S0_given_M1 > P_S1_given_M1:
    conclusion = "**P0** este cel mai probabil sa fi inceput jocul."
else:
    conclusion = "**P1** este cel mai probabil sa fi inceput jocul."

print("\nConcluzie:")
print(f"deoarece P(Starter=P1 | Heads=1) > P(Starter=P0 | Heads=1) (adica {P_S1_given_M1:.4f} > {P_S0_given_M1:.4f}), {conclusion}")