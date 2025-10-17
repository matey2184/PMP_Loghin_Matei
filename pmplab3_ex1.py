#pmplab3_ex1
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# definim structura
model = DiscreteBayesianNetwork([
    ('S', 'O'),  
    ('S', 'L'),  
    ('S', 'M'),  
    ('L', 'M')   
])

# P(S)
cpd_s = TabularCPD(variable='S', variable_card=2, values=[[0.6], [0.4]], state_names={'S': ['0', '1']})

# P(O | S)
# S=0 (non-spam), S=1 (spam)
# P(O=1|S=0) = 0.1, P(O=0|S=0) = 0.9
# P(O=1|S=1) = 0.7, P(O=0|S=1) = 0.3
cpd_o = TabularCPD(variable='O', variable_card=2,
                   values=[[0.9, 0.3],  
                           [0.1, 0.7]], 
                   evidence=['S'], evidence_card=[2],
                   state_names={'O': ['0', '1'], 'S': ['0', '1']})

# P(L | S)
# P(L=1|S=0) = 0.3, P(L=0|S=0) = 0.7
# P(L=1|S=1) = 0.8, P(L=0|S=1) = 0.2
cpd_l = TabularCPD(variable='L', variable_card=2,
                   values=[[0.7, 0.2],  
                           [0.3, 0.8]], 
                   evidence=['S'], evidence_card=[2],
                   state_names={'L': ['0', '1'], 'S': ['0', '1']})

# P(M | S, L)
# L=0, S=0 | L=1, S=0 | L=0, S=1 | L=1, S=1
# P(M=1 | S=0, L=0) = 0.2, P(M=1 | S=0, L=1) = 0.6
# P(M=1 | S=1, L=0) = 0.5, P(M=1 | S=1, L=1) = 0.9
cpd_m = TabularCPD(variable='M', variable_card=2,
                   values=[[0.8, 0.4, 0.5, 0.1],  
                           [0.2, 0.6, 0.5, 0.9]], 
                   evidence=['S', 'L'], evidence_card=[2, 2],
                   state_names={'M': ['0', '1'], 'S': ['0', '1'], 'L': ['0', '1']})

model.add_cpds(cpd_s, cpd_o, cpd_l, cpd_m)
assert model.check_model()
print("--- a) independente in retea ---")

independencies = model.get_independencies()
print(independencies)
print("\n--- interpretarea independentei ---")
print(" O si L sunt independente conditional S (O ⊥ L | S).")
print(" P(O | S, L) = P(O | S) and P(L | S, O) = P(L | S).")


#b) 
inference = VariableElimination(model)

#definim combinatiile posibile pentru O, L, M
observations = []
for o in [0, 1]:
    for l in [0, 1]:
        for m in [0, 1]:
            observations.append(
                {'O': str(o), 'L': str(l), 'M': str(m)}
            )

classification_results = []

#calculam P(S=1 | O, L, M) pentru toate combinatiile
for evidence in observations:
    #P(S | O, L, M)
    result_s = inference.query(variables=['S'], evidence=evidence, joint=False)
    #P(S=1 | O, L, M)
    prob_spam = result_s['S'].values[1] 
    is_spam = 'Spam (S=1)' if prob_spam >= 0.5 else 'Non-Spam (S=0)'

    classification_results.append({
        'O': evidence['O'],
        'L': evidence['L'],
        'M': evidence['M'],
        'P(S=1 | O, L, M)': prob_spam,
        'Classification': is_spam
    })

print("\n--- b) rezultatele clasificarii (P(S=1 | O, L, M)) ---")
header = f"{'O':<2} | {'L':<2} | {'M':<2} | {'P(S=1 | O, L, M)':<20} | {'Classification':<15}"
print(header)
print("-" * len(header))

for res in classification_results:
    line = (f"{res['O']:<2} | {res['L']:<2} | {res['M']:<2} | "
            f"{res['P(S=1 | O, L, M)']:<20.4f} | {res['Classification']:<15}")
    print(line)
print("\n--- Clasificare ---")
print("reteaua Bayesiana clasifica email-ul ca **Spam (S=1)** daca probabilitatea conditionala calculata P(S=1 | O, L, M) este $\\ge 0.5$.")
print("bazat pe valorile calculate, reteaua clasifica email-ul ca Spam in urmatoarele 5 cazuri:")

#fistram si afisam cazurile de clasificare
spam_cases = [res for res in classification_results if res['Classification'] == 'Spam (S=1)']
spam_summary = [f"(O={c['O']}, L={c['L']}, M={c['M']})" for c in spam_cases]

print(f"* **cazuri clasificate ca Spam (S=1)**: {', '.join(spam_summary)}")
print(f"* **cazuri clasificate ca Non-Spam (S=0)**: {', '.join([f'(O={c['O']}, L={c['L']}, M={c['M']})' for c in classification_results if c['Classification'] == 'Non-Spam (S=0)'])}")
 
