#pmplab4_ex1

from pgmpy.models import MarkovModel
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import Mplp
import numpy as np
import itertools

edges = [('A1', 'A2'), ('A1', 'A3'),
         ('A2', 'A4'), ('A2', 'A5'),
         ('A3', 'A4'), ('A4', 'A5')]

model = MarkovModel(edges)
model.check_model()

print("Cliques:", model.get_cliques())

states = [-1, 1]

state_index = {-1:0, 1:1}

def create_factor(variables, coeffs):
    shape = tuple([2] * len(variables))
    values = np.zeros(shape)
    for assignment in itertools.product(states, repeat=len(variables)):
        value = np.exp(sum(c * a for c, a in zip(coeffs, assignment)))
        idx = tuple(state_index[a] for a in assignment)
        values[idx] = value
    return DiscreteFactor(variables, [2] * len(variables), values)

phi_A1A2 = create_factor(['A1', 'A2'], [1, 2])
phi_A1A3 = create_factor(['A1', 'A3'], [1, 3])
phi_A3A4 = create_factor(['A3', 'A4'], [3, 4])
phi_A2A4A5 = create_factor(['A2', 'A4', 'A5'], [2, 4, 5])

model.add_factors(phi_A1A2, phi_A1A3, phi_A3A4, phi_A2A4A5)

inference = Mplp(model)
map_result = inference.map_query()

print("configuratia cea mai posibila MAP:")
for var, val in map_result.items():
    print(f"{var} = {val}")
