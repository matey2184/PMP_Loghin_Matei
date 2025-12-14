#lab6_1
import numpy as np

PB = 0.01
P_plus_given_B = 0.95
P_minus_given_notB = 0.90
P_notB = 1 - PB
P_plus_given_notB = 1 - P_minus_given_notB

P_plus = (P_plus_given_B * PB) + (P_plus_given_notB * P_notB)
P_B_given_plus = (P_plus_given_B * PB) / P_plus

print(f"P_B_given_plus: {P_B_given_plus}")
min_specificity_S = 1 - ((P_plus_given_B * PB) / P_notB)
print(f"min_specificity_S: {min_specificity_S}")