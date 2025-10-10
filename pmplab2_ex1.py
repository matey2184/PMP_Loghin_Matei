#pmplab2_ex1
import random
import numpy as np

initial_urn= ['red']*3 + ['blue']*4 + ['black']*2

num_primes={2,3,5}
n_sim=100000

#store results
draw_results=[]

for _ in range(n_sim):
  urn=initial_urn.copy()
  roll=random.randint(1,6)
  if roll in num_primes:
    urn.append('black')
  elif roll==6:
    urn.append('red')
  else:
    urn.append('blue')
  
  draw=random.choice(urn)
  draw_results.append(draw)

#probability of drawing a red ball
p_red_ball=draw_results.count('red')/n_sim
print(f"Probabilitatea estimata de a lua o bila rosie este: {p_red_ball:.4f}")

#theoretical probability
p_prime=3/6
p_six=1/6
p_other=2/6

p_red_incase_prime=3/(3+4+2+1)
p_red_incase_six=4/(3+4+2+1)
p_red_incase_other=3/(3+4+2+1)

p_red_theoretical=p_prime*p_red_incase_prime + p_six*p_red_incase_six + p_other*p_red_incase_other

print(f"Probabilitatea teoretica de a trage o bila rosie: {p_red_theoretical:.4f}")
print(f"Diferenta:{abs(p_red_ball-p_red_theoretical):.4e}")