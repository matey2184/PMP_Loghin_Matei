#pmplab4_ex2
import numpy as np
from pgmpy.models import MarkovModel
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import BeliefPropagation
import matplotlib.pyplot as plt

# parametri imagine
H, W = 5, 5  
lambda_reg = 2.0  
states = [0, 1]  

#cream imaginea originala
np.random.seed(42)
original_img = np.random.choice(states, size=(H, W))

#adaugam noise: modificam 10% din pixeli
noisy_img = original_img.copy()
num_noisy = max(1, int(0.10 * H * W))
flip_indices = np.random.choice(H * W, num_noisy, replace=False)

for idx in flip_indices:
    r, c = divmod(idx, W)
    noisy_img[r, c] = 1 - noisy_img[r, c]  

print("Original image:\n", original_img)
print("Noisy image:\n", noisy_img)

variables = [f"X_{i}_{j}" for i in range(H) for j in range(W)]

edges = []
for i in range(H):
    for j in range(W):
        if i + 1 < H:
            edges.append((f"X_{i}_{j}", f"X_{i+1}_{j}"))
        if j + 1 < W:
            edges.append((f"X_{i}_{j}", f"X_{i}_{j+1}"))

#construim reteaua Markov
model = MarkovModel(edges)

def unary_factor(var_name, observed_val):
    values = np.zeros(2)
    for s in states:
        values[s] = np.exp(-(lambda_reg * (s - observed_val) ** 2))
    return DiscreteFactor([var_name], [2], values)

def pairwise_factor(var1, var2):
    values = np.zeros((2, 2))
    for x in states:
        for y in states:
            values[x, y] = np.exp(-((x - y) ** 2))
    return DiscreteFactor([var1, var2], [2, 2], values)

factors = []
for i in range(H):
    for j in range(W):
        var = f"X_{i}_{j}"
        factors.append(unary_factor(var, noisy_img[i, j]))

for (u, v) in edges:
    factors.append(pairwise_factor(u, v))

model.add_factors(*factors)
model.check_model()

bp = BeliefPropagation(model)
map_estimate = bp.map_query(variables)

#convertim in image format
denoised_img = np.zeros((H, W), dtype=int)
for i in range(H):
    for j in range(W):
        denoised_img[i, j] = map_estimate[f"X_{i}_{j}"]

print("Denoised image:\n", denoised_img)

fig, ax = plt.subplots(1, 3, figsize=(8, 3))
ax[0].imshow(original_img, cmap="gray", vmin=0, vmax=1)
ax[0].set_title("Original")
ax[1].imshow(noisy_img, cmap="gray", vmin=0, vmax=1)
ax[1].set_title("Noisy")
ax[2].imshow(denoised_img, cmap="gray", vmin=0, vmax=1)
ax[2].set_title("Denoised (MAP)")
for a in ax:
    a.set_xticks([])
    a.set_yticks([])
plt.show()
