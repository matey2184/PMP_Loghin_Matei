#pmp_lab12
import pymc as pm
import pandas as pd
import numpy as np
import arviz as az
import matplotlib.pyplot as plt

def main():
    try:
        data = pd.read_csv('date_promovare_examen.csv')
    except FileNotFoundError:
        print("Eroare: Fisierul csv nu a fost gasit!")
        return
    counts = data['Promovare'].value_counts()
    print("Distributia claselor:")
    print(counts)
    
    x_studiu = data['Ore_Studiu'].values
    x_somn = data['Ore_Somn'].values
    y_obs = data['Promovare'].values

    with pm.Model() as logistic_model:
        alpha = pm.Normal('alpha', mu=0, sigma=10)
        beta_studiu = pm.Normal('beta_studiu', mu=0, sigma=10)
        beta_somn = pm.Normal('beta_somn', mu=0, sigma=10)
        
        mu = alpha + beta_studiu * x_studiu + beta_somn * x_somn
        theta = pm.math.sigmoid(mu)

        y = pm.Bernoulli('y', p=theta, observed=y_obs)
        
        print("\nIncepe sampling-ul...")
        trace = pm.sample(2000, tune=1000, chains=2, return_inferencedata=True)

#b)

#Granita de decizie reprezinta locul unde probabilitatea de a promova este exact 0.5. 
#Aceasta se intampla cand alpha + beta_1*Studiu + beta_2*Somn=0.
# Ecuația graniței de decizie este:Ore\_Somn = -alpha/beta_2 -beta_1/beta_2 *Ore_Studiu
#Sunt datele bine separate? Da,in acest set de date, cele doua grupuri sunt foarte bine separate. 
#Exista o zona minima de suprapunere, ceea ce inseamna ca modelul de regresie logistica va avea o acuratete foarte ridicata (aproape de 100%)
    stats = az.summary(trace)
    alpha_m = trace.posterior['alpha'].mean().item()
    b_studiu_m = trace.posterior['beta_studiu'].mean().item()
    b_somn_m = trace.posterior['beta_somn'].mean().item()

    print("\nRezultate coeficienti (medie):")
    print(f"Intercept (alpha): {alpha_m:.4f}")
    print(f"Beta Studiu: {b_studiu_m:.4f}")
    print(f"Beta Somn: {b_somn_m:.4f}")
    x_range = np.linspace(x_studiu.min(), x_studiu.max(), 100)
    decision_boundary = -(alpha_m + b_studiu_m * x_range) / b_somn_m

    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(x_studiu[y_obs == 1], x_somn[y_obs == 1], color='blue', label='Promovat', alpha=0.5)
    plt.scatter(x_studiu[y_obs == 0], x_somn[y_obs == 0], color='red', label='Respins', alpha=0.5)
    plt.plot(x_range, decision_boundary, color='black', linestyle='--', label='Granita de decizie')
    
#c)
# #Orele de studiu au, de regula, o influenta mai mare asupra probabilitatii de promovare in acest model,desi ambele variabile sunt semnificative statistic 
#(intervalele lor de credibilitate nu includ valoarea 0)
    plt.xlabel('Ore Studiu / Saptamana')
    plt.ylabel('Ore Somn / Zi')
    plt.title('Regresie Logistica - Promovare Examen')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == '__main__':
    main()



