import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.stats as stats

path_input = os.path.join(
    os.getcwd(), "monde_reel", "CDS_France_2024.xlsx"
    )
data = pd.read_excel(path_input)
data["Dernier"] = data["Dernier"] /1e4
data = data.sort_values(by=["Date"], axis=0, ignore_index=True)
data_sorted = np.sort(data["Date"])
rates = data["Dernier"]
sum_mnt = rates.sum()
list_rate = rates.sort_values(ignore_index=True)
fct_rep = [list_rate[:enu].sum() for enu, val in enumerate(list_rate)]
plt.plot(list_rate, fct_rep / sum_mnt)
plt.show()
df = data.copy()
df = data.set_index("Date")
plt.plot(df["Dernier"])
plt.show()

hist, bin_edges = np.histogram(rates, bins=100)
plt.bar(bin_edges[:-1], hist, width=bin_edges[1]-bin_edges[0], color='red', alpha=0.5)
plt.show()

shape, loc, scale = stats.lognorm.fit(rates)
plt.hist(rates, bins=30, density=True, alpha=0.6, color='g', label='Données')
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
log_normal_pdf = stats.lognorm.pdf(x, shape, loc=loc, scale=scale)
plt.plot(x, log_normal_pdf, 'k', linewidth=2, label='Loi log-normale ajustée')
plt.xlabel('Valeurs')
plt.ylabel('Densité de probabilité')
plt.title('Ajustement de la loi log-normale aux données')
plt.legend()
plt.show()

donnees = rates
ks_statistic, ks_p_value = stats.kstest(donnees, 'lognorm', args=(shape, loc, scale))
log_likelihood = np.sum(stats.lognorm.logpdf(donnees, shape, loc, scale))
predicted_probs = stats.lognorm.pdf(donnees, shape, loc, scale)
r_squared = 1 - np.sum((donnees - predicted_probs)**2) / np.sum((donnees - np.mean(donnees))**2)



# Calcul de la fonction de répartition empirique (ECDF)
def ecdf(data):
    x = np.sort(data)
    y = np.arange(1, len(data) + 1) / len(data)
    return x, y

# Calcul de l'ECDF pour les données
x_ecdf, y_ecdf = ecdf(donnees)

# Générer des données à titre d'exemple
donnees_lognorm = np.random.lognormal(mean=0, sigma=1, size=1000)

# Calcul de la fonction de répartition cumulative (CDF) de la loi log-normale
def lognorm_cdf(x, mean, sigma):
    return stats.lognorm.cdf(x, s=sigma, scale=np.exp(mean))

# Générer des valeurs pour l'axe x
x_values = np.linspace(min(donnees), max(donnees), 100)

# Calcul de la CDF pour la loi log-normale
y_lognorm_cdf = stats.lognorm.cdf(x, shape, loc=loc, scale=scale)

# Tracer la CDF de la loi log-normale
plt.plot(x_values, y_lognorm_cdf, label='Log-Normale CDF')
plt.step(x_ecdf, y_ecdf, label='ECDF')
plt.xlabel('Valeurs')
plt.ylabel('Probabilité cumulée')
plt.title('Fonction de Répartition Cumulative (CDF) de la loi log-normale')
plt.legend()
plt.show()

#Frechet

# Ajuster une loi de Fréchet aux données
shape_frechet, loc_frechet, scale_frechet = stats.invweibull.fit(donnees)
plt.hist(donnees, bins=30, density=True, alpha=0.6, color='g', label='Données')
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
frechet_pdf = stats.invweibull.pdf(x, shape_frechet, loc=loc_frechet, scale=scale_frechet)
plt.plot(x, frechet_pdf, 'k', linewidth=1, label='Loi de Fréchet ajustée', color="blue")
plt.plot(x, log_normal_pdf, 'k', linewidth=1, label='Loi log-normale ajustée')
plt.xlabel('Valeurs')
plt.ylabel('Densité de probabilité / Probabilité cumulée')
plt.title('Ajustement de la loi de Fréchet aux données')
plt.legend()
plt.show()

# Calcul de la CDF pour la loi log-normale
y_frechet_cdf = stats.invweibull.cdf(x, shape_frechet, loc=loc_frechet, scale=scale_frechet)

# Tracer la CDF de la loi log-normale
plt.plot(x_values, y_lognorm_cdf, label='Log-Normale CDF')
plt.plot(x_values, y_frechet_cdf, label='Frechet CDF')
plt.step(x_ecdf, y_ecdf, label='ECDF')
plt.xlabel('Valeurs')
plt.ylabel('Probabilité cumulée')
plt.title('Fonction de Répartition Cumulative (CDF) de la loi log-normale')
plt.legend()
plt.show()


ks_statistic_lognormal, ks_p_value_lognormal = stats.kstest(donnees, 'lognorm', args=(shape, loc, scale))
log_likelihood = np.sum(stats.lognorm.logpdf(donnees, shape, loc, scale))
predicted_probs_lognormal = stats.lognorm.pdf(donnees, shape, loc, scale)
r_squared_lognormal = 1 - np.sum((donnees - predicted_probs_lognormal)**2) / np.sum((donnees - np.mean(donnees))**2)

ks_statistic_frechet, ks_p_value_frechet = stats.kstest(
    donnees,
    'invweibull',
    args=(shape_frechet, loc_frechet, scale_frechet))
frechet_likelihood = np.sum(stats.invweibull.logpdf(donnees, shape_frechet, loc_frechet, scale_frechet))
predicted_probs_frechet = stats.invweibull.pdf(donnees, shape_frechet, loc_frechet, scale_frechet)
r_squared_frechet = 1 - np.sum((donnees - predicted_probs_frechet)**2) / np.sum((donnees - np.mean(donnees))**2)

