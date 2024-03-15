import pandas as pd
import numpy as np
import os
from CourbeDesTaux_v0 import SmithWilson
from scipy import optimize
from scipy.linalg import cholesky
import logging
import matplotlib.pyplot as plt
from test.CIR import *

path = os.getcwd()
path_monde_reel = os.path.join(path, 'monde_reel')
path_matrice_transition = os.path.join(
    path_monde_reel,
    "matrice_transition.xlsx")
path_weight_courbe_souveraine = os.path.join(
    path_monde_reel,
    "matrice_poids_courbe_souveraine_par_notation.xlsx"
    )
path_courbe_taux = os.path.join(
    path_monde_reel,
    "courbe_des_taux.xlsx"
    )
# Import
sheet_name = "test1"
data_matrice_transition = pd.read_excel(path_matrice_transition, index_col=0, sheet_name=sheet_name)
array_matrice_transition = np.array(data_matrice_transition)
dim = data_matrice_transition.shape
# parametrage
nbr_annee_projete = 40
year_projected = np.arange(0, int((nbr_annee_projete)/dt), int(1/dt))

param = [0.1037, 5.9279, 0.7490]
list_valeur_initial_by_note = [1.775, 1.775, 1.438, 0.8605, 1.3598, 1.89, 2, 2.1]
dt = 1/12

# Simulation des spread au cours du temps (nbr note X nombre_simu X nbr pas de temps)
list_modelisation_spread = []
for i in range(dim[0]):
    aleas = np.random.normal(size=(1000, nbr_annee_projete*12))
    init_value = list_valeur_initial_by_note[i]
    # il est possible de faire évoluer param
    spread_t = generate_taux_exact(param, dt, aleas, initial_value=init_value) 
    list_modelisation_spread.append(spread_t)
modelisation_spread = np.array(list_modelisation_spread)
modelisation_spread.shape # chaque notation, numero simulation, pas de temps t
modelisation_spread_year = modelisation_spread[:, :, year_projected]
plt.plot(modelisation_spread_year[:, :, 0].mean(axis=1))
plt.show()
modelisation_spread[:, 0, 0]
modelisation_spread[:, 0, 1]

modelisation_spread_year.shape
array_matrice_generatrice_transition = calcul_matrice_generatrice_reel(array_matrice_transition)
array_matrice_generatrice_transition.shape
list_matrice_transition = []
for indice_temporelle in range(nbr_annee_projete):
    model_spread_year = modelisation_spread_year[:, 1, indice_temporelle]
    matrice_transition = calcul_matrice_transition_t(
        array_matrice_generatrice_transition,
        model_spread_year,
        dt)
    list_matrice_transition.append(matrice_transition)
list_matrice_transition = np.array(list_matrice_transition)
list_matrice_transition.shape # nbr_annee X Matrice de trantion (Nbr note X Nbr note)
df_matrice_transition_exp_test = pd.DataFrame(list_matrice_transition[:, 0, :])

plt.plot(modelisation_spread[:, 0, -1].shape)
plt.show()
arrray_maturite = np.arange(1, len(matrice_zcp)+1, 1)
matice_pzc_mat = matrice_zcp[:, 0, 1]
matrice_zcp.shape
taux_rec = 30/100
np.linspace(0, 1, 100)
lres_ = np.array([matrice_zcp[:, 0, 0]*(taux_rec * pr + (1-pr)) for pr in np.linspace(0, 1, 100)])
pd.DataFrame(lres_.T)[[0, 1, 3, 4]].plot()
plt.show()
#faire le calcul Nombre annee de projection fois allant de 0, nbr_annee_projete/dt, 1/dt

res_spread = calcul_spred_theorique(
    matrice_prix = matice_pzc_mat, # courbe des taux
    taux_recouvrement = taux_recouvrement,
    tenor = 0, # anne de projection
    matrice_probabilite_transition = array_matrice_generatrice_transition
    )
plt.plot(res_spread[0:-1, 0:-1])
plt.legend(data_matrice_transition.columns[0:-1])
plt.show()

pd.DataFrame(res_spread, columns=data_matrice_transition.columns)

num_scenario = 0
prix_par_cotation = calcul_price_obligation_t(
    taux_recouvrement,
    price_zc = matrice_zcp[0, 0, num_scenario],
    probabilite_defaut = list_matrice_transition[0, :, -1])
list_matrice_transition.shape
list_matrice_transition.min()
list_matrice_transition.max()
## prix moyenne par cotation:
l_cot = []
taux_recouvrement = 30/100
#Rajoute la variation des simulations
for annee_project in range(8):
    prix = calcul_price_obligation_t(
        taux_recouvrement,
        price_zc = matrice_zcp[:, annee_project, 0][7:],
        probabilite_defaut = list_matrice_transition[:, 0, -1][7:]
        )
    l_cot.append(prix)
l_a_cot = np.array(l_cot)
plt.plot(l_a_cot[:, 0:-1].T)
plt.plot(matrice_zcp[:, annee_project, 0][7:], 'k')
plt.legend(data_matrice_transition.columns[0:-1])
plt.show()
ref = pd.DataFrame(matrice_zcp.shape[:, :, 0].T)
ex = pd.DataFrame(l_a_cot)
ex["ref"] = matrice_zcp[:, annee_project, 0]
diff = ref - ex
l_a_cot.shape
matrice_zcp.shape
list_matrice_transition.shape
ex.plot()
ref.plot()
diff.plot()
plt.show()
df = pd.DataFrame(l_a_cot.T, columns=data_matrice_transition.columns[:-1])
df["ref"] = matrice_zcp[:, annee_project, 0][7:]

list_matrice_transition[0, num_scenario, :].shape
prix_par_cotation
probabilite_defaut = list_matrice_transition[:, num_scenario, 0]
matrice_fluct = taux_recouvrement * probabilite_defaut + (1-probabilite_defaut)
list_matrice_transition.shape

elm_2 = np.dot(array_matrice_generatrice_transition, array_matrice_generatrice_transition)
elm_3 = np.dot(elm_2, array_matrice_generatrice_transition)
elm_4 = np.dot(elm_3, array_matrice_generatrice_transition)
elm_5 = np.dot(elm_4, array_matrice_generatrice_transition)
elm_6 = np.dot(elm_5, array_matrice_generatrice_transition)
res_exp_Q = np.identity(8) + array_matrice_generatrice_transition/1 +\
    elm_2/2 + elm_3/6 + elm_4/24 + elm_5/120 + elm_6/720

pd.DataFrame(res_exp_Q)
