import pandas as pd
import numpy as np
import os
from monde_reel.ModelTaux_G2x import *
import matplotlib.pyplot as plt
from scipy.linalg import cholesky
from typing import Callable, List

def correlation_brownien(
        year_of_projection,
        number_of_simulation,
        correlation_matrix,
        step_of_time,
        number_of_browniens
        ) -> np.ndarray:
    W_bro = np.random.randn( 
        int(year_of_projection * (1 / step_of_time)) + 1,
        int(number_of_browniens),
        int(number_of_simulation)
        )
    for j in range(int(year_of_projection * (1 / step_of_time))+1):
        W_bro[j, :, :] = np.dot(
            cholesky(correlation_matrix, lower=True), W_bro[j, :, :]
        )
    return W_bro

def get_brownien_correlated(
        W_bro: List | np.ndarray
        ) -> List:
    list_of_brownien_correlated = []
    dimension_correlation_brownien = np.shape(W_bro)
    for indice_brownien in range(dimension_correlation_brownien[1]):
        sub_brownien = W_bro[:, indice_brownien, :]
        brownien_correlated = np.column_stack((sub_brownien, -sub_brownien))
        list_of_brownien_correlated.append(brownien_correlated)
    return list_of_brownien_correlated

# Must have
initial_value_stock = 100
initial_value_rates = 0.0032
number_of_simulation = 500
step_of_time = 1/12 # Not required in dict_parametre_simulation
freq_of_data = "monthly" #frequence of data
year_of_projection = 40
max_maturity = 40 # rates only
start_time = "2009/01/01" # Optional
finished_time = "2023/12/31" # Optional


dict_parametre_simulation_rates = {
    "initial value": initial_value_rates,
    "number of simulation": number_of_simulation * 2,
    "frequence of data": freq_of_data,
    "maturity maximal": max_maturity,
    "year of projection": year_of_projection,
    "studied period start": start_time,
    "finished period start": finished_time
}

# rate | action | immo
Crr = np.array(
    [[1, -0.08394538, 0.75497063],
    [-0.08394538, 1, 0.02903245],
    [0.75497063, 0.02903245, 1]]
    )

# parametre model Action / Immoblier
parametre_model_stock = [0.004128898,2.0103,0.0010,0.0000,0.0031,0.0057]
parametre_rates = [0.1, 0.02, 0.01]

W_bro = correlation_brownien(
            year_of_projection=year_of_projection,
            number_of_simulation=number_of_simulation,
            correlation_matrix=Crr,
            step_of_time=step_of_time,
            number_of_browniens=3
            )
list_of_brownien_correlated = get_brownien_correlated(W_bro)
mat_random_rates = list_of_brownien_correlated[0]
mat_random_action = list_of_brownien_correlated[1]
mat_random_immo = list_of_brownien_correlated[2]


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

data_matrice_transition = pd.read_excel(path_matrice_transition)
data_weight_courbe_souveraine = pd.read_excel(path_weight_courbe_souveraine)
data_cdt = pd.read_excel(path_courbe_taux)

# Must have
initial_value_stock = 100
initial_value_rates = -0.0032
number_of_simulation = 500
step_of_time = 1/12 # Not required in dict_parametre_simulation
freq_of_data = "monthly" #frequence of data
year_of_projection = 40
max_maturity = 40 # rates only
start_time = "2009/01/01" # Optional
finished_time = "2023/12/31" # Optional

dict_parametre_simulation_rates = {
    "initial value": initial_value_rates,
    "number of simulation": number_of_simulation * 2,
    "frequence of data": freq_of_data,
    "maturity maximal": max_maturity,
    "year of projection": year_of_projection,
    "studied period start": start_time,
    "finished period start": finished_time
}
from CourbeDesTaux_v0 import SmithWilson
ufr = 0.029
alpha = 0.128562
terms = np.arange(1, max_maturity+1)
terms_target = np.arange(1/12, max_maturity + 1/12, 1/12)
rates = np.array(data_cdt["rate"][terms])
extrapolation = SmithWilson(
    maturity=terms,
    zc=rates,
    is_price=False,
    maturity_extrapolate=terms_target,
    ufr=ufr,
    alpha=alpha
    )

extrapolation.fit_smithwilson_rates()

data_cdt_extrapolate = pd.DataFrame(
    np.reshape(extrapolation.zc_rates_extrapolate, newshape=(-1,1)),
    columns=["taux"])
data_cdt_extrapolate["maturite"] = extrapolation.maturity_extrapolate

m_rates =  G2_plus(
    parametre_simulation = dict_parametre_simulation_rates,
    data=data_cdt_extrapolate,
    random_matrix=mat_random_rates,
    parametre_model=[0.1, 0.3, 0.1, 0.3, -0.5]
    )
m_rates.parametre_model
m_rates.fit() # mauvais instrument
lp = m_rates.predict()

plt.plot(lp[:, 0, 0])
plt.show()

a = m_rates.get_parametre_return_to_mean_1()
sigma_1 = m_rates.get_parametre_volatilite_1()
b = m_rates.get_parametre_return_to_mean_2()
sigma_2 = m_rates.get_parametre_volatilite_2() 
rho = m_rates.get_parametre_correlation_instantaneous()

m_rates.get_list_parametre_from_dict()
m_rates.compute_objectif_function(
    np.array(data_cdt_extrapolate["taux"]),
    np.array(data_cdt_extrapolate["maturite"]),
    m_rates.get_list_parametre_from_dict())
data_filter = m_rates.import_data()
pzc = m_rates.get_price_zc_from_rate_zc(
            np.array(data_cdt_extrapolate["taux"]),
            np.array(data_cdt_extrapolate["maturite"]))
np.log(pzc[478]) - np.log(pzc[479])
m_rates.get_parametre_return_to_mean_1(m_rates.parametre_model)

plt.plot(np.array(data_cdt_extrapolate["taux"]))
plt.plot(rates)
plt.show()
plt.plot(pzc)
plt.show()

maturity_observed = np.array(data_cdt_extrapolate["maturite"])
price_data_observed = np.array(data_cdt_extrapolate["taux"])
