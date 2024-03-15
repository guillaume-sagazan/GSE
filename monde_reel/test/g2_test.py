import pandas as pd
import numpy as np
import os
from CourbeDesTaux_v0 import SmithWilson
from scipy import optimize
from scipy.linalg import cholesky
import logging
import matplotlib.pyplot as plt
from test.g2 import *

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
        W_bro: list | np.ndarray
        ) -> list:
    list_of_brownien_correlated = []
    dimension_correlation_brownien = np.shape(W_bro)
    for indice_brownien in range(dimension_correlation_brownien[1]):
        sub_brownien = W_bro[:, indice_brownien, :]
        brownien_correlated = np.column_stack((sub_brownien, -sub_brownien))
        list_of_brownien_correlated.append(brownien_correlated)
    return list_of_brownien_correlated

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
data_cdt_extrapolate["prix"]=1/(1+data_cdt_extrapolate["taux"])**(data_cdt_extrapolate["maturite"])


from monde_reel.test.g2 import *
fwd = calcul_inst_fwd(data_cdt_extrapolate["prix"], 1/12)
aleas1 = np.random.normal(size=(1000, 480))
aleas2 = np.random.normal(size=(1000, 480))

param = [1.5095, 1.53389, 0.0169, 0.01414, -0.04819]
inst_rates, x, y = generate_taux(param, aleas1, aleas2, 1/12, fwd)

matrice_zcp = calculate_prices_zc(param, data_cdt_extrapolate["prix"], x, y)
np.shape(matrice_zcp)
matrice_zcp[39, :, 1]
plt.plot(fwd)
plt.show()

step_of_time = 1/12
number_of_simulation = 1000
number_of_year_maturity = 40
year_projected = 8
year_annual_projected = np.arange(
        int(1/step_of_time),
        int((year_projected + 1)/step_of_time),
        int(1/step_of_time))
pzc = np.array(data_cdt_extrapolate["prix"])
pzc

ten = 12
indice = 5
pricing_zc(
    param,
    year_annual_projected/12,
    ten/12,
    pzc,
    pzc[2],
    x[3, year_annual],
    y[3, year_annual]
    )
year_annual_projected = np.array(data_cdt_extrapolate["maturite"])
year_annual = np.arange(0, 480, 1)
year_annual_maturity = np.arange(
        int(1/step_of_time),
        int((number_of_year_maturity + 1)/step_of_time),
        int(1/step_of_time))
pricing_pzc(
    param,
    year_annual_maturity/12,
    ten/12,
    pzc[year_annual_maturity-1],
    pzc[2],
    x[3, year_annual_maturity-1],
    y[3, year_annual_maturity-1])

pricing_pzc(
    param,
    year_annual_projected,
    2,
    pzc,
    pzc[2*12],
    x[4, year_annual],
    y[4, year_annual]
    )

matrice_zcp[4, 2, :]
np.shape(x)
np.shape(x[1])
year_annual_projected[1]

lp = calcul_moyenne_process(
    param,
    year_annual_projected, 8,
    x[10, year_annual], y[10, year_annual]
    )

plt.plot( x[2, year_annual])
plt.show()

plt.plot(inst_rates[10, year_annual])
plt.show()

5*12

pricing_pzc(
    param,
    5,
    year_annual_projected,
    pzc[5*12],
    pzc,
    x[4, year_annual],
    y[4, year_annual]
    )[60:]

param = [1.5095, 1.53389, 0.0169, 0.01414, -0.04819]
inst_rates, x, y = generate_taux(param, aleas1, aleas2, 1/12, fwd)
calculate_prices_zc(param, )

for indice_tenor, tenor in enumerate(year_annual_projected):
    PZC_T = pzc[tenor]
    for indice_maturity, maturity in enumerate(year_annual_maturity):
        PZC_t = pzc[maturity-1]
        prix = pricing_pzc(
                    param,
                    tenor,
                    maturity + tenor, 
                    PZC_T,
                    PZC_t,
                    x[:, tenor-1], y[:, tenor-1]
                    )
        print(tenor, PZC_T, maturity, PZC_t, indice_maturity, indice_tenor, np.mean(prix) )






