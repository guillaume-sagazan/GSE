import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt

taux_recouvrement = 30/100
dict_parametre_simulation_credit = {
    "initial value": initial_value_rates_all,
    "number of simulation": number_of_simulation * 2,
    "frequence of data": freq_of_data,
    "maturity maximal": max_maturity,
    "year of projection": year_of_projection,
    "taux recouvrement": taux_recouvrement,
    "matrice transition reel": matrice_transition_reel,
    "step of time": 1/12,
    "variable_output": "rates",
    "data_input_is_prices": True
}
import logging
m_rates_all = Vasicek_RR(
    parametre_simulation = dict_parametre_simulation_rates_all_t,
    data = dict_data['Taux_Allemand'],
    random_matrix = dict_alea['Taux_Allemand']
    )
m_rates_all.fit()
m_rates_all.parametre_model
data_model_tx = m_rates_all.predict("zero-coupon prices")
model_credit = G2(
    dict_parametre_simulation_credit,
    data = data_model_tx.shape,
    parametre_model = param_model_credit
    )
model_credit.predict()

def calcul_matrice_price_zero_coupon_by_note(
            self,
            list_matrice_transition: np.ndarray
            ) -> pd.DataFrame:
        nbr_annee_projection = self.get_year_of_projection()
        nbr_simulation = self.get_number_of_simulation()
        df_price_zc = self.get_zc_prices()
        list_name_year_projected = [f"{year}" for year in range(1,nbr_annee_projection+1)]
        # Exclud Default
        list_data_note_simu = []
        #Rajoute la variation des simulations
        for num_simu in range(nbr_simulation):
            df_price_zc_simulation = \
                df_price_zc[df_price_zc["simulation"] == num_simu][list_name_year_projected]
            for prob_default_for_notes in list_matrice_transition[num_simu, :, :]:
                # calcul_price_obligation_t
                pivot_df = self.calcul_price_obligation_t(
                                price_zc_credit = df_price_zc_simulation,
                                probabilite_defaut = prob_default_for_notes
                                )
                pivot_df["simulation"] = num_simu + 1
                list_data_note_simu.append(pivot_df)
        df_data_note = pd.concat(list_data_note_simu, axis=0)
        return df_data_note

list_matrice_transition = model_credit.calcul_matrice_transition()
calcul_matrice_price_zero_coupon_by_note(model_credit, list_matrice_transition)

def _initialize_data(data):
    if isinstance(data, np.ndarray):
        if len(data.shape) == 3:
            list_df_data_to_flat = [
                pd.DataFrame(
                    data[:, :, num_simu], columns=[f"{i}" for i in range(1, 9)]
                    ).assign(simulation=num_simu) for num_simu in range(1000)]
            df_transforme = pd.concat(list_df_data_to_flat, axis=0)
            df_transforme["maturite"] = df_transforme.index + 1
            df_transforme.reset_index(inplace=True, drop=True)
        else:
            message_error = f"The dimension of data"\
                +f" should be 3 but it is {len(data.shape)}"
            logging.error(message_error)
            raise ValueError(message_error)
    if isinstance(data, pd.DataFrame):
        if not("simulation" in data.columns and \
            "maturite" in data.columns):
            message_error = f"The columns of data"\
                + " should contains 'simulation' and 'maturite'"
            logging.error(message_error)
            raise ValueError(message_error)
    return df_transforme

_initialize_data(data_model_tx)

data_cdt = pd.DataFrame(data_model_tx[:, 0, 1].T, columns=["rate"]) / 100
list_df_data_to_flat = [
    pd.DataFrame(
        data_model_tx[:, :, num_simu] / 100, columns=[f"{i}" for i in range(1, 9)]
        ).assign(simulation=num_simu)
    for num_simu in range(1000)]
df_t = pd.concat( list_df_data_to_flat, axis=0)
df_t["maturite"] = df_t.index + 1
df_t.reset_index(inplace=True)
df_price_zc = df_t.copy()
for year_proj in [f"{i}" for i in range(1,9)]:
    df_price_zc[year_proj] = \
        1/(1+df_price_zc[year_proj])**(df_price_zc["maturite"])

param_model_credit = [0.1009, 9.5146, 0.7519]
# Test
model_credit = G2(
    dict_parametre_simulation_credit,
    data = df_price_zc,
    parametre_model = param_model_credit
    )

array_matrice_transition = model_credit.get_matrice_transition_reel()
matrice_gen_transition = \
model_credit.get_adapted_matrice_generatrice_reel(array_matrice_transition)
valeur_initial = np.array([0.9639] * 8)
modelisation_spread_year = model_credit.calcul_spread_sto(valeur_initial)
model_credit.get_initial_value()
dt = model_credit.get_frequence()
list_simu_year_proj = []
for num_simu in range(nbr_simulation):
    list_year_proj = []
    for year_proj in range(year_of_projection):
        model_spread_year = modelisation_spread_year[:, num_simu, year_proj]
        array_matrice_generatrice_transition = matrice_gen_transition[year_proj]
        matrice_transition = model_credit.calcul_matrice_transition_t(
                                    array_matrice_generatrice_transition,
                                    model_spread_year,
                                    dt)
        list_year_proj.append(matrice_transition[:-1, -1])
    list_simu_year_proj.append(list_year_proj)
list_prob_default = np.array(list_simu_year_proj)

model_spread_year.shape
####------ 2 fonctions
list_data_note_simu = []
df_price_zc = model_credit.get_zc_prices()
maturity_maximal = model_credit.get_maturity_maximal()
year_of_projection = model_credit.get_year_of_projection()
list_maturity =  np.array([
    iter
    for iter in range(1, maturity_maximal+1)
    for mat in range(1, year_of_projection+1)
    ])
list_name_year_projected = [f"{i}" for i in range(1,8+1)]

model_credit.get_matrice_transition_reel().shape[0]
model_credit.get_matrice_transition_reel().columns
list_col_rating = model_credit.get_matrice_transition_reel().columns[:-1]

list_data_note_simu = []
for num_simu in range(nbr_simulation):
    df_price_zc_simulation = \
        df_price_zc[df_price_zc["simulation"] == num_simu][list_name_year_projected]
    for name_rating, prob_default_for_notes in\
                zip(list_col_rating ,list_matrice_transition[num_simu, :, :].T):
        # calcul_price_obligation_t
        print(name_rating, prob_default_for_notes)

        pivot_df = model_credit.calcul_price_obligation_t(
                        price_zc_credit = df_price_zc_simulation,
                        probabilite_defaut = prob_default_for_notes
                        )
        pivot_df["simulation"] = num_simu + 1
        list_data_note_simu.append(pivot_df)
self = model_credit
df_data_note = pd.concat(list_data_note_simu, axis=0)
np.array(list_data_note_simu).shape
pd.DataFrame([df_zc_prices_with_note, df_zc_prices_with_note.shape])

list_prob_default.shape

pivot_df.shape
len(list_data_note_simu)

df_zc_prices_for_a_note.shape

df_price_zc_simulation.shape


df_price_zc_simulation * prob_default_for_notes


0.01048815 * 0.980589 - 0.010285