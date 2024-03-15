from monde_reel.ModelCredit_G2 import *
from matplotlib import pyplot as plt
from CourbeDesTaux_v0 import SmithWilson
import pandas as pd
import numpy as np
import os

initial_value_stock = 100
initial_value_rates = -0.0032
number_of_simulation = 500
step_of_time = 1/12 # Not required in dict_parametre_simulation
freq_of_data = "monthly" #frequence of data
year_of_projection = 8
max_maturity = 20 # rates only
start_time = "2009/01/01" # Optional
finished_time = "2023/12/31" # Optional
initial_value_rates_all = 2 / 100

matrice_transition_reel = pd.read_excel(
    os.path.join(os.getcwd(), "monde_reel", "matrice_transition.xlsx")
    )
matrice_transition_reel = matrice_transition_reel.set_index("note")
dict_parametre_simulation_rates_all_t = {
    "initial value": initial_value_rates_all,
    "number of simulation": number_of_simulation * 2,
    "frequence of data": freq_of_data,
    "maturity maximal": max_maturity,
    "year of projection": year_of_projection - 1
}
taux_recouvrement = 30/100
dict_parametre_simulation_credit = {
    "initial value": initial_value_rates_all,
    "number of simulation": number_of_simulation * 2,
    "frequence of data": freq_of_data,
    "maturity maximal": max_maturity,
    "year of projection": year_of_projection,
    "taux recouvrement": taux_recouvrement,
    "matrice transition reel": matrice_transition_reel,
    "step of time": 1/12
}
list_nom_note_credit = matrice_transition_reel.columns
m_rates_all = Vasicek_RR(
    parametre_simulation = dict_parametre_simulation_rates_all_t,
    data = dict_data['Taux_Allemand'],
    random_matrix = dict_alea['Taux_Allemand']
    )
m_rates_all.fit()
m_rates_all.parametre_model
data_model_tx = m_rates_all.predict()

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

data_cdt.index += 1
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
data_cdt_extrapolate["zero coupon prices"]= 1/(
    1+data_cdt_extrapolate["taux"])**(data_cdt_extrapolate["maturite"])

#-------------------------------------------------
param_model_credit = [0.1009, 9.5146, 0.7519]
model_credit = G2(
    dict_parametre_simulation_credit,
    data = data_cdt_extrapolate,
    parametre_model = param_model_credit
    )

valeur_initial = model_credit.get_initial_value()
model_credit.generate_taux_exact(param=[],
                initial_value=valeur_initial[0]
                )

model_credit.random_matrix.shape
model_credit.get_year_projected()
valeur_initial = model_credit.get_initial_value()
modelisation_spread_year = model_credit.calcul_spread_sto(valeur_initial)
valeur = model_credit.calcul_matrice_transition()
len(valeur[0][0].shape)
data = model_credit.predict()

data_model_tx.shape

np.zeros_like(matrice_transition_reel.shape)

matrice_transition = np.array(model_credit.calcul_matrice_transition()) 
# nbr simu x annee projete x Notation x Notation
matrice_zcp = data_cdt_extrapolate["zero coupon prices"].values
list_data = []
num_year_proj = 0
num_simu = 0
for indice_note in range(0, 8):
    val_note = model_credit.calcul_price_obligation_t(
                            price_zc = matrice_zcp[terms],
                            probabilite_defaut = matrice_transition[
                                num_simu, num_year_proj, indice_note, -1]
                            )
    list_data.append(val_note)
df_note = pd.DataFrame(np.array(list_data).T, columns=matrice_transition_reel.columns)
df_note.index = df_note.index + 1

df_note_tx = df_note.apply(
    lambda x: x**(-1/df_note.index) - 1,
    axis=0)

val_CCC = model_credit.calcul_price_obligation_t(
                        price_zc = matrice_zcp[terms],
                        probabilite_defaut = matrice_transition[0, 0, :, -1]
                        )

array_matrice_transition = model_credit.get_matrice_transition_reel()
nbr_simulation = model_credit.get_number_of_simulation()
nbr_annee_projete = model_credit.get_year_of_projection()
dt = model_credit.get_frequence()
valeur_initial = model_credit.get_initial_value()
valeur_initial = np.array([0.9639] * 8)
matrice_gen_transition = \
model_credit.get_adapted_matrice_generatrice_reel(array_matrice_transition)
modelisation_spread_year = model_credit.calcul_spread_sto(valeur_initial)
year_proj = 0

list_simu_year_proj = []
for num_simu in range(1000):
    list_year_proj = []
    for year_proj in range(7):
        model_spread_year = modelisation_spread_year[:, 0, year_proj]
        array_matrice_generatrice_transition = matrice_gen_transition[year_proj]
        matrice_transition = model_credit.calcul_matrice_transition_t(
                                    array_matrice_generatrice_transition,
                                    model_spread_year,
                                    dt)
        list_year_proj.append(matrice_transition[:-1, -1])
    list_simu_year_proj.append(list_year_proj)
array_year_proj = np.array(list_simu_year_proj)

array_year_proj.shape #Notation X année de proj
moyenne_year_proj = array_year_proj.mean(axis=0)
plt.plot(moyenne_year_proj, label=array_matrice_transition.columns[:-1])
plt.legend()
plt.show()

matrice_zcp = model_credit.get_zc_prices()
list_prob_default = matrice_transition[:-1, -1]
list_prob_default = array_year_proj
moyenne_year_proj = model_credit.get_zc_prices()
df_ref = pd.DataFrame(moyenne_year_proj)
df_ref_init = pd.DataFrame(matrice_zcp, columns=["zero coupon prices"])

list_data_note_simu = []
for num_simu in range(nbr_simulation):
    df_ref = pd.DataFrame()
    for prob_default_for_notes in list_prob_default[num_simu, :, :]:
        taux_recouvrement = model_credit.get_taux_recouvrement()
        arg_fluct_note = taux_recouvrement * prob_default_for_notes +\
            (1 - prob_default_for_notes)
        array_by_col = np.outer(df_ref_init["zero coupon prices"], arg_fluct_note)
        
        df_note_simu = pd.DataFrame(
            array_by_col,
            columns=[note for note in array_matrice_transition.columns[:-1]]
        )
        df_note_simu["simulation"] = num_simu + 1
        list_data_note_simu.append(df_note_simu)

df_data_note = pd.concat(list_data_note_simu, axis=0)


df_note_simu[df_note_simu.columns[:-1]].plot()
plt.show()


# Must have
initial_value_stock_immo = 100
initial_value_stock_action = 100
initial_value_rates_gouv = 1.58/100
initial_value_rates_corpo = 1.74/100
initial_value_rates_infla = 4.9/100
initial_value_rates_all = 3.1/100
initial_value_dividende = 3.9/100
initial_value_loyer = 4.2/100
taux_recouvrement = 30/100
number_of_simulation = 500
step_of_time = 1/12 # Not required in dict_parametre_simulation
freq_of_data = "monthly" #frequence of data
year_of_projection = 8
max_maturity = 20 # rates only

is_output_csv = True # False
path = os.getcwd()
path_data = os.path.join(path, "Donnee calibration")
dict_data = {}
list_files = [
    "Actions_Div", "Actions_Index",
    "Immo_index", "Immo_loyer",
    "Taux_Corpo", "Taux_Gouv",
    "Taux_inflation", "Taux_Allemand"
]

matrice_transition_reel = pd.read_excel(
    os.path.join(path, "monde_reel", "matrice_transition.xlsx"),
    sheet_name="test2"
    )
matrice_transition_reel = matrice_transition_reel.set_index("note")

dict_parametre_simulation_credit = {
    "initial value": initial_value_rates_all,
    "number of simulation": number_of_simulation * 2,
    "frequence of data": freq_of_data,
    "maturity maximal": max_maturity,
    "year of projection": year_of_projection +1,
    "taux recouvrement": taux_recouvrement,
    "matrice transition reel": matrice_transition_reel,
    "variable_output": "rates",
    "data_input_is_prices": True
}
#param_model_credit = [0.1009, 9.5146, 0.7519, 0.9]
param_model_credit = [0.1009, 9.5146, 0.7519, 2]
model_credit = G2(
    dict_parametre_simulation_credit,
    data_zc= data_model_prices,
    parametre_model = param_model_credit
    )
model_credit.parametre_model
model_credit.get_initial_value()
model_credit.get_matrice_transition_reel()
t= model_credit.predict()
list_column = [f"{i}" for i in range(1, 10)]

l = []
for i in range(1, 20+1):
    aaa = t[(t["Note"] == "AAA") & (t["maturite"] == i)][list_column].mean(axis=0)
    bbb = t[(t["Note"] == "BBB") & (t["maturite"] == i)][list_column].mean(axis=0)
    l.append((aaa - bbb) * 10**4)
m = pd.concat(l, axis=1)
t.to_excel(os.path.join(os.getcwd(), "obligation_t.xlsx"))
name_file = os.path.join("monde_reel","data_taux_court.xlsx")
lambda_t = 0.0355279589
mu_t = -0.02931970
sigma = 0.0000040031
valeur_initial = 0.03295
freq_of_data = "monthly"
dt = 1/12

dict_parametre_model = {
    "speed of reversion": lambda_t,
    "long terme mean": mu_t,
    "instantaneous volatility": sigma,
}
dict_parametre_simulation = {
    "step of time": dt,
    "year of projection": year_of_projection,
    "number of simulation" : number_of_simulation * 2,
    "maturity maximal": max_maturity,
    "initial value": valeur_initial,
    "frequence of data": freq_of_data
}
matrix_random = np.random.randn(
    int(year_of_projection / dt) + 1,
    number_of_simulation*2
    )

cdt_rr = Vasicek_RR(
    parametre_simulation = dict_parametre_simulation,
    path_data = name_file,
    random_matrix=matrix_random
)
cdt_rr.fit()
cdt_rr.parametre_model
data_model_prices = cdt_rr.predict('zero-coupon prices')


param_model_credit = [0.1009, 9.5146, 0.7519, 0.9]
param_model_credit = {
    'AAA': {'speed of reversion': 0.1009,
    'long terme mean': 9.5146,
    'instantaneous volatility': 0.7519},
    'CCC': {'speed of reversion': 1.1009,
    'long terme mean': 20,
    'instantaneous volatility': 2,
    "initial value": 10}
    }
param_model_credit = {'speed of reversion': 0.1009,
    'long terme mean': 9.5146,
    'instantaneous volatility': 0.7519}
model_credit = G2(
    dict_parametre_simulation_credit,
    data_zc_prices = data_model_prices,
    parametre_model = param_model_credit
    )
model_credit.parametre_model
model_credit.get_parametre_model()
self = model_credit
parametre_model = param_model_credit
dict_parametre_model_to_check = param_model_credit
dict_parametre_model = param_model_credit

get_initial_value(self)
self = model_credit
model_credit.fit()
model_credit.predict()


data**2


(1+0.015450) * (1 + 0.060181)