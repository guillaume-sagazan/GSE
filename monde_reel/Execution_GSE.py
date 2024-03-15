import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.linalg import cholesky
from monde_reel.ModelActions_RR import *
from monde_reel.ModelTaux_RR import *
from monde_reel.ModelTaux_Inflation import *
from monde_reel.ModelCredit_G2 import *

# Must have
initial_value_stock_immo = 1
initial_value_stock_action = 1
initial_value_rates_gouv = 1.58/100
initial_value_rates_corpo = 1.74/100
initial_value_rates_infla = 4.9/100
initial_value_rates_all = 3.1/100
initial_value_dividende = 3.9/100
initial_value_loyer = 4.2/100
taux_recouvrement = 30/100
number_of_simulation = 5000 # divided by 2, if you want 1000 -> 500
step_of_time = 1/12 # Not required in dict_parametre_simulation
freq_of_data = "monthly" #frequence of data
year_of_projection = 8
max_maturity = 30 # rates only

is_output_csv = True # False/True
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

def get_path_files_from_list_files(
        path_data: str,
        list_files: list
    ) -> list:
    list_path_files = []
    for file in list_files:
        path_file = os.path.join(path_data, file + ".xlsx")
        list_path_files.append(path_file)
    return list_path_files

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

def formatage_prediction(
        data_predict: np.ndarray,
        name_indice: str
        ) -> pd.DataFrame: 
    df_data_predict = pd.DataFrame(data_predict)
    df_data_predict.columns = ["An_" + str(annee)
                               for annee in df_data_predict.columns]
    df_data_predict["Nom_Serie"] = name_indice
    df_data_predict["No_Scenario"] = df_data_predict.index + 1
    return df_data_predict

def formatage_prediction_if_dim_3(
        data_predict: np.ndarray,
        name_indice: str
        ) -> pd.DataFrame:
    dim = data_predict.shape
    matrice_reshape = np.concatenate(
        [data_predict[:, :, i] for i in range(number_of_simulation * 2)],
        axis=0)
    df_data_predict = pd.DataFrame(matrice_reshape)
    df_data_predict.columns = \
        ["An_" + str(annee) for annee in df_data_predict.columns]
    df_data_predict["Nom_Serie"] = \
        [name_indice + "_" + str(keep_formate_mat(maturity)) 
            for num_simu in range(dim[-1])
                for maturity in range(1, dim[0]+1)
                ]
    df_data_predict["No_Scenario"] = \
        [num_simu + 1
            for num_simu in range(dim[-1])
                for maturity in range(1, dim[0]+1)]
    return df_data_predict

def compute_tx_reel(
        df_nom: pd.DataFrame,
        df_infla: pd.DataFrame,
        name_indice: str
        ) -> pd.DataFrame:
    """
    Return a dataframe with the shape of df_nom
    mineus the inflation by simulation from df_infla
    """
    df_reel = df_nom.copy()
    list_col = df_nom.columns[:-2] 
    list_df_reel = []
    for num_simu in range(1, number_of_simulation*2+1):
        df_simu =\
            df_nom[df_nom["No_Scenario"] == num_simu].reset_index(drop=True)
        df_simu[list_col] = df_simu[list_col].sub(
            df_infla[df_infla["No_Scenario"] == num_simu][list_col].iloc[0])
        list_df_reel.append(df_simu)
    df_reel = pd.concat(list_df_reel)
    df_reel.reset_index(drop=True, inplace=True)
    df_reel["Nom_Serie"] = df_reel["Nom_Serie"].apply(
        lambda x: name_indice + "_" +\
            str(x.split("_")[-1])
        )
    return df_reel

def get_df_rdt_index(
        df: pd.DataFrame,
        name_series_index:str
    ) -> pd.DataFrame:
    nom_series = name_series_index.split("_")[0]
    df_copy = df.copy()
    df_copy["An"] = df_copy["An_0"]
    for num_col in range(1, len(df_copy.columns[:-3])):
        list_col = df.columns
        df_copy[list_col[num_col]] = df[list_col[num_col]] / \
            df[list_col[num_col-1]] - 1
    df_copy = df_copy.drop("An", axis=1)
    df_copy["Nom_Serie"] = nom_series
    return df_copy

def get_df_rdt_capitalise(
        df: pd.DataFrame,
        name_series_index: str,
        year_of_projection: int
        ) -> pd.DataFrame:
    df_copy = df.copy()
    df_copy["An_0"] = 1
    for num_col in range(1, year_of_projection+1):
        nom_col = f"An_{num_col}"
        nom_col_pred = f"An_{num_col-1}"
        df_copy[nom_col] = df_copy[nom_col_pred] * (1+df_copy[nom_col])
    df_copy["Nom_Serie"] = name_series_index
    return df_copy

def get_df_index_rdt(
        df: pd.DataFrame,
        name_series_index:str
    ) -> pd.DataFrame:
    df_copy = df.copy()
    df["An"] = 0
    df_copy["An"] = 0
    list_col = np.sort(df_copy.columns)
    df = df[list_col]
    df_copy = df_copy[list_col]
    for num_col in range(1, len(list_col[:-2])):
        df_copy[list_col[num_col]] = (df[list_col[num_col]] + 1) * \
            (df[list_col[num_col-1]] + 1)
    df_copy["Nom_Serie"] = name_series_index
    df_copy[list_col[:-2]] = df_copy[list_col[:-2]] / \
        df_copy.iloc[0,1]
    df_copy = df_copy.drop("An", axis=1)
    return df_copy

def adapt_model_gouv_into_model_corpo(
        df_rates_gouv: pd.DataFrame,
        spread_corpo: np.ndarray
    ) -> pd.DataFrame:
    nbr_maturite, nbr_annee_proj, nbr_simulation = spread_corpo.shape
    list_columns_name = [f"{year}" for year in range(1, nbr_annee_proj + 1)]
    list_df_data_to_flat = [
        pd.DataFrame(
            spread_corpo[:, :, num_simu],
            columns=list_columns_name
            ).assign(simulation=num_simu)
        for num_simu in range(nbr_simulation)]
    df_spread_rates_corpo = pd.concat(list_df_data_to_flat, axis=0)
    df_spread_rates_corpo["maturite"] = df_spread_rates_corpo.index + 1
    df_spread_rates_corpo.reset_index(inplace=True, drop=True)
    df_rates_gouv.reset_index(inplace=True, drop=True)
    df_rates_corpo = df_rates_gouv.copy()
    df_spread_to_add = df_spread_rates_corpo.reset_index()
    list_data_corpo = []
    for note in df_rates_gouv.Note.unique():
        mask_note = df_rates_gouv.Note == note
        df_note_to_add = df_rates_gouv[mask_note].reset_index()
        df_data_corpo = df_spread_to_add[list_columns_name] +\
            df_note_to_add[list_columns_name]
        df_data_corpo["Note"] = note
        df_data_corpo["simulation"] = df_note_to_add["simulation"]
        df_data_corpo["maturite"] = df_note_to_add["maturite"]
        list_data_corpo.append(df_data_corpo)
    df_rates_corpo = pd.concat(list_data_corpo, axis=0)
    return df_rates_corpo

def keep_formate_mat(mat: int | str) -> str:
        maturite_formate = str(mat)
        if len(maturite_formate) == 1:
            maturite_formate = "0" + maturite_formate
        return maturite_formate[:]

def formate_prediction_credit(
        data_simulation: pd.DataFrame,
        name_index: str,
        str_Govt_or_Corp: str):
    list_name_year_projected = [f"{i}" for i in range(1,year_of_projection+2)]
    list_name_year_projected_rename = \
        [f"An_{i}" for i in range(year_of_projection+1)]
    data = data_simulation.copy()
    data["maturite"] = data["maturite"].apply(lambda mat: keep_formate_mat(mat) )
    data["Nom_Serie"] = name_index + "_" + data["Note"] +\
        "_" + str_Govt_or_Corp + "_" + data["maturite"]
    data.rename(columns={"simulation": "No_Scenario"}, inplace=True)
    dict_rename_year_proj = dict(
        zip(list_name_year_projected, list_name_year_projected_rename)
        )
    data.rename(columns=dict_rename_year_proj, inplace=True) 
    data.drop(["Note", "maturite"], axis=1, inplace=True)
    data.reset_index(drop=True, inplace=True)
    return data

def add_spread_to_rate_gouv(df_predict_rating_gouv : pd.DataFrame):
    list_rating_gouv = []
    for enu, note in enumerate(matrice_transition_reel.columns[:-1]):
        if enu == 1 or enu == 2:
            mnt_spread = enu * 0.003
        else:
            mnt_spread = enu * 0.005
        list_rating_gouv.append( df_predict_rating_gouv[
                [str(i) for i in range(1, 10)]].loc[
                df_predict_rating_gouv["Note"] == note,:
                ] + mnt_spread )
    df_data_gouv = pd.concat(list_rating_gouv)
    df_data_gouv[["Note", "maturite", "simulation"]] =\
        df_predict_rating_gouv[["Note", "maturite", "simulation"]]
    return df_data_gouv

dict_remane_actif = {
    "Actions_Index" : "Action",
    "Immo_index" : "Action",
    "Actions_Div" : "taux",
    "Immo_loyer": "taux",
    "Taux_Corpo" : "taux",
    "Taux_Gouv": "taux",
    "Taux_inflation": "Inflation",
    "Taux_Allemand": "taux"
}

dict_data = {}
list_data_to_correl = []
list_path = get_path_files_from_list_files(path_data, list_files)
for name_actif, path_file in zip(list_files, list_path):
    data_imported = pd.read_excel(path_file, sheet_name="Data")
    data_imported = data_imported.sort_values("Date")
    data_imported = data_imported.drop("Date", axis=1)
    data_imported.columns = [dict_remane_actif[name_actif]]
    dict_data[name_actif] = data_imported
    list_data_to_correl.append(data_imported)
data_to_correl = pd.concat(list_data_to_correl, axis=1)
data_to_correl.columns = dict_remane_actif.keys()
correlation_matrix = data_to_correl.corr()
# Calcul des valeurs propres et vecteurs propres
valeurs_propres, vecteurs_propres = \
    np.linalg.eig(correlation_matrix)

# Transformation pour avoir des valeurs propres positives
matrice_transformee = \
    np.dot(vecteurs_propres,
           np.dot(np.diag(np.abs(valeurs_propres)),
                  np.linalg.inv(vecteurs_propres)))
for i in range(len(matrice_transformee)):
    matrice_transformee[:,i] = matrice_transformee[:,i] / matrice_transformee[i,i]
mat_0_sup = np.triu(matrice_transformee)
mat_0_inf = np.tril(mat_0_sup.T)
Crr = cholesky((mat_0_sup + mat_0_inf)/2, lower=True)
for nom_index in ["Actions_Index", "Immo_index"]:
    dict_data[nom_index].loc[-1] = 0
    dict_data[nom_index].index += 1
    dict_data[nom_index].sort_index(inplace=True)
    dict_data[nom_index] = 1 + dict_data[nom_index]
    dict_data[nom_index] = \
        dict_data[nom_index].cumprod() 

W_bro = correlation_brownien(
            year_of_projection=year_of_projection,
            number_of_simulation=number_of_simulation,
            correlation_matrix=Crr,
            step_of_time=step_of_time,
            number_of_browniens=8
            )
list_of_brownien_correlated = get_brownien_correlated(W_bro)
dict_alea = {}
for enu, key in enumerate(dict_remane_actif):
    dict_alea[key] = list_of_brownien_correlated[enu]

dict_parametre_simulation_rates_gouv = {
    "initial value": initial_value_rates_gouv,
    "number of simulation": number_of_simulation * 2,
    "frequence of data": freq_of_data,
    "maturity maximal": max_maturity,
    "year of projection": year_of_projection
}

dict_parametre_simulation_rates_corpo = {
    "initial value": initial_value_rates_corpo,
    "number of simulation": number_of_simulation * 2,
    "frequence of data": freq_of_data,
    "maturity maximal": max_maturity,
    "year of projection": year_of_projection
}

dict_parametre_simulation_rates_infla = {
    "initial value": initial_value_rates_infla,
    "number of simulation": number_of_simulation * 2,
    "frequence of data": freq_of_data,
    "maturity maximal": max_maturity,
    "year of projection": year_of_projection
}

dict_parametre_simulation_spread = {
    "initial value": 0.03,
    "number of simulation": number_of_simulation * 2,
    "frequence of data": freq_of_data,
    "maturity maximal": max_maturity,
    "year of projection": year_of_projection
}

dict_parametre_simulation_rates_all = {
    "initial value": initial_value_rates_all,
    "number of simulation": number_of_simulation * 2,
    "frequence of data": freq_of_data,
    "maturity maximal": max_maturity,
    "year of projection": year_of_projection
}

dict_parametre_simulation_rates_gouv_pour_infla = {
    "initial value": initial_value_rates_gouv,
    "number of simulation": number_of_simulation * 2,
    "frequence of data": freq_of_data,
    "maturity maximal": max_maturity,
    "year of projection": year_of_projection
}

dict_parametre_simulation_stock = {
    "initial value": initial_value_stock_action,
    "number of simulation": number_of_simulation * 2,
    "frequence of data": freq_of_data,
    "maturity maximal": max_maturity,
    "year of projection": year_of_projection
}

dict_parametre_simulation_dividende = {
    "initial value": initial_value_dividende,
    "number of simulation": number_of_simulation * 2,
    "frequence of data": freq_of_data,
    "maturity maximal": max_maturity,
    "year of projection": year_of_projection
}

dict_parametre_simulation_immo = {
    "initial value": initial_value_stock_immo,
    "number of simulation": number_of_simulation * 2,
    "frequence of data": freq_of_data,
    "maturity maximal": max_maturity,
    "year of projection": year_of_projection
}

dict_parametre_simulation_loyer = {
    "initial value": initial_value_loyer,
    "number of simulation": number_of_simulation * 2,
    "frequence of data": freq_of_data,
    "maturity maximal": max_maturity,
    "year of projection": year_of_projection
}

dict_parametre_simulation_credit = {
    "initial value": initial_value_rates_all*100,
    "number of simulation": number_of_simulation * 2,
    "frequence of data": freq_of_data,
    "maturity maximal": max_maturity,
    "year of projection": year_of_projection +1, # Neede to add one year
    "taux recouvrement": taux_recouvrement,
    "matrice transition reel": matrice_transition_reel,
    "step of time": 1/12,
    "variable_output": "rates",
    "data_input_is_prices": True
}

parametre_model_stock = [0.004128898,2.0103,0.0010,0.0000,0.0031,0.0057]
m_act = Merton_RR(
    parametre_simulation=dict_parametre_simulation_stock,
    random_matrix=dict_alea['Actions_Index'],
    data=dict_data['Actions_Index'],
    parametre_model=parametre_model_stock
)

m_immo = Black_Scholes_RR(
    parametre_simulation=dict_parametre_simulation_immo,
    random_matrix=dict_alea['Actions_Index'],
    data=dict_data['Immo_index'],
    parametre_model=[initial_value_rates_gouv]
)

m_loyer =  Vasicek_RR(
    parametre_simulation = dict_parametre_simulation_loyer,
    data = dict_data['Immo_loyer'],
    random_matrix=dict_alea['Immo_loyer']
)

m_dividende = Vasicek_RR(
    parametre_simulation = dict_parametre_simulation_dividende,
    data = dict_data['Actions_Div'],
    random_matrix=dict_alea['Actions_Div']
)

m_rates_all_gouv = Vasicek_RR(
    parametre_simulation = dict_parametre_simulation_rates_all,
    data = dict_data['Taux_Allemand'],
    random_matrix=dict_alea['Taux_Allemand']
)

m_rates_n_pour_infla = Vasicek_RR(
    parametre_simulation = dict_parametre_simulation_rates_gouv_pour_infla,
    data = dict_data['Taux_Gouv'],
    random_matrix=dict_alea['Taux_Gouv']
)

m_rates_r_pour_infla = Vasicek_RR(
    parametre_simulation = dict_parametre_simulation_rates_gouv_pour_infla,
    data = dict_data['Taux_Gouv'],
    random_matrix=dict_alea['Taux_Gouv']
)

m_rates_spread = Vasicek_RR(
    parametre_simulation = dict_parametre_simulation_spread,
    data = dict_data['Taux_Corpo'] - dict_data['Taux_Gouv'],
    random_matrix=dict_alea['Taux_Gouv']
)

m_infla = ModelInflation_RR(
    rate_model_nominaux = m_rates_n_pour_infla,
    rate_model_reel = m_rates_r_pour_infla,
    parametre_simulation = dict_parametre_simulation_rates_infla,
    data = dict_data['Taux_inflation']
)

list_year = np.arange(0, 97, 12)

m_infla.fit()
m_dividende.fit()
m_loyer.fit()
m_immo.fit()
m_act.fit()
m_rates_all_gouv.fit()
m_rates_spread.fit()

# Modification des parmaetres pour correspondre aux attentes
m_dividende.parametre_model['speed of reversion'] /= 81
m_loyer.parametre_model['speed of reversion'] /= 20

# action
# Le Vox n'est jamais en dessous de 10% à partir de 2000
m_act.parametre_model['volatility stock'] = 0.16 
m_act.parametre_model['alpha'] = 0.7
m_act.parametre_model["mean stock"] = 0.7
m_act.parametre_model['standar deviation of jump'] = 1e-7
m_act.parametre_model['intensity of jump'] = 2.5

dict_infla_n = \
    m_infla.get_rate_model_nominal().parametre_model
dict_infla_r = \
    m_infla.get_rate_model_reel().parametre_model

dict_infla_n['speed of reversion'] = 1 # 0.8
dict_infla_r['speed of reversion'] = 0.5 # 0.4

dict_infla_n["long terme mean"] = 3/100
dict_infla_r["long terme mean"] = 1/100

# Spread Corpo
m_rates_spread.parametre_model
m_rates_spread.parametre_model['long terme mean'] = 0.017
m_rates_spread.parametre_model['instantaneous volatility'] *= 1000
m_rates_spread.parametre_model['speed of reversion'] = 4

predict_model_infla = m_infla.predict().T

predict_model_dividende = m_dividende.predict("short_rates")[list_year].T
predict_model_loyer = m_loyer.predict("short_rates")[list_year].T
predict_model_short_rates = m_rates_all_gouv.predict("short_rates")[list_year].T
predict_model_rates_gouv_all = m_rates_all_gouv.predict()
predict_model_immo = m_immo.predict()
predict_model_act = m_act.predict()

data_model_prices = m_rates_all_gouv.predict("zero-coupon prices")

param_model_credit = [0.1009, 9.5146, 0.7519]
model_credit = G2(
    dict_parametre_simulation_credit,
    data_zc = data_model_prices,
    parametre_model = param_model_credit
    )

df_predict_rating_gouv = model_credit.predict()
df_predict_rating_gouv = add_spread_to_rate_gouv(df_predict_rating_gouv)
predict_spread = m_rates_spread.predict()
df_predict_rating_corpo = adapt_model_gouv_into_model_corpo(
        df_rates_gouv = df_predict_rating_gouv,
        spread_corpo = predict_spread)

df_model_credit_corpo = formate_prediction_credit(
    df_predict_rating_corpo,
    name_index="CRBTXN",
    str_Govt_or_Corp="Corp")
df_model_credit_gouv = formate_prediction_credit(
    df_predict_rating_gouv,
    name_index="CRBTXN",
    str_Govt_or_Corp="Govt")

df_model_idx_act = formatage_prediction(
    predict_model_act,
    name_indice="ACTIONS_IDX")
df_model_idx_immo = formatage_prediction(
    predict_model_immo,
    name_indice="IMMOBILIER_IDX")
df_model_infla = formatage_prediction(
    predict_model_infla,
    name_indice="TXINFL")
df_model_short_rate = formatage_prediction(
    predict_model_short_rates,
    name_indice="TXCTN")
df_model_dividende = formatage_prediction(
    predict_model_dividende,
    name_indice="ACTIONS_DIV_TX")
df_model_loyer = formatage_prediction(
    predict_model_loyer,
    name_indice="IMMOBILIER_DIV_TX")
df_model_rates_nom_gouv_all = formatage_prediction_if_dim_3(
    predict_model_rates_gouv_all,
    name_indice="CRBTXN_N_SRIS")
df_model_rates_reel_gouv_all = compute_tx_reel(
    df_nom = df_model_rates_nom_gouv_all,
    df_infla = df_model_infla,
    name_indice="CRBTXR_R_SRIS")

df_model_rdt_act = get_df_rdt_index(
    df_model_idx_act,
    name_series_index="ACTIONS")
df_model_rdt_act["An_0"] = 0 
df_model_rdt_immo = get_df_rdt_index(
    df_model_idx_immo,
    name_series_index="IMMOBILIER")
df_model_rdt_immo["An_0"] = 0
df_model_idx_infla = get_df_index_rdt(
    df_model_infla,
    name_series_index="TXINFL_IDX")
df_model_idx_short_rate = get_df_index_rdt(
    df_model_short_rate,
    name_series_index="TXCTN_IDX")

#last modificaiton
df_model_infla.drop("An", axis=1, inplace=True)
df_model_short_rate.drop("An", axis=1, inplace=True)

# Capitalisation
df_model_idx_act = get_df_rdt_capitalise(
        df=df_model_rdt_act,
        name_series_index="ACTIONS_IDX",
        year_of_projection=year_of_projection)
df_model_idx_immo = get_df_rdt_capitalise(
        df=df_model_rdt_immo,
        name_series_index="IMMOBILIER_IDX",
        year_of_projection=year_of_projection)
df_model_idx_infla = get_df_rdt_capitalise(
        df=df_model_infla,
        name_series_index="TXINFL_IDX",
        year_of_projection=year_of_projection)
df_model_idx_short_rate = get_df_rdt_capitalise(
        df=df_model_short_rate,
        name_series_index="TXCTN_IDX",
        year_of_projection=year_of_projection)

list_data = [
    df_model_idx_act, df_model_rdt_act, df_model_dividende,
    df_model_idx_immo, df_model_rdt_immo, df_model_loyer,
    df_model_infla, df_model_idx_infla,
    df_model_short_rate, df_model_idx_short_rate,
    df_model_rates_nom_gouv_all, df_model_rates_reel_gouv_all,
    df_model_credit_corpo, df_model_credit_gouv
    ]

#test
df_model_result = pd.concat(list_data)
list_col_annee_prof = [f"An_{year}" for year in range(year_of_projection+1)]
list_colonne = ["Nom_Serie"] + list_col_annee_prof + ["No_Scenario"]
df_model_result = df_model_result[list_colonne]

if is_output_csv:
    path_output = os.path.join(
        os.getcwd(),
        "Donnee calibration",
        f"Output_GSE_{number_of_simulation*2}.csv")
    df_model_result.to_csv(path_output, index=False, header=True, sep=";")
else:
    path_output = os.path.join(
        os.getcwd(),
        "Donnee calibration",
        f"Output_GSE_{number_of_simulation*2}.xlsx")
    df_model_result.to_excel(
        path_output,
        index=False,
        header=True,
        sheet_name="Data")

# Analyse
def get_statistic_scenario(
        data_simulation: pd.DataFrame,
        col_to_compute_stat: list,
        name_serie: str
        ) -> pd.DataFrame:
    """
    Return a dataframe that contains the statistic for each column put in 
    argument : col_to_compute_stat and for each Serie defined in name_serie.
    The statistic defined are :
        max : maximun
        min: minimun
        mean : moyenne
        std : ecart type
        quantil : 1, 10 25, 50, 75, 90, 99
    """
    list_data_stat = []
    data_groupby_col_select = data_simulation.groupby(
        name_serie)[col_to_compute_stat]
    df_max = data_groupby_col_select.max()
    df_min = data_groupby_col_select.min()
    df_moy = data_groupby_col_select.mean()
    df_std = data_groupby_col_select.std()
    df_quant_01 = data_groupby_col_select.quantile(0.01)
    df_quant_10 = data_groupby_col_select.quantile(0.1)
    df_quant_25 = data_groupby_col_select.quantile(0.25)
    df_quant_50 = data_groupby_col_select.quantile(0.50)
    df_quant_75 = data_groupby_col_select.quantile(0.75)
    df_quant_90 = data_groupby_col_select.quantile(0.90)
    df_quant_99 = data_groupby_col_select.quantile(0.99)
    list_data_stat = [
        df_max, df_min, df_moy, df_std, df_quant_01, df_quant_10,
        df_quant_25, df_quant_50, df_quant_75, df_quant_90, df_quant_99
    ]
    list_statistic = [
        "Maximun", "Minimun" ,"Moyenne", "Ecart-Type",
        "Quantil 1%", "Quantil 10%", "Quantil 25%", "Quantil 50%",
        "Quantil 75%", "Quantil 90%", "Quantil 99%"
        ]
    df_statistic = pd.concat(list_data_stat, axis=0)
    df_statistic["Statistic"] = list_statistic
    df_statistic = df_statistic[["Statistic"] + col_to_compute_stat]
    return df_statistic

list_col = [f"An_{year}" for year in range(year_of_projection+1)]
stat_rdt_act = get_statistic_scenario(
    data_simulation=df_model_rdt_act,
    col_to_compute_stat=list_col,
    name_serie="Nom_Serie")


df_stat_infla = get_statistic_scenario(
    data_simulation=df_model_infla,
    col_to_compute_stat=list_col,
    name_serie="Nom_Serie")


tb_stat_corpo = df_predict_rating_corpo.groupby(
    ["Note", "maturite"])[[str(i) for i in range(1, 10)]]
tb_stat_gouv = df_predict_rating_gouv.groupby(
    ["Note", "maturite"])[[str(i) for i in range(1, 10)]]

df_spread_ecart_corpo_gouv = tb_stat_corpo.mean() - tb_stat_gouv.mean()
df_spread_ecart_corpo_gouv[df_spread_ecart_corpo_gouv.index.isin(["AAA"], level=0)]

df_data_gouv = df_predict_rating_gouv

gouv_B = df_data_gouv[df_data_gouv["Note"] == "B"].groupby(
    ["maturite"])[[str(i) for i in range(1, 10)]].mean()
gouv_BB = df_data_gouv[df_data_gouv["Note"] == "BB"].groupby(
    ["maturite"])[[str(i) for i in range(1, 10)]].mean()
gouv_BBB = df_data_gouv[df_data_gouv["Note"] == "BBB"].groupby(
    ["maturite"])[[str(i) for i in range(1, 10)]].mean()
gouv_AA = df_data_gouv[df_data_gouv["Note"] == "AA"].groupby(
    ["maturite"])[[str(i) for i in range(1, 10)]].mean()
gouv_A = df_data_gouv[df_data_gouv["Note"] == "A"].groupby(
    ["maturite"])[[str(i) for i in range(1, 10)]].mean()
gouv_AAA = df_data_gouv[df_data_gouv["Note"] == "AAA"].groupby(
    ["maturite"])[[str(i) for i in range(1, 10)]].mean()

df_spread_gouv_BBB = gouv_BBB - gouv_AAA
df_spread_gouv_BB = gouv_BB - gouv_AAA
df_spread_gouv_B = gouv_B - gouv_AAA
df_spread_gouv_AA = gouv_AA - gouv_AAA
df_spread_gouv_A = gouv_A - gouv_AAA

path_output_statistic = os.path.join(
    os.getcwd(),
    "Donnee calibration",
    f"Output_GSE_statistic.xlsx")
with pd.ExcelWriter(path_output_statistic, engine='openpyxl') as writer:
    # spread entre rates gouv
    df_spread_gouv_AA.to_excel(
        writer,
        sheet_name="Spread Gouv AA",
        index=True)
    df_spread_gouv_A.to_excel(
        writer,
        sheet_name="Spread Gouv A",
        index=True)
    df_spread_gouv_BBB.to_excel(
        writer,
        sheet_name="Spread Gouv BBB",
        index=True)
    df_spread_gouv_BB.to_excel(
        writer,
        sheet_name="Spread Gouv BB",
        index=True)
    df_spread_gouv_B.to_excel(
        writer,
        sheet_name="Spread Gouv B",
        index=True)
    # spread entre rates gouv et corporate
    df_spread_ecart_corpo_gouv.to_excel(
        writer,
        sheet_name="Spread Corpo",
        index=True)
    # Statistic
    df_stat_infla.to_excel(
        writer,
        sheet_name="Statistic Infla",
        index=True)
    stat_rdt_act.to_excel(
        writer,
        sheet_name="Statistic Action",
        index=True)

