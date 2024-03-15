from PRIIPS import *
import pandas as pd
import numpy as np
import os

data_type_2 = pd.read_excel(
    os.path.join(os.getcwd(), "PRIIPS", "Data", "test_donnee_PRIIPS.xlsx"),
    sheet_name="rndt type 2")
matrix_yield = np.array(data_type_2["Rendements"])
matrix_rndt_type_4_fav = pd.read_excel(
    os.path.join(os.getcwd(), "PRIIPS", "Data", "test_donnee_PRIIPS.xlsx"),
    sheet_name="rndt type 4 favorable", index_col=0)
matrix_rndt_type_4_defav = pd.read_excel(
    os.path.join(os.getcwd(), "PRIIPS", "Data", "test_donnee_PRIIPS.xlsx"),
    sheet_name="rndt type 4 defavorable", index_col=0)
matrix_yield_fav = np.array(matrix_rndt_type_4_fav)
matrix_yield_defav = np.array(matrix_rndt_type_4_defav)

dict_parametre_scenario_performance = {
    "rhp": 8,
    "category": 4,
    "mnt_investment_initial": 10000, # Reglementaire,
    "data_frequency": "monthly prices", # Mandatory for cat 2 otherwise no
    "est_cout_fil_du_temps_integre": False,
    "dict_quantil": {
                'fav': 90,
                'defav': 10,
                'middle': 50
                },
    "dict_taxes": {
        "rate_chargement_euro": 0.6/100,
        "rate_admission_fees": 2.4/100,
        "rate_management_fees_on_outstandings": 0.6/100,
        "rate_fees_on_global_actif": 0.00012974754562602,
        "cost_of_transaction": 0.07/100,
        "other_cost": 0.16/100,
        "cost_fees_linked_to_results": 0,
        "cost_linked_to_incentive_commissions": 0
        },
    "dict_contract": {
        "type_of_support": "Monosupport", #"Multisupport", # si categorie=4
        "TMGA": 0/100, # net
        "TMG": 0.6/100 # uniquement pour la categorie 4
    }
}

dict_matrix_yield = {
    "favorable":matrix_yield_fav,
    "defavorable":matrix_yield_defav
}

obj_builder_scn_perf = ScenarioPerformanceBuilder()
scn_perf = obj_builder_scn_perf.create_object(
    dict_matrix_yield=dict_matrix_yield,
    dict_of_parametre=dict_parametre_scenario_performance,
    is_logging_activate=True
)

performance_brut = scn_perf.performance_funct()
performance_net = scn_perf.performance_funct_net()
rendement_annuel_net = scn_perf.get_yield_mean(performance_net)












