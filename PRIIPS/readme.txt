Le dossier PRIIPS contient plusieurs classes et 2 dossiers:
- Un dossier qui contient la documentation dans Documentation et les requirements.
- Un dossier qui contient les données utilisés pour réaliser les tests
- Les classes.

Le document "scenario_performance_builder" contient le builder des classes
"scenario_performance" qui permettent leurs initialisation.
On en dénombre 4 pour chaque catégories qui hérite tous de "scenario_performance".

Le builder prend en entré 2 arguments obligatoire et 1 optionnel.
Les argument obligatoire sont :
- dict_matrix_yield: qui contient les matrices de rendements
- dict_parametre_scenario_performance qui contient les parametres.

L'argument optionnel est "is_logging_activate" qui indique sur quel 
fichier on crée et écrit les logs.


Un exemple de dict_matrix_yield peut être:
{
    "favorable":matrix_yield_fav,
    "defavorable":matrix_yield_defav
}

Un exemple de dict_parametre_scenario_performance peut être:
{
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

Un exemple d'initialisation d'un objet à l'aide du builder:
obj_builder_scn_perf = ScenarioPerformanceBuilder()
scn_perf = obj_builder_scn_perf.create_object(
    dict_matrix_yield=dict_matrix_yield,
    dict_of_parametre=dict_parametre_scenario_performance,
    is_logging_activate=True
)

Les 3 fonctions d'appelle sont:
performance_brut = scn_perf.performance_funct() # Performance brute
performance_net = scn_perf.performance_funct_net() # Performance nette
rendement_annuel_net = scn_perf.get_yield_mean(performance_net) # Rendement
net

Pour plus d'information regarde la documentation.
