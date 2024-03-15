5 fichiers sont présents dans le dossier :
3 fichier de modélisation;
2 fichier de test (les test ne sont pas complets).

Le requirement.txt est présent dans l'archive du dossier principal.

Les fichiers de données doivent-être présent dans le dossier Monde réel, au format excel. (LEs colones peuvent être date pour la date et taux pour les taux 
et action pour les actions et l'immobilier) Ce point sera modifi dans une mis à jour ultérieur.

2 modèles sont présentes :
Un modèle de taux Vascicek calibrer et modéliser en monde réel présent au sein du fichier ModelTaux_RR.
Il est accompagné d'un fichier de test qui test quelques fonctions mais pas toutes.
Un modèle d'actif risque, qui regroupe action et l'immobilier, Merton calibrer et modéliser en monde réel présent au sein du fichier ModelAction_RR.
Il est accompagné d'un fichier de test qui test quelques fonctions mais pas toutes.

Chacun de ces fichiers est composé de class qui contient les methodes suivants:
fit -> pour calibrer (modifie l'object et met à jour l'attribut parametre_model qui par défaut est vide = {})
predict -> produit un np.ndarray de dimension (maturité, tenor, nombre de simulation) pour les taux et (année de projection, nombre de simulation) 
pour les actions ou l'immobilier.

Afin d'initialiser les objects les paramètres suivants doivent être inclus:

Pour les taux:
parametre_simulation qui prend en argument un dictionnaire
dict_parametre_simulation_rates = {
    "initial value": initial_value_rates,
    "number of simulation": number_of_simulation, >1
    "frequence of data": freq_of_data, (peut choisir entre : daily, monthly, annual, half-year, quarterly)
    "maturity maximal": max_maturity, >1
    "year of projection": year_of_projection, >1
    "studied period start": start_time, Optional si dans le fichier importer la colonne "date" est présent sinon aucun filtre n'est appliqué
    "finished period start": finished_time, Optional si dans le fichier importer la colonne "date" est présent sinon aucun filtre n'est appliqué
}
path_data = le chemin relatif pour accéder au fichier excel contenant la data
random_matrix = une matrice contenant l'aléatoire des simulations. La taille doit être de (year_of_projection * 1/periode_of_time) 
où periode_of_time est définie à l'aide de freq_of_data est correspond à une fraction d'année 1/12 pour le mois et 1 pour une année.
Optionelle: parametre_model si on dispose de donnée de calibration contenant un dictionnaire de valeur associé aux clés:
[speed of reversion (a), long terme mean (b), instantaneous volatility (sigma)]


Pour les actions et l'immobilier:
parametre_simulation qui prend en argument un dictionnaire
dict_parametre_simulation_stock = {
    "initial value": initial_value_stock, usuellement mis à 100 (ce n'est pas la valeur qui compte mais ces variations)
    "number of simulation": number_of_simulation, >1
    "frequence of data": freq_of_data, (peut choisir entre : daily, monthly, annual, half-year, quarterly)
    "year of projection": year_of_projection, >1
    "studied period start": start_time, Optional si dans le fichier importer la colonne "date" est présent sinon aucun filtre n'est appliqué
    "finished period start": finished_time Optional si dans le fichier importer la colonne "date" est présent sinon aucun filtre n'est appliqué
}
path_data = le chemin relatif pour accéder au fichier excel contenant la data
random_matrix = une matrice contenant l'aléatoire des simulations. La taille doit être de (year_of_projection * 1/periode_of_time) 
où periode_of_time est définie à l'aide de freq_of_data est correspond à une fraction d'année 1/12 pour le mois et 1 pour une année.
Optionelle: parametre_model = si on dispose de donnée de calibration contenant un dictionnaire de valeur (float) associé aux clés:
["alpha", "intensity of jump", "mean of jump", "standar deviation of jump", "volatility stock", "mean stock"]
ou s'il sagit d'une liste alors l'ordre doit être conservé. Un exemple est le suivant de list_parametre_model est le suivant contient les paramètres:
dict_parametre_model = {
            "alpha": float(list_parametre_model[0]),
            "intensity of jump": float(list_parametre_model[1]),
            "mean of jump": float(list_parametre_model[2]),
            "standar deviation of jump": float(list_parametre_model[3]),
            "volatility stock": float(list_parametre_model[4]),
            "mean stock": float(list_parametre_model[5])
        }
ou d'un dictionnaire sous la forme ci-dessus.
En cas d'abscence de valeur pour parametre_model des valeurs par défaut sont générées qui sont ensuite utilisés comme valeur initial pour la méthode fit.
Les valeurs initiales impactent énormément les résultats de fit. Il est conseillé de récupéré les valeurs réalisés par la précédent optimisation comme
point de départ. Le moteur de résolution du problème d'optimiastion probient de la bibliothèque optimize de scypi. Le moteur est le "SLSQP".
Il ne peut pas être modifié par le biais de parametre et se trouve dans la méthode get_optimization. Il a été choisit car il supporte les contraintes
lors de la résolution du problème. 

Pour créer de nouvelle class de modèles, il suffit d'ajouter un fichier ou à la suite et qui prend comme parent la class principal et
d'implémenter fit et predict en conservant les mêmes types de sorties!

Le 3ème fichier Projection_RR permet de réaliser l'exécution des classes précédemment présentés et en incluant la méthode de corrélation des browniens.
La méthode implémenter est une méthode de cholesky. Il faut pour cela une matrice de corrélation définie semi-positif.


