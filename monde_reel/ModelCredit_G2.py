import numpy as np
import pandas as pd
from monde_reel.ModelTaux_RR import *
import os
import logging

class ModelCredit_RR:
    """
    Le modèle de Crédit prend à minima les arguments suivants: 
        parametre_simulation : dict, Contient les informations propres 
        à la simulations.
            "year of projection": annee_projection,
            "number of simulation" : nombre_simulation,
            "maturity maximal": max_maturity,
            "initial value": valeur_initial,
            "frequence of data": freq_of_data
        path_data: str = "" if present will be used to extract the informations
        from data
        data: pd.DataFrame | np.ndarray | dict = [] Informations relatives to the 
        projection and the calibration
        The information needed is a courbe des taux 
    """

    def __init__(
            self,
            parametre_simulation : dict,
            path_data: str = "",
            data_zc: pd.DataFrame | np.ndarray | dict = [],
            ) -> None:
        self.parametre_simulation = parametre_simulation
        self.path_data = path_data
        self.data_zc = data_zc
    
    def get_parametre_simulation(self):
        parametre_simulation = self.parametre_simulation
        return parametre_simulation
    
    def get_path_data(self):
        return self.path_data
    
    def get_data_zc(self):
        return self.data_zc

    @abstractmethod
    def fit(self) -> None:
        pass

    @abstractmethod
    def predict(self):
        pass

    def _initiliaze_frequence_of_data(self) -> None:
        period_map = {
            "annual": 1.0,
            "half-yearly": 0.5,
            "quarterly": 0.25, 
            "monthly": 1/12,
            "daily": 1/360
            }
        if "frequence of data" in self.parametre_simulation and \
                self.parametre_simulation["frequence of data"] in period_map.keys():
            self.parametre_simulation["step of time"] = \
                period_map[self.parametre_simulation["frequence of data"]]
        else:
            message_error = f"There is not such period \
                {self.parametre_simulation['frequence of data']} in \
                {period_map.keys()}."
            logging.error(message_error)
            raise ValueError(message_error)
    
    def import_data(self):
        try:
            if self.path_data != "":
                path_data_completed = os.path.join(
                    os.getcwd(),
                    self.path_data
                    )
                data_imported = pd.read_excel(path_data_completed)
            else:
                data_imported = self.data_zc
        except ValueError as e: 
            logging.error(e)
            raise e
        return data_imported

class G2(ModelCredit_RR):

    def __init__(
            self,
            parametre_simulation: dict,
            random_matrix: np.ndarray = "",
            path_data: str = "",
            data_zc: pd.DataFrame | np.ndarray | dict = [],
            parametre_model: dict = {},
            ) -> None:
        super().__init__(parametre_simulation, path_data, data_zc)
        self._initiliaze_frequence_of_data()
        self._initialize_data()
        self._initialize_parametre_simulation()
        if random_matrix == "":
            self.random_matrix = self.initiat_random_matrix()
        else:
            self.random_matrix = random_matrix
        self.parametre_model = parametre_model
        self._initialize_parametre_model(
            parametre_model
            )
    
    def _initialize_data(self) -> None:
        data_zc = self.get_data_zc()
        if isinstance(data_zc, np.ndarray):
            if len(data_zc.shape) == 3:
                nbr_annee_proj = self.get_year_of_projection()
                nbr_simulation = self.get_number_of_simulation()
                list_df_data_to_flat = [
                    pd.DataFrame(
                        data_zc[:, :, num_simu],
                        columns=[f"{year}" for year in range(1, nbr_annee_proj + 1)]
                    ).assign(simulation=num_simu) for num_simu in range(nbr_simulation)]
                df_transforme = pd.concat(list_df_data_to_flat, axis=0)
                df_transforme["maturite"] = df_transforme.index + 1
                df_transforme.reset_index(inplace=True, drop=True)
                data_zc = df_transforme
            else:
                message_error = f"The dimension of data_zc"\
                    +f" should be 3 but it is {len(data_zc.shape)}"
                logging.error(message_error)
                raise ValueError(message_error)
        if isinstance(data_zc, pd.DataFrame):
            if not("simulation" in data_zc.columns and \
                "maturite" in data_zc.columns):
                message_error = f"The columns of data"\
                    + " should contains 'simulation' and 'maturite'"
                logging.error(message_error)
                raise ValueError(message_error)
        self.data_zc = data_zc
        # Modify the value input if not prices
        self.adapt_data_to_prices_zc() 
    
    def adapt_data_to_prices_zc(self):
        is_prices = self.get_is_data_prices()
        if not is_prices:
            self.data_zc = self.compute_prices_zc(self.data_zc)
        
    def compute_prices_zc(
            self,
            data_rates_zc: pd.DataFrame
            ) -> pd.DataFrame:
        nbr_annee_proj = self.get_year_of_projection()
        df_price_zc = data_rates_zc.copy() 
        for year_proj in [f"{year}" for year in range(1,nbr_annee_proj+1)]:
            df_price_zc[year_proj] = \
                1/(1+data_rates_zc[year_proj])**(data_rates_zc["maturite"])
        return df_price_zc
    
    def compute_rates_zc(
            self,
            data_prices_zc: pd.DataFrame
            ) -> pd.DataFrame:
        nbr_annee_proj = self.get_year_of_projection()
        df_rates_zc = data_prices_zc.copy() 
        for year_proj in [f"{year}" for year in range(1,nbr_annee_proj+1)]:
            df_rates_zc[year_proj] = \
                data_prices_zc[year_proj]**(-1/data_prices_zc["maturite"]) - 1
        return df_rates_zc
    
    def get_parametre_model(
            self,
            param=None,
            note_parametre: str = "AAA"
            ) -> list:
        """
        param: list or dict. Return the value of the parametre if is put in argument 
        if not return the param present in the attribut note_parametre.
        note_parametre: str that indicates with parametre to choose.
        The default value is 'AAA'.

        Return the parametre associated
        """
        if not param:
            parametre_model = \
                self.get_list_from_dict(self.parametre_model[note_parametre])
        else:
            parametre_model = param
        return parametre_model
    
    def _initialize_parametre_model(
            self,
            parametre_model: list | np.ndarray
            ) -> None:
        if parametre_model:
            if isinstance(parametre_model, list) or \
                isinstance(parametre_model, np.ndarray):
               self.parametre_model = self.generate_dict_parametre_model(parametre_model)
            if isinstance(parametre_model, dict):
                self.parametre_model = self.complete_dict_parametre_model(parametre_model)
            
            self.parametre_model = self.complete_dict_parametre_model_with_initial_value()
        else:
            message_warning = "Parametre_model is empty."
            logging.warning(message_warning)
    
    def get_list_from_dict(
            self,
            dict_to_transform: dict
            ) -> list:
        list_of_values = list(dict_to_transform.values())
        return list_of_values
    
    def complete_dict_parametre_model(
            self,
            dict_parametre_model: dict
            ) -> dict:
        """
        Complete a dictionnary from a dictionnary in order to have all note
        """
        list_note = self.get_list_name_rating()
        dict_default_values = \
            self.check_input_dict_parametre_model(dict_parametre_model)
        dict_parametre_to_fill = {}
        for note in list_note:
            if not(note in dict_parametre_model):
                dict_parametre_to_fill[note] = \
                    self.generate_dict_parametre_model(
                        self.get_list_from_dict(dict_default_values)
                    )[note]
            else:
                dict_parametre_to_fill[note] = dict_parametre_model[note]
        return dict_parametre_to_fill
    
    def get_list_name_rating(self):
        return self.get_matrice_transition_reel().columns

    def complete_dict_parametre_model_with_initial_value(self):
        if "initial value" in self.get_parametre_simulation():
            # Complete if the value is defined in the dictionnary of parametre
            initial_value = self.get_parametre_simulation()["initial value"]
            parametre_model = self.parametre_model
            list_name_rating = self.get_list_name_rating()
            for indice_rating, name_rating in enumerate(list_name_rating):
                if not("initial value" in parametre_model[name_rating]) and\
                    isinstance(initial_value, list):
                    parametre_model[name_rating]["initial value"] = initial_value[indice_rating]
                if not("initial value" in parametre_model[name_rating]) and\
                    (isinstance(initial_value, float) or isinstance(initial_value, int)):
                    parametre_model[name_rating]["initial value"] = initial_value
        else:
            # Verify that all parametres mode contains initial value
            parametre_model = self.parametre_model
            list_name_rating = self.get_list_name_rating()
            is_initial_value_not_defined = False
            name_rating_without_init_value = []
            for indice_rating, name_rating in enumerate(list_name_rating):
                if not("initial value" in parametre_model[name_rating]):
                    is_initial_value_not_defined = True
                    name_rating_without_init_value.append(name_rating)
            if is_initial_value_not_defined:
                message_error = "At least one parametre model dont " + \
                    "have an initial value. The rating that dont " +\
                    "have initial value are : " +\
                    f"{name_rating_without_init_value}." +\
                    "You can define it in the parametre simulation argument" + \
                    " with the key word 'initial value'" +\
                    "to be affected for all rating."
                logging.error(message_error)
                raise ValueError(message_error)
        return parametre_model
    
    def check_input_dict_parametre_model(
            self,
            dict_parametre_model_to_check: dict
            ) -> dict:
        """
        Return the dictionnary that contains data form the dict in input
        """
        list_note = self.get_list_name_rating()
        list_name_variable_model = {
            "speed of reversion",
            "long terme mean",
            "instantaneous volatility"
            }
        for note in list_note:
            if note in dict_parametre_model_to_check:
                dict_data_note = dict_parametre_model_to_check[note]
                if list_name_variable_model <= set(dict_data_note.keys()):
                    return dict_data_note
        if (len(list_name_variable_model) == 3 and list_name_variable_model <= \
                set(dict_parametre_model_to_check.keys())) or\
            (len(list_name_variable_model) == 4 and list_name_variable_model >= \
                set(dict_parametre_model_to_check.keys()) and\
                 "initial value" in dict_parametre_model_to_check):
                    return dict_parametre_model_to_check
        message_error = "No dictionnary viable has been found. " +\
            "One dictionnary should contains the keys : " +\
            f"{list_name_variable_model}."
        logging.error(message_error)
        raise ValueError(message_error)

    def generate_dict_parametre_model(
            self,
            list_parametre_model: list
            ) -> dict:
        """
        Create a dictionnary from a list for each note
        """
        list_note = self.get_list_name_rating()
        dict_parametre_model_by_note = {}
        list_parametre_model_copy = list_parametre_model
        dict_parametre_model = self.get_dict_parametre_model(list_parametre_model_copy)
        dict_parametre_model_by_note = {
            note: copy.deepcopy(dict_parametre_model)
            for note in list_note
            }
        return dict_parametre_model_by_note

    def get_dict_parametre_model(
            self,
            list_parametre_model: list | np.ndarray
            ) -> dict:
        if len(list_parametre_model) == 3:
            dict_parametre_model = {
                "speed of reversion": float(list_parametre_model[0]),
                "long terme mean": float(list_parametre_model[1]),
                "instantaneous volatility": float(list_parametre_model[2])
            }
        elif len(list_parametre_model) == 4:
            dict_parametre_model = {
                "speed of reversion": float(list_parametre_model[0]),
                "long terme mean": float(list_parametre_model[1]),
                "instantaneous volatility": float(list_parametre_model[2]),
                "initial value": float(list_parametre_model[3])
            }
        else:
            message_error = f"The length of parametre model is" +\
                 f"{len(list_parametre_model)}. It should be 3 or 4."
            logging.error(message_error)
            raise ValueError(message_error)
        return dict_parametre_model

    def get_is_data_prices(self) -> bool:
        key_is_data_prices = "data_input_is_prices"
        parametre_simulation = self.get_parametre_simulation()
        is_data_prices = True  
        if key_is_data_prices in parametre_simulation:
            is_data_prices = \
                self.get_parametre_simulation()[key_is_data_prices]                             
        return is_data_prices
    
    def initiat_random_matrix(self) -> np.ndarray:
        nbr_simulation = self.get_number_of_simulation()
        frequence = self.get_frequence()
        nbr_annee_projete = self.get_year_of_projection()
        aleas = np.random.normal(
            size=(nbr_simulation, int(nbr_annee_projete/frequence))
            )
        return aleas
    
    def _initialize_parametre_simulation(self) -> None:
        key_variable_output = "variable_output"
        parametre_simulation = self.get_parametre_simulation()
        if key_variable_output in parametre_simulation: 
            variable_output = self.get_variable_output()
            if not(variable_output in ["prices", "rates"]):
                variable_output = "prices"
        else:
            variable_output = "prices"
        parametre_simulation[key_variable_output] = variable_output
        self.parametre_simulation = parametre_simulation
    
    def get_zc_prices(self) -> pd.DataFrame | pd.Series:
        if isinstance(self.data_zc, pd.DataFrame) and\
            "zero coupon prices" in self.data_zc.columns:
            zc_prices = self.data_zc["zero coupon prices"]
        else:
            zc_prices = self.data_zc
        return zc_prices

    def get_frequence(self) -> float | int:
        return self.get_parametre_simulation()["step of time"]
    
    def get_year_of_projection(self) -> int:
        return self.get_parametre_simulation()["year of projection"]

    def get_maturity_maximal(self) -> int:
        return self.get_parametre_simulation()["maturity maximal"]
    
    def get_random_alea(self) -> np.ndarray:
        return self.random_matrix
    
    def get_variable_output(self) -> str:
        key_variable_output = "variable_output"
        parametre_simulation = self.get_parametre_simulation()
        return parametre_simulation[key_variable_output]
    
    def get_taux_recouvrement(self) -> float | int:
        return self.get_parametre_simulation()["taux recouvrement"]
    
    def get_matrice_transition_reel(self) -> np.ndarray:
        return self.get_parametre_simulation()["matrice transition reel"]
    
    def get_number_of_simulation(self) -> int:
        return self.get_parametre_simulation()["number of simulation"]
    
    def get_initial_value(self) -> list:
        list_note = self.get_list_name_rating()
        list_initial_value = []
        for note in list_note:
            list_initial_value.append(
                self.parametre_model[note]["initial value"]
                )
        return list_initial_value
    
    def set_random_matrix(
            self,
            random_matrix_new: np.ndarray
            ) -> None:
        self.random_matrix = random_matrix_new
    
    def fit(self) -> None:
        logging.warning(NotImplemented)
    
    def generate_taux_discretisation(
            self,
            parametre_model
            ) -> np.ndarray:
        """Depreciated method used instead generate_taux_exact"""
        alpha, mu, sigma, initial_value = parametre_model
        dt = self.get_frequence()
        aleas = self.get_random_alea()

        # Nombre de simulations et nombre de périodes
        N, T = aleas.shape

        # Initialisation x0=0, y0=0, r0
        r = np.zeros((N, T+1))
        r[:, 0] = initial_value
        for i in range(1, T+1):
            # Durée de temps entre 0 et t

            determinist_calcul = r[:, i-1] + alpha*(mu - r[:, i-1])*dt
            aleas_calcul = sigma * np.sqrt(dt) * np.sqrt(r[:, i-1]) * aleas[:, i-1] + \
                sigma**2/4 * dt * (aleas[:, i-1]**2 - 1)
            r[:, i] = determinist_calcul + aleas_calcul
        return r
    
    def generate_taux_exact(
            self,
            parametre_model,
            initial_value=0
            ) -> np.ndarray:
        dt = self.get_frequence()
        aleas = self.get_random_alea()        
        if len(parametre_model) == 4 or len(parametre_model) == 0:
            alpha, mu, sigma, initial_value = parametre_model
        if len(parametre_model) == 3:
            alpha, mu, sigma = parametre_model

        c = sigma**2/(4*alpha) * (1 - np.exp(-alpha*dt))
        # Nombre de simulations et nombre de périodes
        if len(aleas.shape) == 2:
            N, T = aleas.shape
        if len(aleas.shape) == 1:
            T = int(aleas.shape[0])
            N = 1
        # Initialisation x0=0, y0=0, r0
        r = np.zeros((N, T+1))
        r[:, 0] = initial_value
        for i in range(1, T+1):
            # Durée de temps entre 0 et t
            theta_t_delta = r[:, i-1] * np.exp(-alpha * dt) / c
            kappa = (4*alpha*mu) / sigma**2
            Xi = np.random.noncentral_chisquare(kappa, theta_t_delta, N)
            r[:, i] = c * Xi
        return r
    
    def calcul_spread_market(
            self,
            price_risque: np.ndarray,
            price_reference: np.ndarray,
            maturite: np.ndarray
            ) -> np.ndarray:
        spread = price_risque**(-1/maturite) - price_reference**(-1/maturite) 
        return spread
    
    def calcul_probabilite_defaut(
            self,
            processus_spread : np.ndarray,
            probabilite: np.ndarray
            ) -> np.ndarray:
        return processus_spread * probabilite

    def calcul_zc_risque(
            self,
            matrice_prix: np.ndarray | pd.DataFrame,
            probabilite_defaut: np.ndarray | pd.DataFrame
            ) -> np.ndarray | pd.DataFrame:
        taux_recouvrement = self.get_taux_recouvrement()
        if probabilite_defaut.shape == matrice_prix.shape:
            prix_risque = matrice_prix * (1 - (1 - taux_recouvrement)*probabilite_defaut)
        else:
            message_error = f"The matrix matrice_prix and probabilite_defaut \
                should have the same dimension.\
                The dimension for matrice_prix is {matrice_prix.shape} and\
                probabilite_defaut is {probabilite_defaut.shape}."
            logging.error(message_error)
            raise ValueError(message_error)
        return prix_risque
    
    def calcul_spred_theorique(
            self,
            matrice_prix: np.ndarray,
            taux_recouvrement: float,
            tenor: int,
            matrice_probabilite_transition: np.ndarray
            ) -> np.ndarray:
        spread = np.zeros(shape=(len(matrice_prix), matrice_probabilite_transition.shape[1]))
        probabilite_defaut = matrice_probabilite_transition[:, -1]
        arrray_maturite = np.arange(1, len(matrice_prix)+1, 1)
        for indice_maturite, each_maturite in enumerate(arrray_maturite):
            risque = 1 - (1-taux_recouvrement)*probabilite_defaut
            facteur_actu = -1/(each_maturite - tenor)
            prix_mat_tenor = matrice_prix[indice_maturite]
            spread[indice_maturite, :] = prix_mat_tenor**facteur_actu * (risque**facteur_actu - 1)
        return spread
    
    def calcul_matrice_generatrice_reel(
            self,
            matrice_transition_reel,
            nombre_developpement=10
            ) -> np.ndarray:
        """
        Calcul the matrice generatrice reel from the matrice reel of transition
        """
        # Utilisation du developpement en série entière du log
        dim = matrice_transition_reel.shape
        matrice_generatrice = np.zeros_like(matrice_transition_reel)
        mat_inter = np.identity(dim[0])
        for nieme_developpement in range(1, nombre_developpement):
            mat_inter = np.dot(mat_inter, np.identity(dim[0]) - matrice_transition_reel)
            matrice_generatrice += mat_inter/nieme_developpement
        #matrice_generatrice = transformation_matrice_generatrice_reel(-matrice_generatrice)
        return -matrice_generatrice
    
    def exponentiel_matriciel(
            self,
            matrice_to_compute: np.ndarray | pd.DataFrame | pd.Series,
            nbr_iter = 10
            ) -> np.ndarray:
        mat_init = np.identity(n=matrice_to_compute.shape[0])
        factoriel = 1
        for iter in range(1, nbr_iter + 1):
            factoriel *= iter
            mat_init += np.dot(mat_init, matrice_to_compute)/factoriel
        return mat_init
    
    def calcul_matrice_transition_t(
            self,
            matrice_transition_hist: np.ndarray,
            spread_for_a_note: np.ndarray,
            dt: float | int
            ) -> np.ndarray:
        try:
            matrice_transition_trans = matrice_transition_hist.copy()
            matrice_transition_trans = matrice_transition_trans.T * spread_for_a_note
            matrice_transition = self.exponentiel_matriciel(matrice_transition_trans.T * dt)
            matrice_transition = np.array(matrice_transition)
        except Exception as e:
            message_info = f"The dim of matrice_transition_hist" +\
                f"is {matrice_transition_hist.shape} and" +\
                f"spread_for_a_note is {spread_for_a_note.shape}." +\
                f"The error is {e}."
            logging.error(message_info)
            raise e(message_info)
        return matrice_transition
    
    def calcul_price_obligation_t(
            self,
            price_zc_credit: pd.DataFrame,
            probabilite_defaut: np.ndarray,
            note: str,
            num_simu: int
            ) -> pd.DataFrame:
        """
        Compute the zero coupon for one the rating of all year of projection.

        Return a dataframe that  contains for all year of projection
        the value of the zero coupon for one simulation.
        """
        taux_recouvrement = self.get_taux_recouvrement()
        maturity_maximal = self.get_maturity_maximal()
        list_maturity =  np.arange(1, maturity_maximal+1)
        arg_fluct_note = taux_recouvrement * probabilite_defaut +\
            (1 - probabilite_defaut)
        df_zc_prices_with_note = price_zc_credit * arg_fluct_note
        df_zc_prices_with_note["Note"] = note
        df_zc_prices_with_note["maturite"] = list_maturity
        df_zc_prices_with_note["simulation"] = num_simu + 1
        
        return df_zc_prices_with_note
    
    def get_adapted_matrice_generatrice_reel(
            self,
            array_matrice_transition: np.ndarray
            ) -> np.ndarray:
        array_matrice_generatrice_transition = \
            self.calcul_matrice_generatrice_reel(array_matrice_transition)
        nbr_annee_projete = self.get_year_of_projection()

        if len(array_matrice_generatrice_transition.shape) == 2:
            matrice_gen_trans_for_period_proj = np.array(
                [array_matrice_generatrice_transition] * nbr_annee_projete)
        if len(array_matrice_generatrice_transition.shape) == 3:
            matrice_gen_trans_for_period_proj = array_matrice_generatrice_transition
        return matrice_gen_trans_for_period_proj
    
    def get_year_projected(self) -> np.ndarray:
        nbr_annee_projete = self.get_year_of_projection()
        dt = self.get_frequence()
        #year_projected = np.arange(int(1/dt), int((nbr_annee_projete+1)/dt), int(1/dt))
        year_projected = np.arange(0, int((nbr_annee_projete)/dt), int(1/dt))
        return year_projected
    
    def calcul_spread_sto(
            self,
            list_valeur_initial_by_note: list | np.ndarray
            ) -> np.ndarray:
        # Simulation des spread au cours du temps (nbr note X nombre_simu X nbr pas de temps)
        list_modelisation_spread = []
        nbr_simulation = self.get_number_of_simulation()
        nbr_credit_note = self.get_matrice_transition_reel().shape
        list_name_note = self.get_list_name_rating()
        for indice_note in range(nbr_credit_note[0]):
            init_value = list_valeur_initial_by_note[indice_note]
            note_parametre = list_name_note[indice_note]
            param_model = \
                self.get_parametre_model(note_parametre=note_parametre)
            # il est possible de faire évoluer param
            spread_t = self.generate_taux_exact(
                parametre_model=param_model,
                initial_value=init_value
                ) 
            list_modelisation_spread.append(spread_t)

        modelisation_spread = np.array(list_modelisation_spread)        
        year_projected = self.get_year_projected()
        modelisation_spread_year = modelisation_spread[:, :, year_projected]
        return modelisation_spread_year
    
    def calcul_matrice_transition(self) -> np.ndarray:
        """
        Dimension of Return : 
        Year of Projection X Nbr_Simulation X Note de crédit (AAA - Default)
        """
        array_matrice_transition = self.get_matrice_transition_reel()
        nbr_simulation = self.get_number_of_simulation()
        nbr_annee_projete = self.get_year_of_projection()
        dt = self.get_frequence()
        valeur_initial = self.get_initial_value()

        matrice_gen_transition = \
            self.get_adapted_matrice_generatrice_reel(array_matrice_transition)
        modelisation_spread_year = self.calcul_spread_sto(valeur_initial)
        table_simulation_mat_transition = []
        for num_simu in range(nbr_simulation):
            table_rating_projected = []
            for year_proj in range(nbr_annee_projete):
                model_spread_year = modelisation_spread_year[:, num_simu, year_proj]
                array_matrice_generatrice_transition = matrice_gen_transition[year_proj]
                matrice_transition = self.calcul_matrice_transition_t(
                        array_matrice_generatrice_transition,
                        model_spread_year,
                        dt)
                table_rating_projected.append(matrice_transition[:-1, -1]) # Exclude Dafault
            table_simulation_mat_transition.append(table_rating_projected)
        table_simulation_mat_transition = np.array(table_simulation_mat_transition)
        return table_simulation_mat_transition

    def calcul_matrice_price_zero_coupon_by_note(
            self,
            list_matrice_transition: np.ndarray
            ) -> pd.DataFrame:
        nbr_annee_projection = self.get_year_of_projection()
        nbr_simulation = self.get_number_of_simulation()
        df_price_zc = self.get_data_zc()
        list_name_year_projected = [f"{year}" for year in range(1,nbr_annee_projection+1)]
        list_col_rating = self.get_list_name_rating()[:-1]
        # Exclud Default
        list_data_note_simu = []
        #Rajoute la variation des simulations
        for num_simu in range(nbr_simulation):
            df_price_zc_simulation = \
                df_price_zc[df_price_zc["simulation"] == num_simu][list_name_year_projected]
            for name_rating, prob_default_for_notes in\
                zip(list_col_rating ,list_matrice_transition[num_simu, :, :].T):
                # calcul_price_obligation_t
                df_note_simu = self.calcul_price_obligation_t(
                                price_zc_credit = df_price_zc_simulation,
                                probabilite_defaut = prob_default_for_notes,
                                note = name_rating,
                                num_simu = num_simu
                                )
                list_data_note_simu.append(df_note_simu)
        df_data_note = pd.concat(list_data_note_simu, axis=0)
        df_data_note.reset_index(inplace=True, drop=True)
        return df_data_note

    def predict(self, *args, **kargs) -> pd.DataFrame:
        
        list_matrice_transition = self.calcul_matrice_transition()
        df_zero_coupon = self.calcul_matrice_price_zero_coupon_by_note(
            list_matrice_transition
            )
        variable_output = self.get_variable_output()
        if "rates" == variable_output:
            df_zero_coupon = self.compute_rates_zc(df_zero_coupon)
               
        return df_zero_coupon


if "__main__" == __name__:
    
    #Exemple d'utilisation
    path = os.getcwd()
    initial_value_rates_all = 3.1/100
    taux_recouvrement = 30/100
    number_of_simulation = 500
    step_of_time = 1/12 # Not required in dict_parametre_simulation
    freq_of_data = "monthly" #frequence of data
    year_of_projection = 8
    max_maturity = 20 # rates only

    #La matrice de transition réeel doit être une matrice de transition de chaîne de Markov régulière
    matrice_transition_reel = pd.read_excel(
        os.path.join(path, "monde_reel", "matrice_transition.xlsx")
        ) 
    matrice_transition_reel.set_index("note", inplace=True) # The order must be conserved
    # AAA, AA, A, BBB, BB, B, CCC, Default
    dict_parametre_simulation_credit = {
        "initial value": initial_value_rates_all, # int Optional, indicates the default value for all rating
        "number of simulation": number_of_simulation * 2, # int, that contains thenomber of simulation
        "frequence of data": freq_of_data, # str, indique the frequence of data for calibration et projection
        "maturity maximal": max_maturity, # int, indicate the number of maturity maximal
        "year of projection": year_of_projection +1, # int, indicate the number of year projected
        "taux recouvrement": taux_recouvrement, # float, indicate the recovery rate
        "matrice transition reel": matrice_transition_reel, # Dataframe, that contains the matrice of transition
        "variable_output": "rates", # str, indicate if the output is a rates or a prices
        "data_input_is_prices": True # bool, indicates if the input is a price or a rate
    }

    #Génère un modèle de taux
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

    # Parametre du modèle de crédit (différentes possibilités)
    param_model_credit = [0.1009, 9.5146, 0.7519]
    param_model_credit = {
        'AAA': {
            'speed of reversion': 0.1009,
            'long terme mean': 9.5146,
            'instantaneous volatility': 0.7519
            },
        'CCC': {
            'speed of reversion': 1.1009,
            'long terme mean': 20,
            'instantaneous volatility': 2,
            'initial value': 10 # valeur appliquée si
            #pas de valeur par défaut présent dans parametre simulation
            }
        }
    param_model_credit = {
        'speed of reversion': 0.1009,
        'long terme mean': 9.5146,
        'instantaneous volatility': 0.7519
        }
    model_credit = G2(
        # dictionnaire contenant les informations de lancement
        dict_parametre_simulation_credit,
        # prix zero coupon qui sont décliné pour obtenir des prix zc par note
        data_zc = data_model_prices,
        # list ou dict contenant les parametre utilisés pour les projections
        parametre_model = param_model_credit
        )

    model_credit.parametre_model
    model_credit.fit()
    model_credit.predict()

