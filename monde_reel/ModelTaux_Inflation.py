from monde_reel.ModelTaux_RR import ModelTaux_RR
from dateutil.relativedelta import relativedelta
import copy
import pandas as pd
import numpy as np
import logging
import os

class ModelInflation_RR:

    def __init__(
            self,
            rate_model_nominaux: ModelTaux_RR,
            parametre_simulation: dict,
            path_data_input: str = "",
            data: dict = {},
            rate_model_reel: ModelTaux_RR = None
            ) -> None:
        self.rate_model_nominaux = rate_model_nominaux
        self.rate_model_reel = rate_model_reel
        self.parametre_simulation = parametre_simulation
        self.path_data_input = path_data_input
        self.data = data

        self.verify_input()
        self._initiliaze_frequence_of_data()
        self.import_data()
        self.initialize_model_rate()
        self.filtre_data()
    
    def get_rate_model_nominal(self) -> ModelTaux_RR:
        return self.rate_model_nominaux
    
    def get_rate_model_reel(self) -> ModelTaux_RR:
        return self.rate_model_reel

    def set_rate_model_reel(
            self,
            rate_model_reel: ModelTaux_RR
            ) -> None:
        self.rate_model_reel = rate_model_reel
    
    def get_parametre_simulation(self) -> dict:
        return self.parametre_simulation

    def get_initial_value(self) -> float:
        parametre_simu = self.get_parametre_simulation()
        if 'initial value' in parametre_simu:
            return parametre_simu['initial value']
        else:
            message_error = "There is no 'initial value' in the parametre."
            logging.error(message_error)
            raise ValueError(message_error)
    
    def set_initial_value(
            self,
            initial_value: float
            ) -> None:
        parametre_simu = self.get_parametre_simulation()
        if isinstance(parametre_simu, dict):
            parametre_simu['initial value'] = initial_value
            self.parametre_simulation = parametre_simu
        else:
            message_error = "The parametre are not a dictionnary."\
                + " Initialisation impossible."
            logging.error(message_error)
            raise ValueError(message_error)
    
    def get_path_data_input(self) -> str:
        return self.path_data_input
    
    def get_data(self) -> np.ndarray | pd.Series:
        return self.data
    
    def set_data(
            self,
            data_to_add: np.ndarray | pd.Series
            ) -> np.ndarray | pd.Series:
        self.data = data_to_add
    
    def get_step_of_time(self) -> float:
        if "step of time" in self.get_parametre_simulation():
            step_of_time = self.get_parametre_simulation()["step of time"] 
        else:
            message_error = f"The parametre step_of_time is not present in\
                parametre_simulation."
            logging.error(message_error)
            raise ValueError(message_error)
        return step_of_time
    
    def get_year_of_projection(self) -> int:
        if "year of projection" in self.get_parametre_simulation():
            year_of_projection = self.get_parametre_simulation(
                )["year of projection"] 
        else:
            message_error = f"The parametre step_of_time is not present in\
                parametre_simulation."
            logging.error(message_error)
            raise ValueError(message_error)
        return year_of_projection
    
    def get_number_of_simulation(self) -> int:
        if "number of simulation" in self.get_parametre_simulation():
            number_of_simulation = self.get_parametre_simulation(
                )["number of simulation"] 
        else:
            message_error = f"The parametre step_of_time is not present in\
                parametre_simulation."
            logging.error(message_error)
            raise ValueError(message_error)
        return number_of_simulation
    
    def _initiliaze_frequence_of_data(self) -> None:
        period_map = {
            "annual": 1.0,
            "half-yearly": 0.5,
            "quarterly": 0.25, 
            "monthly": 1/12,
            "daily": 1/360
            }
        if "frequence of data" in self.parametre_simulation and \
                self.parametre_simulation["frequence of data"] \
                    in period_map.keys():
            self.parametre_simulation["step of time"] = \
                period_map[self.parametre_simulation["frequence of data"]]
        else:
            message_error = f"There is not such period \
                {self.parametre_simulation['frequence of data']} in \
                {period_map.keys()}."
            logging.error(message_error)
            raise ValueError(message_error)
    
    def check_verify_data_input(self) -> None:
        if self.get_path_data_input() != "" and \
            self.get_data() != {}:
            message_error = f"Error in input No data has been detected"
            logging.error(message_error)
            raise ValueError(message_error)
    
    def check_verify_parametre_simulation(self) -> None:
        if not isinstance(self.get_parametre_simulation(), dict):
            message_error = f"The type of parametre_simulation have to be\
                a dict be it is : {type(self.get_parametre_simulation())}."
            logging.error(message_error)
            raise TypeError(message_error)
    
    def mapping_funct_check(self) -> dict:
        dict_funct_verify = {
            "data_input": self.check_verify_data_input,
            "parametre_simulation": self.check_verify_parametre_simulation
        }
        return dict_funct_verify
    
    def verify_input(self) -> None:
        dict_funct_verify = self.mapping_funct_check()
        for index_each_check, check_input in dict_funct_verify.items():
            check_input()
    
    def copy_methode_rate(self) -> ModelTaux_RR:
        method_rate_copy = self.rate_model_reel
        return method_rate_copy
    
    def adapt_data_input(
            self,
            data_1: np.ndarray | pd.Series,
            data_2: np.ndarray | pd.Series
            ) -> np.ndarray | pd.Series:
        length_data_1 = len(data_1)
        length_data_2 = len(data_2)
        if length_data_1 < length_data_2:
            data_2 = data_2[:length_data_1]
        if length_data_2 < length_data_1:
            data_1 = data_1[:length_data_2]
        if len(data_1) == len(data_2):

            return data_1, data_2
        else:
            message_error = f"The length of data nominal or inflation\
                do not correspond."
            logging.error(message_error)
            raise ValueError(message_error)
    
    def get_delta_time(self):
        freq_of_data = self.parametre_simulation["frequence of data"]
        if freq_of_data == "daily":
            delta_time = relativedelta(days=1)
        elif freq_of_data == "monthly":
            delta_time = relativedelta(months=1)
        elif freq_of_data == "annual":
            delta_time = relativedelta(years=1)
        else:
            message_error = f"There is not such period {freq_of_data}. \
                Only daily, monthly and annual are allowed"
            logging.error(message_error)
            raise ValueError(message_error)
        return delta_time
    
    def import_data(self) -> None:
        try:
            if self.path_data_input != "":
                path_data_completed = os.path.join(
                    os.getcwd(),
                    self.path_data_input
                    )
                self.data = pd.read_excel(path_data_completed)
        except ValueError as e: 
            logging.error(e)
            raise e
    
    def change_data_annee_to_date(
            self,
            data_to_modify: pd.DataFrame
            ) -> pd.DataFrame:
        data_to_modify["Date"] = data_to_modify["Annee"].astype(str)
        data_to_modify["Date"] = pd.to_datetime(
            data_to_modify["Date"],
            format="%Y")
        return data_to_modify
    
    def filtre_data(self) -> None:
        data_to_filtre = self.get_data().copy()
        if isinstance(data_to_filtre, pd.DataFrame):
            if "Annee" in data_to_filtre.columns:
                data_to_filtre = self.change_data_annee_to_date(
                    data_to_modify = data_to_filtre)
                if "studied period start" in self.parametre_simulation:
                    min_date = \
                        pd.to_datetime(self.parametre_simulation["studied period start"])
                    delta_date = self.get_delta_time()
                    data_to_filtre = data_to_filtre[
                        data_to_filtre["Date"] >= min_date - delta_date]
                if "finished period start" in self.parametre_simulation:
                    max_date = \
                        pd.to_datetime(self.parametre_simulation["finished period start"])
                    data_to_filtre = data_to_filtre[
                        data_to_filtre["Date"] <= max_date]
        if isinstance(data_to_filtre, pd.DataFrame):
            self.set_data(data_to_filtre["Inflation"])
        elif isinstance(data_to_filtre, pd.Series):
            self.set_data(data_to_filtre)
        elif isinstance(data_to_filtre, np.ndarray):
            self.set_data(data_to_filtre)
        else:
            message_error = f"The type is incorrect,\
                it is {type(data_to_filtre)}.\
                It should be Dataframe, Series, ndarray."
            logging.error(message_error)
            raise TypeError(message_error)
    
    def adapt_data_Dataframe_to_Series(
            self,
            data_to_modify: pd.Series or pd.DataFrame
            ) -> pd.Series:
        if isinstance(data_to_modify, pd.Series):
            return data_to_modify
        if isinstance(data_to_modify, pd.DataFrame):
            data_to_modify = data_to_modify[
                data_to_modify.columns[-1]
                ]
            return data_to_modify
        else:
            message_error = f"Wrong type of data :" +\
            f"{type(data_to_modify)}. It sould be Series or Dataframe."
            logging.error(message_error)
            raise TypeError(message_error)
        
    def initialize_model_rate(self) -> None:
        method_rate_reel = self.copy_methode_rate()
        inflation_initial = self.get_initial_value()
        nominal_initial = method_rate_reel.get_initial_value() 
        method_rate_reel.set_initial_value(
            initial_value = nominal_initial - inflation_initial)
        
        data_rates_nominal = self.get_rate_model_nominal().get_data()
        data_inflation = self.get_data()

        data_rates_nominal = self.adapt_data_Dataframe_to_Series(
            data_to_modify = data_rates_nominal 
        )
        data_inflation = self.adapt_data_Dataframe_to_Series(
            data_to_modify = data_inflation
        )

        data_rates_nominal, data_inflation = self.adapt_data_input(
            data_rates_nominal, data_inflation
        )
        self.set_rate_model_reel(method_rate_reel)
        self.get_rate_model_reel().set_data(data_rates_nominal - data_inflation)
        self.get_rate_model_nominal().set_data(data_rates_nominal)
        try:
            matrix_random_to_add = \
                self.get_rate_model_reel().set_random_normal_value_to_matrix(
                    matrix_reference=self.get_rate_model_nominal().random_matrix
                )
            self.get_rate_model_reel().set_random_matrix(matrix_random_to_add)
        except Exception as e:
            logging.warning(e)
    
    def fit(self) -> None:
        self.get_rate_model_reel().fit()
        self.get_rate_model_nominal().fit()
    
    def compute_inflation(
            self, 
            short_rate_nominal: np.ndarray,
            short_rate_reel: np.ndarray,
            ) -> np.ndarray:
        year_of_projection = self.get_year_of_projection()
        step_of_time = self.get_step_of_time()
        number_of_simulation = self.get_number_of_simulation()
        number_of_interval = int(year_of_projection / step_of_time)
        inflation_matrix = np.zeros(
            shape=(number_of_interval+1, number_of_simulation)
            )
        inflation_matrix[0, :] = self.get_initial_value()
        for year_projected in range(1, number_of_interval + 1):
            arg_exp_nom = short_rate_nominal[year_projected - 1, :] 
            arg_exp_reel = short_rate_reel[year_projected - 1, :] + \
                short_rate_reel[year_projected, :]
            inflation_for_year_projected = np.exp(arg_exp_nom - arg_exp_reel/2)
            inflation_matrix[year_projected, :] = \
                inflation_for_year_projected - 1
        return inflation_matrix
    
    def adapt_output_to_have_projection_annualy(
            self,
            matrice_to_adapt: np.ndarray
            ) -> np.ndarray:
        year_of_projection = self.get_year_of_projection()
        step_of_time = self.get_step_of_time()
        number_of_interval = int(year_of_projection / step_of_time)
        maturity_to_take_account = np.arange(0,
                                             int(number_of_interval)+1,
                                             int(1/step_of_time)
                                             )
        matrice_adapted = matrice_to_adapt[maturity_to_take_account, :]
        return matrice_adapted
    
    def predict(self) -> np.ndarray:
        short_rate_nominal = self.get_rate_model_nominal().predict("short_rates")
        short_rate_reel = self.get_rate_model_reel().predict("short_rates")

        inflation_matrix = self.compute_inflation(
            short_rate_nominal,
            short_rate_reel)
        inflation_matrix_by_year = self.adapt_output_to_have_projection_annualy(
            inflation_matrix
        )
        return inflation_matrix_by_year




