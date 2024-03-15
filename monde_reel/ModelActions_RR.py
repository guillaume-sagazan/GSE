import pandas as pd
import numpy as np
import os
import logging
from scipy import optimize
from scipy.interpolate import splrep, splev, BSpline
from typing import Union, List, Optional, Any
from abc import ABC, abstractmethod
from dateutil.relativedelta import relativedelta

class ModelActions_RR(ABC):

    def __init__(
            self,
            parametre_simulation : dict,
            path_data: str = "",
            data: pd.DataFrame | np.ndarray = [],
            ) -> None:
        self.parametre_simulation = parametre_simulation
        self.path_data = path_data
        self.data = data
    
    def get_parametre_simulation(self):
        return self.parametre_simulation
    
    def get_path_data(self):
        return self.path_data
    
    def set_random_matrix(
            self,
            random_matrix_new: np.ndarray
            ) -> None:
        self.random_matrix = random_matrix_new
    
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
            logging.error(f"There is not such period {self.period}")
            raise ValueError(f"There is not such period {self.period}")
    
    def import_data(self):
        try:
            if self.path_data != "":
                path_data_completed = os.path.join(
                    os.getcwd(),
                    self.path_data
                    )
                data_imported = pd.read_excel(path_data_completed)
            else:
                data_imported = self.data
        except ValueError as e: 
            logging.error(e)
            raise e
        return data_imported
    
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

    @abstractmethod
    def fit(self) -> None:
        pass
    
    @abstractmethod
    def predict(self):
        pass

class Black_Scholes_RR(ModelActions_RR):

    def __init__(
            self,
            parametre_simulation: dict,
            random_matrix: np.ndarray,
            path_data: str = "",
            data: pd.DataFrame | np.ndarray = [],
            parametre_model: dict | list = {},
            ) -> None:
        self._initialize_parametre_model(parametre_model)
        super().__init__(parametre_simulation, path_data, data)
        self._initiliaze_frequence_of_data()
        self.random_matrix = random_matrix
    
    def get_parametre_model(self):
        return self.parametre_model

    def get_initial_value(self):
        if "initial value" in self.get_parametre_simulation():
            init_value = self.get_parametre_simulation()["initial value"]
        else:
            init_value = 100 
        return init_value
    
    def get_number_of_simulation(self):
        return self.get_parametre_simulation()["number of simulation"]
    
    def get_frequence_of_data(self) -> str:
        return self.get_parametre_simulation()["frequence of data"]
    
    def get_step_of_time(self) -> float:
        return self.get_parametre_simulation()["step of time"]
    
    def get_year_of_projection(self):
        return self.get_parametre_simulation()["year of projection"]
    
    def get_parametre_model_risk_free_rate(self):
        parametre_model = self.get_parametre_model()
        if isinstance(parametre_model, list) |\
            isinstance(parametre_model, np.ndarray):
            risk_free_rate = parametre_model[0]
        elif isinstance(parametre_model, dict):
            risk_free_rate = parametre_model["risk-free rate"]
        else:
            message_error = f"The type of parametre model does not correpond.\
                It should be:\
                    list, np.ndarray or dict but it is {type(parametre_model)}."
            logging.error(message_error)
            raise ValueError(message_error)
        return risk_free_rate
    
    def get_parametre_model_volatility(self):
        parametre_model = self.get_parametre_model()
        if isinstance(parametre_model, list) |\
            isinstance(parametre_model, np.ndarray):
            risk_free_rate = parametre_model[1]
        elif isinstance(parametre_model, dict):
            risk_free_rate = parametre_model["volatility"]
        else:
            message_error = f"The type of parametre model does not correpond.\
                It should be:\
                    list, np.ndarray or dict but it is {type(parametre_model)}."
            logging.error(message_error)
            raise ValueError(message_error)
        return risk_free_rate

    def get_dict_parametre_model_initialisation(
            self
            ) -> dict:
        dict_parametre_model = {
            "risk-free rate": float(np.random.uniform(0, 1)),
            "volatility": float(np.random.uniform(0, 1))
        }
        return dict_parametre_model
    
    def get_dict_parametre_model(
            self,
            list_parametre_model: list | np.ndarray 
            ) -> dict:
        if len(list_parametre_model) == 1:
            risk_free = float(list_parametre_model[0])
            volatility = 0
        if len(list_parametre_model) == 2:
            risk_free = float(list_parametre_model[0])
            volatility = float(list_parametre_model[1])

        dict_parametre_model = {
            "risk-free rate": risk_free,
            "volatility": volatility
        }
        return dict_parametre_model

    def _initialize_parametre_model(
            self,
            parametre_model: dict | list
            ):
        if not parametre_model and isinstance(parametre_model, dict):
            self.parametre_model = self.get_dict_parametre_model_initialisation()
        elif isinstance(parametre_model, dict):
            self.parametre_model = parametre_model
        elif isinstance(parametre_model, list):
            self.parametre_model = self.get_dict_parametre_model(parametre_model)
        else:
            message_error = f"There is not such type of parametre_model allowed \
                            {type(parametre_model)}. Only list and dict autorize"
            logging.error(message_error)
            raise ValueError(message_error)
    
    def compute_parametre_model(
            self,
            data: np.ndarray | pd.DataFrame
            ) -> dict:
        risk_free_rate = self.get_parametre_model_risk_free_rate()
        volatility = np.std(data)
        dict_parametre = self.get_dict_parametre_model(
            [risk_free_rate, volatility])
        return dict_parametre
    
    def filter_data(
            self,
            stock_prices: pd.DataFrame | pd.Series
            ) -> np.ndarray | pd.Series:
        """Should take element+1 for allowing the yield calculus"""
        data = stock_prices.copy()

        if "studied period start" in self.parametre_simulation and \
            "date" in data.columns:
            min_date = pd.to_datetime(self.parametre_simulation["studied period start"])
            delta_date = self.get_delta_time()
            data = data[data["date"] >= min_date - delta_date]
        if "finished period start" in self.parametre_simulation and \
            "date" in data.columns:
            max_date = pd.to_datetime(self.parametre_simulation["finished period start"])
            data = data[data["date"] <= max_date]

        data_stock = data["Action"]
        return data_stock
    
    def get_log_yield(
            self,
            list_stock_prices: list | np.ndarray | pd.Series
            ) -> np.ndarray | pd.Series:
        list_stock_prices_shifted = np.array(list_stock_prices[1:])
        list_stock_prices_adapted = np.array(list_stock_prices[:-1])
        list_log_yield = np.log(list_stock_prices_shifted/list_stock_prices_adapted)
        return list_log_yield
    
    def fit(self):
        try:
            data_stock_prices = self.import_data()
            data_stock_prices_filter = self.filter_data(data_stock_prices)
            data_log_yield = self.get_log_yield(data_stock_prices_filter)
            parametre_model = self.compute_parametre_model(data_log_yield)
            self.parametre_model = parametre_model
        except Exception as e:
            logging.error(e)
            raise e
    
    def get_simulation_stock_prices(
            self,
            time_of_projection: float | int,
            aleas: np.ndarray
            ) -> np.ndarray:
        risk_free_rate = self.get_parametre_model_risk_free_rate()
        volatility = self.get_parametre_model_volatility()
        initial_value = self.get_initial_value()

        component_cst = risk_free_rate - volatility**2/2
        arg_exp = component_cst*time_of_projection + volatility*aleas
        stock_t = initial_value * np.exp(arg_exp)
        return stock_t
    
    def simulations_stock_prices(self):
        step_of_time = self.get_step_of_time()
        year_of_projections = self.get_year_of_projection()
        number_of_simulation = self.get_number_of_simulation()
        number_of_interval = int(year_of_projections / step_of_time)
        initial_value = self.get_initial_value()

        matrix_stock_prices = np.ones( (number_of_interval+1, number_of_simulation) )
        matrix_stock_prices = matrix_stock_prices * initial_value
        for maturity in range(1, number_of_interval+1):
            matrix_stock_prices[maturity, :] = self.get_simulation_stock_prices(
                time_of_projection=maturity * step_of_time,
                aleas=self.random_matrix[maturity - 1, :]
            )
        return matrix_stock_prices
    
    def predict(self, *args, **kwargs):
        try:
            freq_of_time = int(1/self.get_step_of_time())
            year_of_projection = self.get_year_of_projection()
            matrix_stock_prices = self.simulations_stock_prices()
            matrix_stock_prices = np.transpose(
                np.array(
                matrix_stock_prices[
                range(0, year_of_projection*freq_of_time+1, freq_of_time)]
                )
            )
        except Exception as e:
            logging.error(e)
            raise e
        return matrix_stock_prices

class Merton_RR(ModelActions_RR):

    def __init__(
            self,
            parametre_simulation: dict,
            random_matrix: np.ndarray,
            path_data: str = "",
            data: pd.DataFrame | np.ndarray = [],
            parametre_model: dict | list = {},
            ) -> None:
        self._initialize_parametre_model(parametre_model)
        super().__init__(parametre_simulation, path_data, data)
        self._initiliaze_frequence_of_data()
        self.random_matrix = random_matrix

    def _initialize_parametre_model(
            self,
            parametre_model: dict | list
            ):
        if not parametre_model and isinstance(parametre_model, dict):
            self.parametre_model = self.get_dict_parametre_model_initialisation()
        elif isinstance(parametre_model, dict):
            self.parametre_model = parametre_model
        elif isinstance(parametre_model, list):
            self.parametre_model = self.get_dict_parametre_model(parametre_model)
        else:
            message_error = f"There is not such type of parametre_model allowed \
                            {type(parametre_model)}. Only list and dict autorize"
            logging.error(message_error)
            raise ValueError(message_error)
    
    def filter_data(
            self,
            stock_prices: pd.DataFrame | pd.Series
            ) -> np.ndarray | pd.Series:
        """Should take element+1 for allowing the yield calculus"""
        data = stock_prices.copy()

        if "studied period start" in self.parametre_simulation and \
            "date" in data.columns:
            min_date = pd.to_datetime(self.parametre_simulation["studied period start"])
            delta_date = self.get_delta_time()
            data = data[data["date"] >= min_date - delta_date]
        if "finished period start" in self.parametre_simulation and \
            "date" in data.columns:
            max_date = pd.to_datetime(self.parametre_simulation["finished period start"])
            data = data[data["date"] <= max_date]

        data_stock = data["Action"]
        return data_stock
    
    def get_log_yield(
            self,
            list_stock_prices: list | np.ndarray | pd.Series
            ) -> np.ndarray | pd.Series:
        list_stock_prices_shifted = np.array(list_stock_prices[1:])
        list_stock_prices_adapted = np.array(list_stock_prices[:-1])
        list_log_yield = np.log(list_stock_prices_shifted/list_stock_prices_adapted)
        return list_log_yield
    
    def calcul_moment_of_order_k(
            self,
            list_log_yield: np.ndarray | pd.Series,
            mean_log_yield: float,
            order_of_moment: int
            ) -> np.ndarray | pd.Series:
        moment_of_order_k = np.power(list_log_yield - mean_log_yield,
                                     order_of_moment)
        return moment_of_order_k
    
    def get_list_of_empiric_moment(
            self,
            list_log_yield: np.ndarray | pd.Series
            ) -> np.ndarray:
        mean_log_yield = np.mean(list_log_yield)
        var_log_yield = np.var(list_log_yield)
        list_moment_of_order_3 = self.calcul_moment_of_order_k(
            list_log_yield, mean_log_yield,3)
        list_moment_of_order_4 = self.calcul_moment_of_order_k(
            list_log_yield, mean_log_yield, 4)
        list_moment_of_order_5 = self.calcul_moment_of_order_k(
            list_log_yield, mean_log_yield, 5)
        list_moment_of_order_6 = self.calcul_moment_of_order_k(
            list_log_yield, mean_log_yield, 6)
        
        moment_of_order_3 = np.mean(list_moment_of_order_3)
        moment_of_order_4 = np.mean(list_moment_of_order_4) - 3 * var_log_yield**2
        moment_of_order_5 = np.mean(list_moment_of_order_5) - \
            10 * var_log_yield * moment_of_order_3
        moment_of_order_6 = np.mean(list_moment_of_order_6) - \
            15 * var_log_yield * moment_of_order_4 - \
            10 * moment_of_order_3**2 - \
            15 * var_log_yield**3
        list_of_empiric_moment = np.array([mean_log_yield, var_log_yield,
                        moment_of_order_3, moment_of_order_4, moment_of_order_5,
                        moment_of_order_6])
        return list_of_empiric_moment

    def get_list_of_theoretical_moment(
            self,
            parametre_model: list,
            h: float = 1
            ) -> list:
        alpha = parametre_model[0]
        lambda_a = parametre_model[1]
        gamma_a = parametre_model[2]
        omega_a = parametre_model[3]
        sigma2_a = parametre_model[4]
        mu_a = parametre_model[5]
        coeff_k = np.exp(gamma_a + omega_a**2 / 2) - 1

        theoritical_moment_1 = (alpha - lambda_a * (coeff_k - gamma_a)) *  h 
        theoritical_moment_2 = h * (sigma2_a + lambda_a * (gamma_a**2 + omega_a**2))
        theoritical_moment_3 = lambda_a * h * (3 * gamma_a * omega_a**2 + gamma_a**3)
        theoritical_moment_4 = lambda_a * h * (3 * omega_a**4 + \
                                               6*gamma_a**2*omega_a**2 + gamma_a**2)
        theoritical_moment_5 = lambda_a * h * (15 * gamma_a * omega_a**4 + \
                                               10 * gamma_a**3 * omega_a**2 + gamma_a**5)
        theoritical_moment_6 = lambda_a * h * (15 * omega_a**6 + \
                                               45 * omega_a**4*gamma_a**2 + \
                                               15 * omega_a**2*gamma_a**4 + gamma_a**6)
        list_theoritical_moment = np.array(
            [theoritical_moment_1, theoritical_moment_2, theoritical_moment_3,
             theoritical_moment_4, theoritical_moment_5, theoritical_moment_6]
        )
        return list_theoritical_moment
    
    def calcul_objectif_function(
            self,
            parametre: np.ndarray,
            list_of_empiric_moment: np.ndarray
            ) -> float:
        list_of_theoritical_moment = self.get_list_of_theoretical_moment(
            parametre,
            h=1
            )
        objectif_cost = np.sum(
            np.abs(
                list_of_empiric_moment - list_of_theoritical_moment
            )
        )
        return objectif_cost
    
    def get_dict_parametre_model_initialisation(
            self
            ) -> dict:
        dict_parametre_model = {
            "alpha": float(np.random.uniform(0, 1)),
            "intensity of jump": float(np.random.uniform(0, 1)),
            "mean of jump": float(np.random.uniform(0, 1)),
            "standar deviation of jump": float(np.random.uniform(0, 1)),
            "volatility stock": float(np.random.uniform(0, 1)),
            "mean stock": float(np.random.uniform(0, 1))
        }
        return dict_parametre_model
    
    def get_dict_parametre_model(
            self,
            list_parametre_model: list | np.ndarray 
            ) -> dict:
        dict_parametre_model = {
            "alpha": float(list_parametre_model[0]),
            "intensity of jump": float(list_parametre_model[1]),
            "mean of jump": float(list_parametre_model[2]),
            "standar deviation of jump": float(list_parametre_model[3]),
            "volatility stock": float(list_parametre_model[4]),
            "mean stock": float(list_parametre_model[5])
        }
        return dict_parametre_model
    
    def get_list_parametre_from_dict(self)-> list:
        list_parametre_model = []
        if isinstance(self.parametre_model, dict):
            for key, value in self.parametre_model.items():
                list_parametre_model.append(value)
        if isinstance(self.parametre_model, list):
            list_parametre_model = self.parametre_model
        return list_parametre_model

    def get_optimization(
            self,
            function_to_optimize,
            parametre_initial: list,
            ) -> list:
        # Définir les contraintes
        constrain_ineq_intensity_jump = {
            'type': 'ineq',
            'fun': lambda x: x[1] 
        }
        constrain_ineq_mean_jump = {
            'type': 'ineq',
            'fun': lambda x: x[2]
        }
        constrain_ineq_std_jump = {
            'type': 'ineq',
            'fun': lambda x: x[3] 
        }
        constrain_ineq_volatility_stock = {
            'type': 'ineq',
            'fun': lambda x: x[4]
        }
        constrain_ineq_mean_stock = {
            'type': 'ineq',
            'fun': lambda x: x[5]
        }
        list_constrains = [
            constrain_ineq_intensity_jump, constrain_ineq_mean_jump, constrain_ineq_std_jump,
            constrain_ineq_volatility_stock, constrain_ineq_mean_stock
                ]
        # To increase the stabilization of the result
        parametre_optimal = optimize.minimize(
            function_to_optimize,
            parametre_initial,
            method='SLSQP',
            constraints=list_constrains
            ).x
        parametre_optimal = optimize.minimize(
            function_to_optimize,
            parametre_optimal,
            method='SLSQP',
            constraints=list_constrains
            ).x
        return parametre_optimal
    
    def compute_parametre_model(
            self,
            list_stock_prices: np.ndarray | pd.Series
            ) -> dict:
        try:
            data_filter = self.filter_data(list_stock_prices)
            list_log_yield = self.get_log_yield(data_filter)
            list_of_empiric_moment = self.get_list_of_empiric_moment(list_log_yield)
            f_optimize = lambda parametre: self.calcul_objectif_function(
                parametre,
                list_of_empiric_moment
            )
            parametre_initial = self.get_list_parametre_from_dict()
            parametre_optimal = self.get_optimization(f_optimize, parametre_initial)
            parametre_optimal[-1] = parametre_optimal[0] + parametre_optimal[-2]/2
            parametre_model = self.get_dict_parametre_model(parametre_optimal)
        except ValueError as e:
            logging.error(e)
            raise e
        return parametre_model
    
    def fit(self) -> None:
        try:
            data_stock_prices = self.import_data()
            parametre_model = self.compute_parametre_model(data_stock_prices)
            self.parametre_model = parametre_model
        except Exception as e:
            logging.error(e)
            raise e
                 
    def get_processus_jump(self, number_of_interval):
        lambda_a = self.parametre_model["intensity of jump"]

        valeurs_poisson = np.random.poisson(lambda_a, number_of_interval + 1)
        valeurs_poisson_diff_consectif = np.diff(valeurs_poisson)
        vector_poisson_call = [0] + [max(0, value) 
                                     for value in valeurs_poisson_diff_consectif]
        return vector_poisson_call
    
    def get_jump_value(
            self,
            number_of_jumps: float
            ) -> float | int:
        gamma_a=  self.parametre_model["mean of jump"]
        omega_a= self.parametre_model["standar deviation of jump"]
        jump_value = 0
        if number_of_jumps != 0:
            list_normal_value = np.random.normal(gamma_a, omega_a**2, number_of_jumps)
            jump_value = np.sum(list_normal_value)
        return jump_value

    def get_simulations_stock_prices(
            self,
            last_value: float | np.ndarray,
            random_vector: float | np.ndarray,
            random_jump_processus: float | np.ndarray,
            ) -> float | np.ndarray:
        lambda_a =  self.parametre_model["intensity of jump"]
        gamma_a =  self.parametre_model["mean of jump"]
        omega_a = self.parametre_model["standar deviation of jump"]
        sigma2_a = self.parametre_model["volatility stock"]
        mu_a = self.parametre_model["mean stock"]

        step_of_time = self.parametre_simulation["step of time"]

        # Intermedaire
        k_a = np.exp(gamma_a+(omega_a**2)/2)-1
        arg_1 = (mu_a-sigma2_a/2-lambda_a*k_a)
        arg_2 = np.sqrt(sigma2_a * step_of_time)
        arg_exp = arg_1*step_of_time + arg_2*random_vector + random_jump_processus

        stocks = last_value * np.exp(arg_exp)
        return stocks

    def simulations_stock_prices(self) -> np.ndarray:
        initial_value = self.parametre_simulation["initial value"]
        number_of_simulation = self.parametre_simulation["number of simulation"]
    
        step_of_time = self.parametre_simulation["step of time"]
        year_of_projections = self.parametre_simulation["year of projection"]
        number_of_interval = int(year_of_projections / step_of_time)

        matrix_stock_prices = np.ones( (number_of_interval+1, number_of_simulation) ) 
        matrix_stock_prices = matrix_stock_prices * initial_value
        matrix_poisson_processus = self.get_processus_jump(number_of_interval)

        for maturity in range(1, number_of_interval+1):
            jump_value = self.get_jump_value(matrix_poisson_processus[maturity])
            matrix_stock_prices[maturity, :] = self.get_simulations_stock_prices(
                last_value=matrix_stock_prices[maturity, :],
                random_vector=self.random_matrix[maturity - 1, :],
                random_jump_processus=jump_value
            )
        return matrix_stock_prices
    
    def predict(self, *args, **kwargs):
        try:
            freq_of_time = int(1/self.parametre_simulation["step of time"])
            year_of_projection = self.parametre_simulation["year of projection"]
            matrix_stock_prices = self.simulations_stock_prices()
            matrix_stock_prices = np.transpose(
                np.array(
                matrix_stock_prices[
                range(0, year_of_projection*freq_of_time+1, freq_of_time)]
                )
            )
        except Exception as e:
            logging.error(e)
            raise e
        return matrix_stock_prices

if __name__ == '__main__' :
    name_data = os.path.join("monde_reel","data_actions_cours.xlsx")

    # Must have
    initial_value = 100
    number_of_simulation = 1000
    step_of_time = 1/12 # Not required in dict_parametre_simulation
    freq_of_data = "monthly" #frequence of data
    year_of_projection = 40

    start_time = "2009/01/01" # Optional
    finished_time = "2023/12/31" # Optional

    dict_parametre_simulation = {
        "initial value": initial_value,
        "number of simulation": number_of_simulation,
        "frequence of data": freq_of_data,
        "year of projection": year_of_projection,
        "studied period start": start_time,
        "finished period start": finished_time
    }

    matrix_random = np.random.randn(
        int(year_of_projection / step_of_time) + 1,
        number_of_simulation
        )
    parametre_model = [0.004128898,2.0103,0.0010,0.0000,0.0031,0.0057]

    m_act = Merton_RR(
        parametre_simulation=dict_parametre_simulation,
        random_matrix=matrix_random,
        path_data=name_data,
        parametre_model=parametre_model
    )

    m_act.fit()
    array_predit = m_act.predict()
