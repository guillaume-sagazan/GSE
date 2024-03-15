import pandas as pd
import numpy as np
import os
import logging
from abc import ABC, abstractmethod
from dateutil.relativedelta import relativedelta
import copy

class ModelTaux_RR(ABC):

    def __init__(
            self,
            parametre_simulation : dict,
            path_data: str = "",
            data: pd.DataFrame | np.ndarray = [],
            ) -> None:
        self.parametre_simulation = parametre_simulation
        self.path_data = path_data
        if len(data) == 0 and path_data != "":
            self.data = self.import_data()
        elif len(data) != 0:
            self.data = data
        else:
            message_warning = "No data or path data in input"
            logging.warning(message_warning)

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
    
    def get_path_data(self):
        return self.path_data
    
    def get_data(self):
        return self.data

    @abstractmethod
    def fit(self) -> None:
        pass

    @abstractmethod
    def predict(self):
        pass

    def get_vector_full_a_value(
            self,
            value: float,
            nbr_occurence: int
            ):
        return np.full(nbr_occurence, value)

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
                data_imported = self.data
        except ValueError as e: 
            logging.error(e)
            raise e
        return data_imported
    
    def set_random_normal_value_to_matrix(
            self,
            matrix_reference: np.ndarray = [],
            shape_of_matrice: tuple or list= ()
            ) -> np.ndarray:
        if len(matrix_reference) != 0:
            random_matrix = np.random.normal(
            loc=0,
            scale=1, 
            size=np.shape(matrix_reference)
            )
        elif shape_of_matrice:
            random_matrix = np.random.normal(
            loc=0,
            scale=1, 
            size=shape_of_matrice
            )
        return random_matrix

class Vasicek_RR(ModelTaux_RR):

    def __init__(
            self,
            parametre_simulation: dict,
            random_matrix: np.ndarray,
            path_data: str = "",
            data: pd.DataFrame | np.ndarray = [],
            parametre_model: dict = {},
            ) -> None:
        super().__init__(parametre_simulation, path_data, data)
        self._initiliaze_frequence_of_data()
        self.random_matrix = random_matrix
        self.parametre_model = parametre_model
    
    def copy(self):
        return copy.deepcopy(self)
    
    def get_step_of_time(self):
        if "step of time" in self.parametre_simulation:
            step_of_time = self.parametre_simulation["step of time"] 
        else:
            message_error = f"The parametre step_of_time is not present in\
                parametre_simulation."
            logging.error(message_error)
            raise ValueError(message_error)
        return step_of_time
    
    def set_data(
            self,
            data: np.ndarray
            ) -> None:
        self.data = data
    
    def set_random_matrix(
            self,
            random_matrix_new: np.ndarray
            ) -> None:
        self.random_matrix = random_matrix_new
    
    def shifted_data(
            self,
            data: pd.Series | np.ndarray
            ) -> pd.Series | np.ndarray:
        if isinstance(data, pd.Series) :
            list_data = data[1:]
        if isinstance(data, list):
            list_data = np.ndarray(data)[1:]
        if isinstance(data, np.ndarray):
            list_data = data[1:]
        return list_data
    
    def get_coeff_a_b(
            self,
            number_of_element: int,
            list_data_adapted: np.ndarray | pd.Series,
            list_data_shifted: np.ndarray | pd.Series
            ):
        list_square = list_data_adapted * list_data_adapted
        list_multiplication_shifted = list_data_adapted * list_data_shifted

        sum_multiplication_shifted = np.sum(list_multiplication_shifted)
        sum_data_adapted = np.sum(list_data_adapted)
        sum_data_shifted = np.sum(list_data_shifted)
        sum_data_square = np.sum(list_square)

        a_1 = sum_multiplication_shifted-(sum_data_shifted * sum_data_adapted)/number_of_element
        a_2 = sum_data_square - sum_data_adapted**2/number_of_element
        a = a_1/a_2

        b = (sum_data_shifted - a * sum_data_adapted)/number_of_element

        return a, b
    
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
    
    def filter_data(self,
            rates: pd.DataFrame | pd.Series
            ) -> np.ndarray | pd.Series:
        data = rates.copy()
        if "studied period start" in self.parametre_simulation and \
            isinstance(rates, np.ndarray) and "date" in data.columns:
            min_date = pd.to_datetime(self.parametre_simulation["studied period start"])
            delta_date = self.get_delta_time()
            data = data[data["date"] >= min_date - delta_date]
        if "finished period start" in self.parametre_simulation and \
            isinstance(rates, np.ndarray) and "date" in data.columns:
            max_date = pd.to_datetime(self.parametre_simulation["finished period start"])
            data = data[data["date"] <= max_date]
        if isinstance(rates, pd.DataFrame):
            data_rates = data["taux"]
        if isinstance(rates, np.ndarray) or \
            isinstance(rates, pd.Series):
            data_rates = rates
        return data_rates

    def compute_parametre_model(
            self,
            list_data: np.ndarray | pd.Series
            ) -> tuple:
        data_filter = self.filter_data(rates=list_data)
        list_data_shifted = self.shifted_data(data_filter)
        list_data_adapted = data_filter[:-1]
        number_of_element = len(list_data_adapted)
        #get a et b
        coeff_a, coeff_b = self.get_coeff_a_b(
            number_of_element,
            list_data_adapted,
            list_data_shifted
            )
        
        coeff_lambda = (1-coeff_a) / self.parametre_simulation["step of time"]
        coeff_mu = coeff_b / (1 - coeff_a)
        coeff_sigma = np.var(list_data_adapted)
        if coeff_lambda < 0:
            message_warnig = f"The value of 'speed of reversion'" +\
                "is negatif. The absolute value will be taking."
            logging.warning(message_warnig)
            coeff_lambda = abs(coeff_lambda)
        return coeff_lambda, coeff_mu, coeff_sigma
    
    def get_dict_parametre_model_from_list(
            self,
            list_parametre_model: list | np.ndarray 
            ) -> dict:
        dict_parametre_model = {
            "speed of reversion": float(list_parametre_model[0]),
            "long terme mean": float(list_parametre_model[1]),
            "instantaneous volatility": float(list_parametre_model[2])
        }
        return dict_parametre_model
    
    def alter_type_paremetre_model(
            self,
            list_parametre_model
            ) -> list:
        """Convert parametre into a float"""
        list_of_parametre_altered = []
        for coeff in list_parametre_model:
            if isinstance(coeff, pd.Series):
                coeff = coeff.iloc[0]
            coeff_float = float(coeff)
            list_of_parametre_altered.append(coeff_float)
        return list_of_parametre_altered

    def fit(self) -> None:
        data_rate = self.get_data()
        # Une étape de filtration doit être inclus
        coeff_lambda, coeff_mu, coeff_sigma = self.compute_parametre_model(data_rate)
        list_parametre_model = self.alter_type_paremetre_model(
            [coeff_lambda, coeff_mu, coeff_sigma]
        )
        self.parametre_model = self.get_dict_parametre_model_from_list(list_parametre_model)

    def get_deflator(
            self,
            matrix_rates: np.ndarray
            ):
        step_of_time = self.parametre_simulation["step of time"]
        year_of_projections = self.parametre_simulation["year of projection"]
        number_of_interval = year_of_projections / step_of_time
        maturity_to_take_account = np.arange(
            int(1/step_of_time)-1,
            int(number_of_interval),
            int(1/step_of_time)
            )
        deflator = np.exp(
                    -np.apply_along_axis(
                        np.cumsum,
                        0,
                        matrix_rates * self.parametre_simulation["step of time"]
                        )
                    )
        if not isinstance(deflator, np.ndarray):
            deflator = np.array(deflator)
        deflator = deflator[maturity_to_take_account, :]
        return deflator
    
    def get_prices_zc(
            self,
            maturity: int | float,
            rates: float | np.ndarray
            ):
        a = self.parametre_model["speed of reversion"]
        b = self.parametre_model["long terme mean"]
        sigma = self.parametre_model["instantaneous volatility"]
        sigma_sqrt = np.sqrt(sigma)

        sigma_theta = sigma_sqrt * (1 - np.exp(-a*maturity) ) / a
        ln_a_theta_1 = sigma_theta / sigma_sqrt - maturity
        ln_a_theta_2 = b - sigma / (2*a**2)
        ln_a_theta_3 = sigma_theta**2 / (4*a)
        a_theta = np.exp(ln_a_theta_1 * ln_a_theta_2 - ln_a_theta_3)
        pzc_partial = a_theta * np.exp(- sigma_theta/sigma_sqrt * rates)
        return pzc_partial
    
    def calculate_prices_zc(
            self,
            matrix_rates: np.ndarray,
            is_only_year: bool
            ):
        step_of_time = self.parametre_simulation["step of time"]
        year_of_projections = self.parametre_simulation["year of projection"]
        number_of_interval = int(year_of_projections / step_of_time)
        if is_only_year:
            year_to_take_account = np.arange(
                0, int(number_of_interval)+1, int(1/step_of_time)
                )
        else:
            year_to_take_account = np.arange(
                0, int(number_of_interval)+1, 1)
        matrix_rates_annual = matrix_rates[year_to_take_account, :] 
        maturity_max_projection = self.parametre_simulation["maturity maximal"]
        dim_rates = np.shape(matrix_rates_annual)

        matrice_prices_zc = np.zeros(
            (maturity_max_projection, dim_rates[0], dim_rates[1])
            )
        for tenor in range(dim_rates[0]):
            for maturity in range(maturity_max_projection):
                matrice_prices_zc[maturity, tenor, :] = self.get_prices_zc(
                    maturity = maturity + 1,
                    rates = matrix_rates_annual[tenor,:]
                    )
        return matrice_prices_zc

    def get_rates_zc(
            self,
            maturity: float | int,
            prices_zc: float | np.ndarray
            ):
        rates_zc = prices_zc ** (-1/maturity) - 1
        return rates_zc

    def calculate_rates_zc(
            self,
            matrix_prices_zc: np.ndarray
            ):
        matrice_rates_zc = np.zeros_like( matrix_prices_zc )
        dim_prices_zc = np.shape(matrix_prices_zc)
        for tenor in range(dim_prices_zc[1]):
            for maturity in range(dim_prices_zc[0]):
                matrice_rates_zc[maturity, tenor, :] = self.get_rates_zc(
                    maturity = maturity + 1,
                    prices_zc = matrix_prices_zc[maturity, tenor, :])
        return matrice_rates_zc

    def get_simulation_rates(
            self,
            rate_precedent: float | np.ndarray,
            random_vector: float | np.ndarray
            ) -> float:
        """Compute the rate for t"""
        a = self.parametre_model["speed of reversion"]
        b = self.parametre_model["long terme mean"]
        sigma = self.parametre_model["instantaneous volatility"]
        step_of_time = self.parametre_simulation["step of time"]

        value_int_0 = rate_precedent * (1-a*step_of_time) 
        value_int_1 =  a * b * step_of_time 
        value_int_2 = np.sqrt( sigma*(1 - np.exp(-2*b*step_of_time))/ (2 * b)) 
        rates = value_int_0 + value_int_1 + value_int_2 * random_vector
        return rates
    
    def simulate_rates(self):
        """Execute the Vasicek rate simulation"""
        initial_value = self.parametre_simulation["initial value"]
        number_of_simulation = self.parametre_simulation["number of simulation"]
        vector_initial = self.get_vector_full_a_value(initial_value, number_of_simulation)

        step_of_time = self.parametre_simulation["step of time"]
        year_of_projection = self.parametre_simulation["year of projection"]
        number_of_interval = int(year_of_projection / step_of_time)

        matrix_rates = np.zeros( (number_of_interval+1, number_of_simulation) )
        matrix_rates[0,:] = vector_initial
        for maturity in range(1, number_of_interval + 1):
            matrix_rates[maturity, :] = self.get_simulation_rates(
                matrix_rates[maturity-1, :],
                self.random_matrix[maturity - 1, :]
                )        
        return matrix_rates
    
    def predict(self, key_output='', is_only_year=True, *args, **kwargs):
        try:
            # simulation with a step of time of frequence(not annual)
            matrix_rates = self.simulate_rates()
            deflator = self.get_deflator(matrix_rates)
            matrix_prices_zc = self.calculate_prices_zc(
                matrix_rates,
                is_only_year=is_only_year
                )
            matrix_rates_zc = self.calculate_rates_zc(
                matrix_prices_zc
                )
            dict_simulation = {
                "short_rates": matrix_rates,
                "deflator": deflator,
                "zero-coupon prices": matrix_prices_zc,
                "zero-coupon rates": matrix_rates_zc
            }
            if key_output == '':
                key_output = "zero-coupon rates"
        except ValueError as e:
            logging.error(e)
            raise e
        return dict_simulation[key_output]


if __name__ == '__main__':
    # parametre de teste
    name_file = os.path.join("monde_reel","data_taux_court.xlsx")
    lambda_t = 0.0355279589
    mu_t = -0.02931970
    sigma = 0.0000040031
    valeur_initial = 0.03295
    nombre_simulation = 1000
    annee_projection = 40
    max_maturity = 40
    freq_of_data = "monthly"
    dt = 1/12

    dict_parametre_model = {
        "speed of reversion": lambda_t,
        "long terme mean": mu_t,
        "instantaneous volatility": sigma,
    }
    dict_parametre_simulation = {
        "step of time": dt,
        "year of projection": annee_projection,
        "number of simulation" : nombre_simulation,
        "maturity maximal": max_maturity,
        "initial value": valeur_initial,
        "frequence of data": freq_of_data
    }
    matrix_random = np.random.randn(
        int(annee_projection / dt) + 1,
        nombre_simulation
        )
    # cdt_rr = Vasicek_RR(
    #     parametre_simulation = dict_parametre_simulation,
    #     path_data = name_file,
    #     random_matrix=matrix_random
    # )
    cdt_rr = Vasicek_RR(
        parametre_simulation = dict_parametre_simulation,
        path_data = name_file,
        #data = rates,
        random_matrix=matrix_random
    )
    cdt_rr.parametre_simulation
    data = cdt_rr.data
    cdt_rr.fit()
    cdt_rr.parametre_model
    cdt_rr.predict('zero-coupon prices')

