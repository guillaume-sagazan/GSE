from monde_reel.ModelTaux_RR import *
import pandas as pd
import numpy as np
import os
from scipy import optimize
import logging

class Black_Karinski_RR(ModelTaux_RR):
    """
    On working. Rajouter la possiblité de calibrer sur une courbe des taux
    et vérifier la formule des prix zc
    """

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
        if not parametre_model:
            self.parametre_model = \
                self.get_dict_parametre_model_initialisation()
        elif isinstance(parametre_model, list) or\
                isinstance(parametre_model, np.ndarray) :
            self.parametre_model = self.get_dict_parametre_model_from_list(parametre_model)
        else:
            self.parametre_model = parametre_model 
    
    def get_list_parametre_from_dict(self)-> list:
        list_parametre_model = []
        if isinstance(self.parametre_model, dict):
            for key, value in self.parametre_model.items():
                list_parametre_model.append(value)
        if isinstance(self.parametre_model, list):
            list_parametre_model = self.parametre_model
        return list_parametre_model
    
    def get_parametre_return_to_mean(
            self,
            parametre: list | np.ndarray | dict = []
            ) -> float:
        return_to_mean = 0
        try:
            if isinstance(parametre, dict) and \
                    "return to the mean" in parametre:
                return_to_mean = parametre["return to the mean"]
            elif isinstance(parametre, list) and len(parametre) > 0:
                return_to_mean = parametre[1]
            elif isinstance(parametre, np.ndarray) and len(parametre) > 0:
                return_to_mean = parametre[1]
            elif len(parametre) == 0:
                return_to_mean = self.parametre_model["return to the mean"]
        except Exception as e:
            message_error = f"'message', {parametre}, {type(parametre)}. Error {e}"
            raise ValueError(message_error)
        return return_to_mean
    
    def get_parametre_mean_level(
            self,
            parametre: list | np.ndarray | dict = []
            ) -> float:
        mean_level = 0
        if isinstance(parametre, dict) and \
            "mean level" in parametre:
            mean_level = parametre["mean level"]
        elif isinstance(parametre, list) and len(parametre) > 1:
            mean_level = parametre[0]
        elif isinstance(parametre, np.ndarray) and len(parametre) > 1:
            mean_level = parametre[0]
        elif len(parametre) == 0:
            mean_level = self.parametre_model["mean level"]
        return mean_level
    
    def get_parametre_std_reversion(
            self,
            parametre: list | np.ndarray | dict = []
            ) -> float:
        standard_reversion = 0
        if isinstance(parametre, dict) and \
            "standard reversion" in parametre:
            standard_reversion = parametre["standard reversion"]
        elif isinstance(parametre, list) and len(parametre) > 2:
            standard_reversion = parametre[2]
        elif isinstance(parametre, np.ndarray) and len(parametre) > 2:
            standard_reversion = parametre[2]
        elif len(parametre) == 0:
            standard_reversion = self.parametre_model["standard reversion"]
        return standard_reversion
    
    def get_parametre_shift(
            self,
            parametre: list | np.ndarray | dict = []
            ) -> float:
        if isinstance(parametre, dict) and \
            "shifte rates" in parametre:
            drift = parametre["shifte rates"]
        elif isinstance(parametre, list) and len(parametre) > 3:
            drift = parametre[3] 
        elif isinstance(parametre, np.ndarray) and len(parametre) > 3:
            drift = parametre[3]
        elif "shifte rates" in self.parametre_model:
            drift = self.parametre_model["shifte rates"]
        else:
            drift = 0
        return drift

    def set_random_normal_value_to_matrix(
            self,
            random_matrix_new: np.ndarray
            ) -> None:
        self.random_matrix = random_matrix_new
    
    def _initialize_parametre_model(
            self,
            parametre_model: dict | list
            ):
        if isinstance(parametre_model, dict) and not parametre_model:
            self.parametre_model = self.get_dict_parametre_model_initialisation()
        elif isinstance(parametre_model, dict):
            self.parametre_model = parametre_model
        elif isinstance(parametre_model, list):
            self.parametre_model = self.get_dict_parametre_model_from_list(parametre_model)
        else:
            message_error = f"There is not such type of parametre_model allowed \
                            {type(parametre_model)}. Only List and dict autorize."
            logging.error(message_error)
            raise ValueError(message_error)
    
    def get_dict_parametre_model_initialisation(
            self
            ) -> dict:
        dict_parametre_model = {
            "mean level": float(np.random.uniform(0, 1)),
            "return to the mean": float(np.random.uniform(0, 1)),
            "standard reversion": float(np.random.uniform(0, 1)),
            "shifte rates": 0
        }
        return dict_parametre_model
    
    def get_dict_parametre_model_from_list(
            self,
            list_parametre_model: list | np.ndarray
        ) -> dict:
        if len(list_parametre_model) == 3 and \
            (isinstance(list_parametre_model, list) or\
             isinstance(list_parametre_model, np.ndarray)):
            dict_parametre_model = {
                "mean level": float(list_parametre_model[0]),
                "return to the mean": float(list_parametre_model[1]),
                "standard reversion": float(list_parametre_model[2]),
                "shifte rates": 0
            }
        elif len(list_parametre_model) == 4 and \
            (isinstance(list_parametre_model, list) or\
             isinstance(list_parametre_model, np.ndarray)):
            dict_parametre_model = {
                "mean level": float(list_parametre_model[0]),
                "return to the mean": float(list_parametre_model[1]),
                "standard reversion": float(list_parametre_model[2]),
                "shifte rates": float(list_parametre_model[3])
            }
        elif len(list_parametre_model) in (3, 4) and isinstance(list_parametre_model, dict):
            dict_parametre_model = list_parametre_model
        else:
            message_error = f"The length of list_parametre_model is incorrect.\
                Only the length 3 and 4 are authorized or \
                the length presented is {len(list_parametre_model)} and\
                {type(list_parametre_model)}."
            logging.error(message_error)
            raise ValueError(message_error)
        return dict_parametre_model
    
    def adapt_data(
            self,
            data: pd.Series | pd.DataFrame
        ) -> pd.Series | pd.DataFrame:
        new_data = data.copy()

        shift = self.get_parametre_shift()
        new_data = new_data + shift
        #new_data = new_data.apply(lambda x: np.exp(x))
        return new_data
    
    def get_mean(
            self,
            alpha: float,
            mu: float,
            sigma: float
            ) -> float:
        return mu + sigma**2/(2*alpha)
    
    def get_variance(
            self,
            alpha: float,
            sigma: float,
            h: float
        ) -> float:
        return sigma*2*(1-np.exp(-2*alpha*h))
    
    def get_skewness(
            self,
            mean: float,
            var: float
            ) -> float:
        
        arg_1 = 3*mean + 3*var/2
        exp_1 = 3*np.exp(arg_1)*(np.exp(var)-1)

        arg_2 = 3*mean + var/2
        exp_2 = -3*np.exp(arg_2)

        exp_3 = 2*np.exp(mean + var**2)

        exp_4 = (np.exp(var) - 1)**(3/2) * np.exp(var/2)
        theoretical_skewness = (exp_1 + exp_2 + exp_3) / exp_4
        return theoretical_skewness

    def get_list_of_theoretical_moment(
            self,
            parametre: np.ndarray,
            nbr_years_observed: int,
            initial_rate: float
        ) -> list:
        alpha = self.get_parametre_return_to_mean(parametre)
        mu = self.get_parametre_mean_level(parametre)
        sigma = self.get_parametre_std_reversion(parametre)
        shift = self.get_parametre_shift(parametre)

        # Moments théoriques du modèle
        theoretical_mean = self.get_mean(alpha=alpha, mu=mu, sigma=sigma)
        theoretical_variance = self.get_variance(
            alpha=alpha,
            sigma=sigma,
            h=nbr_years_observed)
        theoretical_skewness = self.get_skewness(
            mean=theoretical_mean,
            var=theoretical_variance
            )

        # Moments théoriques du modèle (skewness)
        # t1 = (initial_rate - mu) * np.exp(-alpha * nbr_years_observed)
        # t2 = (sigma**2 / alpha) * (1 - np.exp(-alpha * nbr_years_observed))
        # theoretical_skewness = (t1 + 2 * t2) / theoretical_variance**(3/2)

        list_theoritical_moment = np.array([
            theoretical_mean, theoretical_variance, theoretical_skewness
            ])
        return list_theoritical_moment
    
    def get_list_of_empiric_moment(
            self,
            observed_returns: np.ndarray | pd.Series
            ) -> np.ndarray:
        observed_mean = np.mean(observed_returns)
        observed_variance = np.var(observed_returns)
        observed_skewness = np.mean((observed_returns - observed_mean)**3) / observed_variance**(3/2)

        list_of_empiric_moment = np.array([
            observed_mean, observed_variance, observed_skewness
            ])
        return list_of_empiric_moment
    
    def compute_objectif_function(
            self,
            parametre: np.ndarray | dict,
            list_of_empiric_moment: np.ndarray,
            nbr_years_observed: int,
            initial_rate: float,
        ) -> float:
        list_of_theoritical_moment = self.get_list_of_theoretical_moment(
                parametre,
                nbr_years_observed = nbr_years_observed,
                initial_rate = initial_rate 
                )
        objectif_cost = np.sum(
            np.abs(
                list_of_empiric_moment - list_of_theoritical_moment
                )
            )
        # print(f"parametre : {parametre}, objectif cost : {objectif_cost}\
        #       list_of_theoritical_moment: {list_of_theoritical_moment}")
        return objectif_cost, list_of_theoritical_moment
    
    def calcul_objectif_function(
            self,
            parametre: np.ndarray,
            list_of_empiric_moment: np.ndarray,
            nbr_years_observed: int,
            initial_rate: float
            ) -> float:
        list_of_theoritical_moment = []
        if 1000 >= parametre[1] >= 1e-10 and \
                1000 >= parametre[2] > 1e-10 and\
                1000 >= parametre[0] > 1e-10: 
            objectif_cost, list_of_theoritical_moment = \
                self.compute_objectif_function(
                    parametre,
                    list_of_empiric_moment,
                    nbr_years_observed,
                    initial_rate
                    )
        else:
            objectif_cost = 1e100
        return objectif_cost
    
    def get_optimization(
            self,
            function_to_optimize,
            parametre_initial: list,
            ) -> list:
        constrain_ineq_std = {
            'type': 'ineq',
            'fun': lambda x: x[2] 
        }
        constrain_ineq_return_to_mean = {
            'type': 'ineq',
            'fun': lambda x: x[1] - 0.001 # to prevent to divide by 0
        }
        constrain_ineq_level_mean = {
            'type': 'ineq',
            'fun': lambda x: x[0]
        }
        list_constrains = [
            constrain_ineq_std, constrain_ineq_return_to_mean, constrain_ineq_level_mean
                ]
        # To increase the stabilization of the result
        parametre_optimal = optimize.minimize(
            function_to_optimize,
            parametre_initial,
            method='SLSQP',
            constraints=list_constrains,
            #options = {'maxiter': 1000}
            ).x
        parametre_optimal = optimize.minimize(
            function_to_optimize,
            parametre_optimal,
            method='SLSQP',
            constraints=list_constrains
            ).x
        return parametre_optimal
    
    def get_initial_rates(
            self,
            data: list | np.ndarray | pd.DataFrame | pd.Series | float
            ) -> float:
        if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            valeur_initial = float(data.iloc[0]) 
        elif isinstance(data, list) or isinstance(data, np.ndarray):
            valeur_initial = float(data[0])
        elif isinstance(data, float):
            valeur_initial = data
        else:
            message_error = f"Wrong type of data.\
                The type is {type(data)}.\
                It should be Dataframe, Series, list, array or float."
            logging.error(message_error)
            raise ValueError(message_error)
        return valeur_initial
    
    def compute_parametre_model(
            self,
            list_stock_prices: np.ndarray | pd.Series
            ) -> dict:
        try:
            data_filter = self.filter_data(list_stock_prices)
            list_of_empiric_moment = self.get_list_of_empiric_moment(data_filter)
            f_optimize = lambda parametre: self.calcul_objectif_function(
                parametre,
                list_of_empiric_moment,
                nbr_years_observed=len(data_filter),
                initial_rate=self.get_initial_rates(data=data_filter)
            )
            parametre_initial = self.get_list_parametre_from_dict() # transforme dict parametre model to list if necessary
            parametre_optimal = self.get_optimization(f_optimize, parametre_initial)
            parametre_model = self.get_dict_parametre_model_from_list(parametre_optimal)
        except ValueError as e:
            logging.error(e)
            raise e
        return parametre_model

    def black_karasinski(self, initial_rates, theta, alpha, sigma):
        dt = self.parametre_simulation["step of time"]
        n = self.parametre_simulation["number of simulation"]
        T = self.parametre_simulation["year of projection"]
        rt = np.zeros(n+1)
        rt[0] = initial_rates
        dwt = np.random.normal(size=n)* np.sqrt(dt)
        for i in range(n):
            rt[i+1] = rt[i] + theta[i]*dt - alpha[i]*rt[i]*dt + sigma[i]*dwt[i]
        return rt

    def calibration_obj(self, x, r0, r_market):
        n = self.parametre_simulation["number of simulation"]
        theta, alpha, sigma = x[:n], x[n:2*n], x[2*n:]
        r_model = self.black_karasinski(r0, theta, alpha, sigma)
        return np.sum((r_model - r_market)**2)
    
    def compute_parametre_model_2(self, data_market) -> list:
        n = self.parametre_simulation["number of simulation"]
        r0 = self.parametre_simulation["initial value"]

        # initial guess for parameters
        theta0 = np.ones(n) * 0.05
        alpha0 = np.ones(n) * 0.3
        sigma0 = np.ones(n) * 0.1
        x0 = np.concatenate([theta0, alpha0, sigma0])
        
        # calibration
        res = optimize.minimize(self.calibration_obj, x0, args=(r0, data_market)).x
        dict_parametre_model = {
                "mean level": res[:n],
                "return to the mean": res[n:2*n],
                "standard reversion": res[2*n:],
                "shifte rates": 0
            }
        return dict_parametre_model

    def fit(self) -> None:
        try:
            data_rates = self.import_data()
            data_adapted = self.adapt_data(data_rates)
            parametre_model = self.compute_parametre_model(data_adapted)
            self.parametre_model = parametre_model
        except Exception as e:
            logging.error(e)
            raise e

    def evaluate_validity_calibration(self) -> tuple:
        data_rates = self.import_data()
        data_adapted = self.adapt_data(data_rates)
        data_filter = self.filter_data(data_adapted)
        list_of_empiric_moment = self.get_list_of_empiric_moment(data_filter)
        list_of_theorical_moment = self.get_list_of_theoretical_moment(
            self.parametre_model,
            nbr_years_observed=len(data_filter),
            initial_rate=self.get_initial_rates(data=data_filter))
        diff_error_mse = np.sum(
           np.abs( list_of_empiric_moment - list_of_theorical_moment)
        )
        rapport_error_mse = np.sum(
           np.abs( list_of_empiric_moment - list_of_theorical_moment) / list_of_theorical_moment
        )
        message_informatif = f"L'écart absolu est {diff_error_mse}. \
            L'écart relatif est {rapport_error_mse}."
        return message_informatif, list_of_empiric_moment, list_of_theorical_moment
    
    def get_simulation_rates(
            self,
            rate_precedent: float | np.ndarray,
            random_vector: float | np.ndarray,
            indice: int
            ) -> float | np.ndarray:
        """
        Execute the Black-Karinski short-rate model.
        
        Args:
            r0 (float): initial short-rate value
            n (int): number of time steps
            dt (float): time step size
            theta (ndarray): array of mean-reversion levels for each time step
            sigma (ndarray): array of volatility values for each time step
            alpha (ndarray): array of mean-reversion speed values for each time step

        Returns:
            ndarray: array of simulated short-rate values
        """
        theta = self.get_parametre_mean_level()
        alpha = self.get_parametre_return_to_mean()
        sigma = self.get_parametre_std_reversion()
        shift = self.get_parametre_shift()

        # Paramètres de discrétisation
        log_rate_precedent = np.log(rate_precedent)
        dt = self.parametre_simulation["step of time"]
        d_r = theta - alpha * log_rate_precedent
        rates = log_rate_precedent + d_r * dt + sigma * np.sqrt(dt) * random_vector
        rates = np.exp(rates)
        return rates
    
    def simulate_rates(self):
        initial_value = self.parametre_simulation["initial value"]
        number_of_simulation = self.parametre_simulation["number of simulation"]
        vector_initial = self.get_vector_full_a_value(initial_value, number_of_simulation)

        step_of_time = self.parametre_simulation["step of time"]
        year_of_projections = self.parametre_simulation["year of projection"]
        number_of_interval = int(year_of_projections / step_of_time)

        matrix_rates = np.zeros( (number_of_interval+1, number_of_simulation) )
        matrix_rates[0,:] = vector_initial
        for maturity in range(1, number_of_interval + 1):
            matrix_rates[maturity, :] = self.get_simulation_rates(
                matrix_rates[maturity-1, :],
                self.random_matrix[maturity, :],
                maturity-1
                )
        
        return matrix_rates
    
    def calculate_prices_zc(
            self,
            matrix_rates: np.ndarray
            ):
        step_of_time = self.parametre_simulation["step of time"]
        year_of_projections = self.parametre_simulation["year of projection"]
        number_of_interval = int(year_of_projections / step_of_time)
        maturity_to_take_account = np.arange(0,
                                             int(number_of_interval)+1,
                                             int(1/step_of_time)
                                             )
        matrix_rates_annual = matrix_rates[maturity_to_take_account, :] 
        maturity_max_projection = self.parametre_simulation["maturity maximal"]
        dim_rates = np.shape(matrix_rates_annual)
        matrice_prices_zc = np.zeros(
            (maturity_max_projection, year_of_projections, dim_rates[1])
            )
        for tenor in range(year_of_projections):
            for maturity in range(maturity_max_projection):
                matrice_prices_zc[maturity, tenor, :] = self.get_prices_zc(
                    maturity = maturity + 1,
                    rates = matrix_rates[tenor,:]
                    )
        return matrice_prices_zc
    
    def get_prices_zc(
            self,
            maturity: int | float,
            rates: float | np.ndarray
            ):
        a = self.parametre_model["return to the mean"]
        b = self.parametre_model["mean level"]
        sigma = self.parametre_model["standard reversion"]
        sigma_sqrt = np.sqrt(sigma)

        sigma_theta = sigma_sqrt * (1 - np.exp(-a*maturity) ) / a
        ln_a_theta_1 = sigma_theta / sigma_sqrt - maturity
        ln_a_theta_2 = b - sigma / (2*a**2)
        ln_a_theta_3 = sigma_theta**2 / (4*a)
        a_theta = np.exp(ln_a_theta_1 * ln_a_theta_2 - ln_a_theta_3)
        pzc_partial = a_theta * np.exp(- sigma_theta/sigma_sqrt * rates)
        return pzc_partial
    
    def get_rates_zc(
            self,
            maturity: float | int,
            prices_zc: float | np.ndarray
            ):
        rates_zc = prices_zc ** (-1/maturity) - 1
        return rates_zc * 100

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
                    prices_zc = matrix_prices_zc[maturity, tenor,:])
        return matrice_rates_zc
        
    
    def predict(self, *args, **kwargs):
        try:
            matrix_rates = self.simulate_rates()
            matrix_prices_zc = self.calculate_prices_zc(
                matrix_rates
                )
            matrix_rates_zc = self.calculate_rates_zc(
                matrix_prices_zc
                )
            dict_simulation = {
                "short_rates": matrix_rates,
                "zero-coupon prices": matrix_prices_zc,
                "zero-coupon rates": matrix_rates_zc
            }
            key_output = "zero-coupon rates"
        except ValueError as e:
            logging.error(e)
            raise e
        return dict_simulation[key_output]