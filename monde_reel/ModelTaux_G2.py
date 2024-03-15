from monde_reel.ModelTaux_RR import *
import pandas as pd
import numpy as np
import os
from scipy import optimize
import logging

class G2_plus(ModelTaux_RR):
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
    
    def get_zero_coupon(self):
        return self.data
    
    def get_step_of_time(self):
        return self.parametre_simulation["step of time"]
    
    def get_parametre_return_to_mean_1(
            self,
            parametre: list | np.ndarray | dict = []
            ) -> float:
        return_to_mean = 0
        try:
            if isinstance(parametre, dict) and \
                    "1: return to the mean" in parametre:
                return_to_mean = parametre["1: return to the mean"]
            elif isinstance(parametre, list) and len(parametre) > 0:
                return_to_mean = parametre[0]
            elif isinstance(parametre, np.ndarray) and len(parametre) > 0:
                return_to_mean = parametre[0]
            elif len(parametre) == 0:
                return_to_mean = self.parametre_model["1: return to the mean"]
        except Exception as e:
            message_error = f"'message', {parametre}, {type(parametre)}. Error {e}"
            raise ValueError(message_error)
        return return_to_mean
    
    def get_parametre_return_to_mean_2(
            self,
            parametre: list | np.ndarray | dict = []
            ) -> float:
        return_to_mean = 0
        try:
            if isinstance(parametre, dict) and \
                    "2: return to the mean" in parametre:
                return_to_mean = parametre["2: return to the mean"]
            elif isinstance(parametre, list) and len(parametre) > 2:
                return_to_mean = parametre[2]
            elif isinstance(parametre, np.ndarray) and len(parametre) > 2:
                return_to_mean = parametre[2]
            elif len(parametre) == 0:
                return_to_mean = self.parametre_model["2: return to the mean"]
        except Exception as e:
            message_error = f"'message', {parametre}, {type(parametre)}. Error {e}"
            raise ValueError(message_error)
        return return_to_mean
    
    def get_parametre_volatilite_1(
            self,
            parametre: list | np.ndarray | dict = []
            ) -> float:
        volatility = 0
        try:
            if isinstance(parametre, dict) and \
                    "1: volatility" in parametre:
                volatility = parametre["1: volatility"]
            elif isinstance(parametre, list) and len(parametre) > 1:
                volatility = parametre[1]
            elif isinstance(parametre, np.ndarray) and len(parametre) > 1:
                volatility = parametre[1]
            elif len(parametre) == 0:
                volatility = self.parametre_model["1: volatility"]
        except Exception as e:
            message_error = f"'message', {parametre}, {type(parametre)}. Error {e}"
            raise ValueError(message_error)
        return volatility
    
    def get_parametre_volatilite_2(
            self,
            parametre: list | np.ndarray | dict = []
            ) -> float:
        volatility = 0
        try:
            if isinstance(parametre, dict) and \
                    "2: volatility" in parametre:
                volatility = parametre["2: volatility"]
            elif isinstance(parametre, list) and len(parametre) > 3:
                volatility = parametre[3]
            elif isinstance(parametre, np.ndarray) and len(parametre) > 3:
                volatility = parametre[3]
            elif len(parametre) == 0:
                volatility = self.parametre_model["2: volatility"]
        except Exception as e:
            message_error = f"'message', {parametre}, {type(parametre)}. Error {e}"
            raise ValueError(message_error)
        return volatility
    
    def get_parametre_correlation_instantaneous(
            self,
            parametre: list | np.ndarray | dict = []
            ) -> float:
        correlation_instantaneous = 0
        try:
            if isinstance(parametre, dict) and \
                    "correlation instantaneous" in parametre:
                correlation_instantaneous = parametre["correlation instantaneous"]
            elif isinstance(parametre, list) and len(parametre) > 4:
                correlation_instantaneous = parametre[4]
            elif isinstance(parametre, np.ndarray) and len(parametre) > 4:
                correlation_instantaneous = parametre[4]
            elif len(parametre) == 0:
                correlation_instantaneous = self.parametre_model["correlation instantaneous"]
        except Exception as e:
            message_error = f"'message', {parametre}, {type(parametre)}. Error {e}"
            raise ValueError(message_error)
        return correlation_instantaneous
    
    def get_dict_parametre_model_initialisation(
            self
            ) -> dict:
        dict_parametre_model = {
                "1: return to the mean": float(np.random.uniform(0, 1)),
                "1: volatility": float(np.random.uniform(0, 1)),
                "2: return to the mean": float(np.random.uniform(0, 1)),
                "2: volatility": float(np.random.uniform(0, 1)),
                "correlation instantaneous": float(np.random.uniform(0, 1))
            }
        return dict_parametre_model
    
    def get_dict_parametre_model_from_list(
            self,
            list_parametre_model: list | np.ndarray
        ) -> dict:
        if len(list_parametre_model) == 5 and \
            (isinstance(list_parametre_model, list) or\
             isinstance(list_parametre_model, np.ndarray)):
            dict_parametre_model = {
                "1: return to the mean": float(list_parametre_model[0]),
                "1: volatility": float(list_parametre_model[1]),
                "2: return to the mean": float(list_parametre_model[2]),
                "2: volatility": float(list_parametre_model[3]),
                "correlation instantaneous": float(list_parametre_model[4])
            }
        elif len(list_parametre_model) == 5 and isinstance(list_parametre_model, dict):
            dict_parametre_model = list_parametre_model
        else:
            message_error = f"The length of list_parametre_model is incorrect.\
                Only the length 3 and 4 are authorized or \
                the length presented is {len(list_parametre_model)} and\
                {type(list_parametre_model)}."
            logging.error(message_error)
            raise ValueError(message_error)
        return dict_parametre_model
    
    def filter_data(self,
            rates: pd.DataFrame | pd.Series
            ) -> np.ndarray | pd.Series:
        data = rates.copy()
        if "studied period start" in self.parametre_simulation and \
            isinstance(rates, pd.DataFrame) and "date" in data.columns:
            min_date = pd.to_datetime(self.parametre_simulation["studied period start"])
            delta_date = self.get_delta_time()
            data = data[data["date"] >= min_date - delta_date]
        if "finished period start" in self.parametre_simulation and \
            isinstance(rates, pd.DataFrame) and "date" in data.columns:
            max_date = pd.to_datetime(self.parametre_simulation["finished period start"])
            data = data[data["date"] <= max_date]
        if isinstance(rates, pd.DataFrame) and "taux" in data.columns:
            data_rates = data[[col for col in data.columns if col != "date"]]
        elif isinstance(rates, pd.Series):
            data_rates = rates 
        return data_rates
    
    def get_correlated_brownian(
            self,
            mouvement_brownian: float | list | np.ndarray,
            parametre: list | float = []
            ) -> np.ndarray:
        if isinstance(parametre, list) or isinstance(parametre, dict): 
            rho = self.get_parametre_correlation_instantaneous(parametre)
        elif isinstance(parametre, float):
            rho = parametre
        else:
            message_error = f"The type of parmaetre do not correspond to\
                list, dict, float. The type is {type(parametre)}."
            logging.error(message_error)
            raise TypeError(message_error)
        try:
            # Matrice de corrélation
            mat_correl = np.array([[1, rho], [rho, 1]])

            # Décomposition de Cholesky
            mat_triangulaire = np.linalg.cholesky(mat_correl)

            random_normal_var = np.random.normal(
                size=np.shape(
                    np.array(mouvement_brownian)
                    )
                )
            var_to_corralated = np.array([random_normal_var, mouvement_brownian])
            mat_correlated_variable = np.array(
                [np.dot(mat_triangulaire, var_to_corralated[:, mat, :])
                 for mat in range(np.shape(mouvement_brownian)[0])])
        except Exception as e:
            raise e
        return mat_correlated_variable
    
    def adapt_data(
            self,
            data: pd.Series | pd.DataFrame
        ) -> pd.Series | pd.DataFrame:
        new_data = data["maturite"].copy()
        # shift = self.get_parametre_shift()
        # new_data = new_data + shift
        #new_data = new_data.apply(lambda x: np.exp(x))
        return new_data
    
    def extract_maturity_data(
            self,
            data: pd.Series | pd.DataFrame
            ):
        list_of_maturity = data["maturite"]
        return list_of_maturity
    
    def calcul_forward_rate_t0(
            self,
            pzc: np.ndarray | list | pd.Series,
            t1: float,
            t2: float
            ) -> float:
        """
        Calcul the forward rates uniquely for time t=0.
            Args:
                self:
                zc_prices: 1d array of the zero coupon prices of dimension higher than int(t1+t2/tau)
                freq: float defines as the steps (in the fraction of the year) of the forward rate
                        ie: if tau = 0.5 then it means semestrial
                t1: value of the maturity
                t2: value of the maturity such as t1 < t2
            Returns:
                float:
                    value of the forward at time 0
        """
        try:
            # initialisation
            freq = self.get_step_of_time()

            if freq <= 0:
                raise ValueError(f"The value of tau is {freq}. It can not be negative or nul")
            if len(pzc) == 0:
                raise ValueError(f"The length of price zero coupon is nul")
            #forward_t0 = pzc[int(t1/freq)]/pzc[int(t2/freq)-1] - 1
            forward_inst_0_T = -(np.log(pzc[int(t1/freq)]/pzc[int(t1/freq)-1]) )
        except Exception as e:
            logging.error(e)
            raise e 
        return forward_inst_0_T/freq
    
    def get_price_zc_from_rate_zc(
            self,
            rates: np.ndarray | pd.Series,
            maturity: np.ndarray | pd.Series
            ) -> pd.Series | np.ndarray:
        prices_array = np.power(1 + rates, -maturity)
        return prices_array
    
    def calibrate_determinist_function(
            self,
            price_data_observed: np.ndarray | pd.Series,
            maturity_observed: np.ndarray,
            parametre: list | np.ndarray
            ) -> np.array:
        a = self.get_parametre_return_to_mean_1(parametre)
        sigma_1 = self.get_parametre_volatilite_1(parametre)
        b = self.get_parametre_return_to_mean_2(parametre)
        sigma_2 = self.get_parametre_volatilite_2(parametre) 
        rho = self.get_parametre_correlation_instantaneous(parametre)

        maturity_observed = np.insert(maturity_observed, 0, 1)
        array_determinist_funct = np.array(
            [self.calcul_forward_rate_t0(price_data_observed, maturity, maturity) + \
             sigma_1**2 / (2*a) * (1 - np.exp(-a*maturity))**2 +\
             sigma_2**2 / (2*b) * (1 - np.exp(-b*maturity))**2 +\
             rho * sigma_1*sigma_2/(a*b)*(1 - np.exp(-a*maturity))\
                *(1 - np.exp(-b*maturity))
             for maturity in maturity_observed
             if maturity/self.get_step_of_time() < len(maturity_observed)-1
             ] 
        )
        return array_determinist_funct
    
    def compute_objectif_function_determinist_funct(
            self,
            reference_curve: np.ndarray | pd.Series,
            maturity_observed: np.ndarray,
            parametre: list | np.ndarray
            ) -> float:
        """Depreciated. A supprimer"""
        price_data_observed = self.get_price_zc_from_rate_zc(
            rates = reference_curve,
            maturity = maturity_observed)
        
        array_theorique_determinist = self.calibrate_determinist_function(
            price_data_observed=price_data_observed,
            maturity_observed=maturity_observed,
            parametre=parametre
        )

        objectif_cost = np.sum(
            np.abs(
                array_theorique_determinist - price_data_observed
                )
            )
        print(f"parametre : {parametre} et objectif_cost:  {objectif_cost}")
        diff = np.abs(array_theorique_determinist - price_data_observed)
        print(f"min: {np.min(diff)}, {np.argmin(diff)}, max: {np.max(diff)}, {np.argmax(diff)}")
        return objectif_cost
    
    def get_optimization(
            self,
            function_to_optimize,
            parametre_initial: list,
            ) -> list:
        constrain_ineq_return_to_mean_1 = {
            'type': 'ineq',
            'fun': lambda x: x[0]
        }
        constrain_ineq_std_1 = {
            'type': 'ineq',
            'fun': lambda x: x[1] 
        }
        constrain_ineq_return_to_mean_2 = {
            'type': 'ineq',
            'fun': lambda x: x[2]
        }
        constrain_ineq_std_2 = {
            'type': 'ineq',
            'fun': lambda x: x[3] 
        }
        constrain_ineq_correl_inf = {
            'type': 'ineq',
            'fun': lambda x: x[4] - 1
        }
        constrain_ineq_correl_sup = {
            'type': 'ineq',
            'fun': lambda x: 1 - x[4] 
        }
        list_constrains = [
            constrain_ineq_return_to_mean_1, constrain_ineq_std_1, \
            constrain_ineq_return_to_mean_2, constrain_ineq_std_2, \
            constrain_ineq_correl_inf, constrain_ineq_correl_sup
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
    
    def calcul_objectif_function(
            self,
            parametre: list | np.ndarray | dict,
            maturity_observed: np.ndarray | pd.Series,
            reference_curve: np.ndarray | pd.Series
        ) -> float:
        if 10 > parametre[0] > 0 and 10 > parametre[1] > 0 and \
           10 > parametre[2] > 0 and 10 > parametre[3] > 0 and \
           1 > parametre[4] > -1:
            objectif_cost = self.compute_objectif_function_determinist_funct(
                maturity_observed = maturity_observed,
                parametre = parametre,
                reference_curve = reference_curve
            )
        else:
            objectif_cost = 1e100
        return objectif_cost
    
    def compute_parametre_model(
            self,
            data: np.ndarray | pd.Series,
            maturity: np.ndarray | pd.Series
            ) -> dict:
        try:
            f_optimize = lambda parametre: self.calcul_objectif_function(
                parametre=parametre,
                maturity_observed=np.array(maturity),
                reference_curve=np.array(data), 
            )
            parametre_initial = self.get_list_parametre_from_dict() # transforme dict parametre model to list if necessary
            parametre_optimal = self.get_optimization(f_optimize, parametre_initial)
            parametre_model = self.get_dict_parametre_model_from_list(parametre_optimal)
        except ValueError as e:
            logging.error(e)
            raise e
        return parametre_model
    
    @staticmethod
    def split_data_to_median(
            self,
            data: np.ndarray
            ) -> list[np.ndarray]:

        # Trouve la médiane
        median_value = np.median(data)
        # Sépare le jeu de données en deux parties
        lower_half = data[data <= median_value]
        if len(lower_half) != len(data)/2:
            upper_half = data[data >= median_value]
        else:
            upper_half = data[data > median_value]
        return [lower_half, upper_half]

    def calcul_mean_reverse_by_MLE(
        self,
        data: np.ndarray
        ) -> float:
        dt = 1

        data_shifted = np.array(data)[1:]
        data_ref = np.array(data)[:-1]
        data_diff = (data_shifted - data_ref) * (-data_ref)

        data_ref_carre = data_ref * data_ref

        reverse_mean = np.sum(data_diff) / (np.sum(data_ref_carre) * dt)
        return reverse_mean

    def calcul_ecart_type_by_MLE(
        self,
        data: np.ndarray,
        reverse_mean: float
        ) -> np.ndarray:
        dt = 1 

        data_shifted = np.array(data)[1:]
        data_ref = np.array(data)[:-1]

        data_sum = data_shifted - data_ref - reverse_mean * data_ref * dt
        data_sum_carre = data_sum * data_sum

        ecart_type = np.sum(data_sum_carre) / (dt*len(data_shifted))
        return np.sqrt(ecart_type)
    
    def compute_parametre_model_MLE(
            self,
            data: np.ndarray
            ) -> np.ndarray:
        """
        Determine parametre by the maximun likelihood estimator"""
        data_under_median, data_upper_median = self.split_data_to_median(data)
        lambda_1 = self.calcul_mean_reverse_by_MLE(data_under_median)
        lambda_2 = self.calcul_mean_reverse_by_MLE(data_upper_median)
        std_1 = self.calcul_ecart_type_by_MLE(data_under_median, lambda_1)
        std_2 = self.calcul_ecart_type_by_MLE(data_upper_median, lambda_2)
        coeff_correl = np.corrcoef(data_under_median, data_upper_median)[0,1]
        parametre_model = self.get_dict_parametre_model_from_list(
            [lambda_1, std_1, lambda_2, std_2, coeff_correl]
        )
        return parametre_model

    def fit(self) -> None:
        try:
            data_rates = self.import_data()
            data_filter = self.filter_data(data_rates)
            data_adapted = self.adapt_data(data_filter)
            maturity_of_data = self.extract_maturity_data(data_filter)
            parametre_model = self.compute_parametre_model(
                data_adapted, maturity_of_data)
            self.parametre_model = parametre_model
        except Exception as e:
            logging.error(e)
            raise e
        
    def calcul_v(
            self,
            t: int | float,
            T: int | float
            ) -> float:
        a = float(self.get_parametre_return_to_mean_1())
        sigma_1 = float(self.get_parametre_volatilite_1())
        b = float(self.get_parametre_return_to_mean_2())
        sigma_2 = float(self.get_parametre_volatilite_2() )
        rho = float(self.get_parametre_correlation_instantaneous())
        # Calcul de la fonction V(t, T) subdivisé en plusieurs étapes
        int_1 = T - t + 2 * np.exp(-a * (T - t)) / a - \
            np.exp(-2 * a * (T - t)) / (2 * a) - 3 / (2 * a)
        term1 = (sigma_1/ a)**2 * int_1

        int_2 = T - t + 2 * np.exp(-b * (T - t)) / b - \
            np.exp(-2 * b * (T - t)) / (2 * b) - 3 / (2 * b)
        term2 = (sigma_1/ b)**2  * int_2

        int_3 = T - t + (np.exp(-(a+b)*(T - t)) - 1) / (a+b) +\
            (np.exp(-a * (T - t)) - 1)/a +\
            (np.exp(-b * (T - t)) - 1)/b 
        term3 = 2 * rho * sigma_1 * sigma_2 * int_3 / (a * b)

        result = term1 + term2 + term3
        return result
    
    def get_prices_zc(
            self,
            t: float | int,
            T: float | int,
            PM_0_t: float | int,
            PM_0_T: float | int,
            x: float | np.ndarray,
            y: float | np.ndarray,
            ) -> float | np.ndarray:
        """
        Compute the zero coupon bond price.
        Args:
            self,
                t: float | int, value of time tenor t
                T: float | int, value of time tenor T
                PM_0_t: float | int, value of theprices of the zc bond at t
                PM_0_T: float | int, value of theprices of the zc bond at T
                x: float | np.ndarray, value of the random model 1
                y: float | np.ndarray, value of the random model 2
        Return:
            float or np.ndarray, it depends function of x or y type
            that correspond to the value of a zero coupon bond
        """
        a = self.get_parametre_return_to_mean_1()
        b = self.get_parametre_return_to_mean_2()

        # Calcul des paramètres A, B1, B2, V
        v_t_T = self.calcul_v(t, T)
        v_0_T = self.calcul_v(0, T)
        v_0_t = self.calcul_v(0, t)

        A = PM_0_T / PM_0_t * np.exp((v_t_T - v_0_T + v_0_t)/2)
        B1 = (1 - np.exp(-a * (T - t))) / a
        B2 = (1 - np.exp(-b * (T - t))) / b

        # Calcul de P(t, T)
        P_value = A * np.exp(-B1 * x[t] - B2 * y[t])

        return P_value
    
    def calculate_prices_zc(
            self,
            matrix_model_1: np.ndarray,
            matrix_model_2: np.ndarray
        ) -> np.ndarray:
        step_of_time = self.parametre_simulation["step of time"]
        year_of_projections = self.parametre_simulation["year of projection"]
        number_of_interval = int(year_of_projections / step_of_time)
        maturity_to_take_account = np.arange(
            0,
            int(number_of_interval)+1,
            int(1/step_of_time)
            )
        matrix_rates_annual_model_1 = matrix_model_1[maturity_to_take_account, :]
        matrix_rates_annual_model_2 = matrix_model_2[maturity_to_take_account, :]  
        maturity_max_projection = self.parametre_simulation["maturity maximal"]

        dim_rates = np.shape(matrix_rates_annual_model_1)
        matrice_prices_zc = np.zeros(
            (maturity_max_projection, year_of_projections, dim_rates[1])
            )
        
        curve_prices_zc = np.insert(self.get_zero_coupon(), 0, 1)
        for tenor in range(year_of_projections):
            for maturity in range(maturity_max_projection):
                matrice_prices_zc[maturity, tenor, :] = self.get_prices_zc(
                    t = maturity,
                    T = tenor + 1,
                    PM_0_t = curve_prices_zc[maturity],
                    PM_0_T = curve_prices_zc[tenor + 1],
                    x = matrix_rates_annual_model_1[maturity, :],
                    y = matrix_rates_annual_model_2[maturity, :],
                    )
        return matrice_prices_zc
    
    def get_simulation_rates(
            self,
            rate_precedent: float | np.ndarray,
            random_vector: float | np.ndarray,
            indice_variable: int
            ) -> float | np.ndarray:
        if indice_variable == 0:
            a = self.get_parametre_return_to_mean_1()
            sigma = self.get_parametre_volatilite_1()
        if indice_variable == 1:
            a = self.get_parametre_return_to_mean_2()
            sigma = self.get_parametre_volatilite_2()
        
        alea_part = sigma * np.sqrt((1-np.exp(-2*a))/(2*a))*random_vector
        rates = rate_precedent * np.exp(-a) + alea_part
        return rates

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
    
    def simulate_rates(self):
        rho = self.get_parametre_correlation_instantaneous()

        mouvement_browniens = self.get_correlated_brownian(
            self.random_matrix,
            parametre=float(rho)
            )
        # information simulation
        number_of_simulation = self.parametre_simulation["number of simulation"]
        step_of_time = self.parametre_simulation["step of time"]
        year_of_projections = self.parametre_simulation["year of projection"]
        number_of_interval = int(year_of_projections / step_of_time)

        matrix_rates_x = np.zeros( (number_of_interval+1, number_of_simulation) )
        matrix_rates_y = np.zeros( (number_of_interval+1, number_of_simulation) )
        for maturity in range(1, number_of_interval + 1):
            matrix_rates_x[maturity, :] = self.get_simulation_rates(
                matrix_rates_x[maturity-1, :],
                mouvement_browniens[maturity, 0, :],
                indice_variable = 0
                )
            matrix_rates_y[maturity, :] = self.get_simulation_rates(
                matrix_rates_y[maturity-1, :],
                mouvement_browniens[maturity, 1, :],
                indice_variable = 1
                )
        return matrix_rates_x, matrix_rates_y
    
    def predict(self, *args, **kwargs):
        try:
            matrix_rates_model_1, matrix_rates_model_2 = self.simulate_rates()
            matrix_prices_zc = self.calculate_prices_zc(
                matrix_rates_model_1,
                matrix_rates_model_2
                )
            matrix_rates_zc = self.calculate_rates_zc(
                matrix_prices_zc
                )
            dict_simulation = {
                "short_rates": [matrix_rates_model_1, matrix_rates_model_2],
                "zero-coupon prices": matrix_prices_zc,
                "zero-coupon rates": matrix_rates_zc
            }
            key_output = "zero-coupon rates"
        except ValueError as e:
            logging.error(e)
            raise e
        return dict_simulation[key_output]

