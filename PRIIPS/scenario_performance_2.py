from PRIIPS.scenario_performance import ScenarioPerformance
import pandas as pd
import numpy as np
from scipy.stats import norm
import logging
import re

class ScenarioPerformance2(ScenarioPerformance):
       
    def __init__(
            self,
            matrix_yield: np.ndarray,
            dict_parametre_scenario_performance: dict,
            is_logging_activate: bool = False
        ) -> None:
        super().__init__(
            matrix_yield,
            dict_parametre_scenario_performance,
            is_logging_activate=is_logging_activate)
        self.get_category_mapping_dictionary_for_funct_performance()
        self.verify_input_for_creation_of_object()
        self.associate_category_to_type_of_performance()
        self.adapt_dict_parametre_quantil_perf_if_exist()
    
    def get_matrix_yield(self):
        if isinstance(self.matrix_yield, dict):
            matrix_yield = self.matrix_yield["defavoable"]
        if isinstance(self.matrix_yield, np.ndarray):    
            matrix_yield = self.matrix_yield
        return matrix_yield
    
    def get_category_mapping_dictionary_for_funct_performance(self) -> None:
        """
        all the test function have to be add to the dictionary and including the terme test
        performance_funct is a mandatory key to have and must be associated with 
        the function that calculate scenario_performance
        """
        dict_map = {
            "test_function": self.verify_if_all_input_of_cat_2_is_present_in_dict,
            "test_rhp_valide": self.verify_that_rhp_correct_value,
            "performance_funct": self.scenario_performance_cat_2,
            "get_performance_net": self.scenario_performance_net_cat_2,
        }
        self.mapping_funct = dict_map
    
    def get_dict_mapping_funct(self):
        return self.mapping_funct
    
    def verify_if_all_input_of_cat_2_is_present_in_dict(self) -> None:
        """
        Verify if all the input of the dictionary dict_parametre_scenario_performance
        are present for the category 2
        Return:
            None: raise an error if any of the key are missing 
        """
        list_of_key_mandatory = [
            "rhp", "data_frequency", "mnt_investment_initial", "category"]
        dict_parametre = self.dict_parametre_scenario_performance
        if not all(key_tested in dict_parametre 
               for key_tested in list_of_key_mandatory):
            message_error = f"For the category the element required is \
            {list_of_key_mandatory} but {dict_parametre.keys} is presented."
            logging.error(message_error)
            raise ValueError(message_error)
    
    def adapt_dict_parametre_quantil_perf_if_exist(self) -> None:
        """
        if dict_quantil exist in the attribut dict_parametre_scenario_performance
        than adapt the code.
        """
        if not "dict_quantil" in self.dict_parametre_scenario_performance:
            dict_quantil_personalised_adapted = {}
            regex_fav = r'\b(\w*)(_{1}|^)(fav)(_{1}|$)(\w*)\b'
            regex_defav = r'\b(\w*)(_{1}|^)(defav)(_{1}|$)(\w*)\b'
            regex_middle = r'\b(\w*)(_{1}|^)(middle)(_{1}|$)(\w*)\b'
            for each_key, values in self.dict_parametre_scenario_performance.items():
                if re.match(regex_fav, each_key):
                    dict_quantil_personalised_adapted["fav"] = values
                elif re.match(regex_defav, each_key):
                    dict_quantil_personalised_adapted["defav"] = values
                elif re.match(regex_middle, each_key):
                    dict_quantil_personalised_adapted["middle"] = values
            if dict_quantil_personalised_adapted == {}:
                message_error = f"One quantil argument is lacking. \n\
                    At least one quantil must be present or detected"
                logging.error(message_error)
                raise ValueError(message_error)
            self.dict_parametre_scenario_performance["dict_quantil"] = \
                dict_quantil_personalised_adapted

    def get_dict_frequence_of_data(self) -> dict:
                dict_map_freq = {
                    "annual": 1,
                    "half-yearly": 2,
                    "quarterly": 4,
                    "monthly": 12,
                    "bimonthly": 24,
                    "weekly": 52,
                    "biweekly": 104,
                    "daily": 252
                    }
                return dict_map_freq

    def get_rhp_to_str(self,
                       rhp: int | float
                    ) -> str:
        if rhp == 1:
            str_rhp = "1 ans"
        elif rhp > 1:
            str_rhp = "sup 1 ans"
        else:
            message_error = f"The value of rhp is not valid.\
                It can not be negatif or null. The rhp's value is {rhp}."
            logging.error(message_error)
            raise ValueError(message_error)
        return str_rhp

    def get_dict_parametre_interval_to_calcul_vol_tension(self,) -> dict:
        """
        Initialize the dictionary of parametre use to calculate the 
        volatily under tension for the category 2 by initializing the interval
        of data selectionned. 
        Update after the reform 2021 
        """
        dict_parametrage = {
            "1 ans": {
                "daily prices": 21,
                "weekly prices": 8,
                "monthly prices": 6,
            },
            "sup 1 ans": {
                "daily prices": 63,
                "weekly prices": 16,
                "monthly prices": 12,
            }
        }
        return dict_parametrage
    
    def get_dict_parametre_quantil_perf(self) -> dict:
        """
        Initialize the dictionary of parametre use to calculate the 
        performance scenario. 
        Update after the reform 2021 
        """
        if "dict_quantil" in self.dict_parametre_scenario_performance and \
            isinstance(self.dict_parametre_scenario_performance["dict_quantil"], dict) and\
            self.dict_parametre_scenario_performance["dict_quantil"] != {}:
            dict_parametrage = self.dict_parametre_scenario_performance["dict_quantil"]
        else:
            dict_parametrage = {
                'fav': 95,
                'defav': 10,
            }
        return dict_parametrage

    def get_dict_parametre_quantil_calcul_vol_tension(self,) -> dict:
        """
        Initialize the dictionary of parametre use to calculate the 
        volatily under tension for the category 2. 
        Update after the reform 2021 
        """
        dict_parametrage = {
            '1 ans': 99,
            'sup 1 ans': 95
        }
        return dict_parametrage

    def get_quantil_to_calculate_tension_performance(
        self,
        rhp: int | float
    ) -> float:
        rhp_str = self.get_rhp_to_str(rhp=rhp)
        dict_parametre_quantil = self.get_dict_parametre_quantil_calcul_vol_tension()
        quantil = dict_parametre_quantil[rhp_str]
        return quantil

    def split_array_in_subarray(
            self,
            list_to_split: list | np.ndarray | pd.Series,
            length_of_sublist: int
        ) -> list[np.ndarray]:
        """
        Create a list of subarray with the size length_of_sublist.
        """
        number_of_subarray = len(list_to_split) - length_of_sublist + 1
        list_of_subarray = [
            np.array(list_to_split[indice:indice + length_of_sublist]) 
            for indice in range(number_of_subarray)
            ]
        return list_of_subarray

    def get_vol_tension_cat_2(
            self,
            log_yield: np.ndarray | pd.Series | pd.DataFrame,
            rhp: str,
            data_frequency: str,
            number_of_negotiation_in_period_calcul: int | float
        ):
        """
        Compute the tension volatility for the category 2

        Args:
            log_yield: np.ndarray | pd.Series | pd.DataFrame
        contains the log yield. The set must be a 1nd array
            rhp: str can be choose between :
                '1 ans',
                'sup 1 ans'.
        Indicates the year of projection. It will be impactful to the parametre
        used to create the subinterval and the quantil.
            data_frequency: str can be choose between :
                'daily prices',
                'weekly prices',
                'monthly prices'.
        Indicates the frequence refresh of data. It will be impactful to the parametre
        used to create the subinterval.
            number_of_negotiation_in_period_calcul: int | float. Indicates the number 
        of contract negotiations during the period of time considered.

        Returns:
            float: value of the tension volatility for the category 2
        """
        dict_parametre_interval = self.get_dict_parametre_interval_to_calcul_vol_tension()
        dict_parametre_quantil = self.get_dict_parametre_quantil_calcul_vol_tension()
        length_of_interval = dict_parametre_interval[rhp][data_frequency]
        quantil = dict_parametre_quantil[rhp]
        subarray_log_yield = self.split_array_in_subarray(
            log_yield,
            length_of_interval
            )
        std_subarray_log_yield = [np.std(subarray) for subarray in subarray_log_yield]
        vol_tension = np.percentile(std_subarray_log_yield, quantil) *\
            number_of_negotiation_in_period_calcul**0.5
        return vol_tension

    def compute_tension_vol(
            self,
            log_yield_data: np.ndarray,
            rhp: int | float,
            data_frequency: str,
            category: int,
            number_of_negotiation_in_period_calcul: int | float
        ) -> float:   
        """
        Compute the tension volatility.
        
        Args: 
            log_yield_data: 1 ndarray of the yield considerated
            rhp: int | float that indicates the rhp studied
        Indicates the year of projection. It will be impactful to the parametre
        used to create the subinterval and the quantil.
            data_frequency: str can be choose between :
                'daily prices',
                'weekly prices',
                'monthly prices'.
            category: int that indicates the category of the action .
        This parametre changes the way to calculate the tension volatility
        number_of_negotiation_in_period_calcul: int | float. Indicates the number 
        of contract negotiations during the period of time considered.

        Returns:
            float : that correspond to the tension volatility
        """
        rhp_str = self.get_rhp_to_str(rhp=rhp)
        if category==2:
            tension_volatility = self.get_vol_tension_cat_2(
                log_yield=log_yield_data,
                rhp=rhp_str,
                data_frequency=data_frequency,
                number_of_negotiation_in_period_calcul=\
                    number_of_negotiation_in_period_calcul
            )
        else:
            tension_volatility = np.var(log_yield_data)
        return tension_volatility

    def compute_skewness(
            self,
            matrix_yield: np.ndarray
            ) -> float:
        """
        The shape of matrix_yield must be 
        (number of simulation x 1 projection of the time)
        """
        # Calcul de la moyenne et de l'écart type des rendements
        moyenne = self.compute_mean(matrix_yield)
        ecart_type = self.compute_volatility(matrix_yield)  

        # Calcul du coefficient d'asymétrie
        denominator = (len(matrix_yield) * ecart_type ** 3)
        numerator = np.sum((matrix_yield - moyenne) ** 3)
        skewness =  numerator/ denominator
        return skewness

    def compute_kurtosis(
            self,
            matrix_yield: np.ndarray
        ) -> float:
        # Calcul de la moyenne et de l'écart type des rendements
        moyenne = self.compute_mean(matrix_yield)
        ecart_type = self.compute_volatility(matrix_yield)

        # Calcul du coefficient de kurtosis
        denominator = (len(matrix_yield) * ecart_type ** 4)
        numerator = np.sum((matrix_yield - moyenne) ** 4)
        kurtosis =  numerator/denominator  - 3
        return kurtosis

    def compute_volatility(
            self,
            matrix_yield: np.ndarray
        ) -> float:
        ecart_type = np.std(matrix_yield, ddof=0) 
        return ecart_type

    def compute_mean(
            self,
            matrix_yield: np.ndarray
        ) -> float:
        moyenne = np.mean(matrix_yield) 
        return moyenne

    # Calcul les scénarios de performances
    def perfomance_tension(
            self,
            vol_tension: float,
            number_of_negotiation_in_period_calcul: int,
            skewness: float | np.ndarray,
            kurtosis: float | np.ndarray,
            value_threshold: float
        ) -> float | np.ndarray:
        """
        Compute the tension scenario performance
        
        Args:
            vol_tension: float La volatilité sous tension calculee a partir de l'
            historique du support.
            number_of_negotiation_in_period_calcul: int le nombre de périodes de 
            négociation durant la période de calcul (si t = RHP alors n= N).
            skewness: float | np.ndarray le coefficient d'asymétrie mesuré à
            partir de la distribution des rendements.
            kurtosis: float | np.ndarray le coefficient d'excès d'aplatissement 
            mesuré à partir de la distribution des rendements.
            value_threshold: float le centile de la loi Normale au seuil.
        
        return:
            float or np.ndarray with the same type than a np.ndarray
            if on of the input are np.ndarray 
        """
        arg_exp_1_1 = (value_threshold**2-1)/(
            6*np.sqrt(number_of_negotiation_in_period_calcul)
        )*skewness
        arg_exp_1_2 = (value_threshold**3 - 3*value_threshold)*kurtosis/(
                24*number_of_negotiation_in_period_calcul
            )
        arg_exp_1_3 = -(2*value_threshold**3 - 5*value_threshold)*skewness**2/(
                36*number_of_negotiation_in_period_calcul
            )
        arg_exp_1 = value_threshold + arg_exp_1_1 + arg_exp_1_2 + arg_exp_1_3
        arg_exp_2 = -vol_tension**2/2

        arg_exp = vol_tension * arg_exp_1 + arg_exp_2
        
        return np.exp(arg_exp)

    def perfomance_favorable_defavorable(
            self,
            number_of_negotiation_in_period_calcul: int,
            mean: float,
            volatility: float | np.ndarray,
            skewness: float | np.ndarray,
            kurtosis: float | np.ndarray,
            value_threshold: float
        ) -> float | np.ndarray:
        """
        Compute the fovarable or defavorable scenario performance
        Only the value of the threshold is modified.
        
        Args:
            mean: float sous jacent en t=1
            number_of_negotiation_in_period_calcul: int le nombre de périodes de 
            négociation durant la période de calcul (si t = RHP alors n= N).
            volatility: float | np.ndarray la volatilité mesurée à partir de la
            distribution des rendements.
            skewness: float | np.ndarray le coefficient d'asymétrie mesuré à
            partir de la distribution des rendements.
            kurtosis: float | np.ndarray le coefficient d'excès d'aplatissement 
            mesuré à partir de la distribution des rendements.
            value_threshold: float le centile de la loi Normale au seuil.
        
        return:
            float or np.ndarray with the same type than a np.ndarray
            if on of the input are np.ndarray
        """
        arg_exp_1_1 = (value_threshold**2-1)/(
            6*np.sqrt(number_of_negotiation_in_period_calcul)
        ) * skewness
        arg_exp_1_2 = (value_threshold**3 - 3*value_threshold)*kurtosis/(
            24*number_of_negotiation_in_period_calcul
        )
        arg_exp_1_3 = -(2*value_threshold**3 - 5*value_threshold)*skewness**2/(
            36*number_of_negotiation_in_period_calcul
        )
        arg_exp_1 = value_threshold + arg_exp_1_1 + arg_exp_1_2 + arg_exp_1_3
        arg_exp_2 = -number_of_negotiation_in_period_calcul * volatility**2/2
        arg_exp_3 = mean * number_of_negotiation_in_period_calcul
        arg_exp = volatility * np.sqrt(number_of_negotiation_in_period_calcul)*(
            arg_exp_1) + arg_exp_2 + arg_exp_3
        
        return np.exp(arg_exp)

    def perfomance_middle(
            self,
            mean: float,
            number_of_negotiation_in_period_calcul: int,
            volatility: float | np.ndarray,
            skewness: float | np.ndarray
        ) -> float | np.ndarray:
        """
        Compute the middle scenario performance
        Only the value of the threshold is modified.
        
        Args:
            mean: float sous jacent en t=1
            number_of_negotiation_in_period_calcul: int le nombre de périodes de 
            négociation durant la période de calcul (si t = RHP alors n= N).
            volatility: float | np.ndarray la volatilité mesurée à partir de la
            distribution des rendements.
            skewness: float | np.ndarray le coefficient d'asymétrie mesuré à
            partir de la distribution des rendements.
        return:
            float or np.ndarray with the same type than a np.ndarray
            if on of the input are np.ndarray
        """
        arg_exp_1 = mean * number_of_negotiation_in_period_calcul
        arg_exp_2 = -volatility * skewness/6
        arg_exp_3 = -number_of_negotiation_in_period_calcul * volatility**2/2

        mnt_exp = np.exp(arg_exp_1 + arg_exp_2 + arg_exp_3)
        return mnt_exp

    def scenario_performance_cat_2(
            self,
        ) -> tuple: 
        """
        The matrix : matrix_index as the shape of 
        (number of simulation x number of Projection).
        
        Args : 
        self:
            matrix_yield: 1 ndarray of the yield considerated in the lapse of studied
            dict_parametre_scenario_performance: dict
                rhp: str can be choose between :
                    '1 ans',
                    'sup 1 ans'.
        Indicates the year of projection. It will be impactful to the parametre
        used to create the subinterval and the quantil.
                data_frequency: str can be choose between :
                    'daily prices',
                    'weekly prices',
                    'monthly prices'.
                category: bool that indicates the category of the action.
        This parametre changes the way to calculate the tension volatility
                dict_quantil:
                    defav: float threshold used to calculate the defavorable performance scenario
                fav: float threshold used to calculate the favorable performance scenario
            
        Returns: 
            tuple of float that contains the 4 performances:
            fav_perf, defav_perf, middle_perf, tension_perf
        """
        matrix_yield = self.get_matrix_yield()
        # extraction of parametre
        rhp = self.get_rhp()
        data_frequency = self.get_data_frequency()
        category = self.get_category()
        dict_quantil = self.get_dict_parametre_quantil_perf()
        threshold_trust_for_defav_perf = dict_quantil["defav"]
        threshold_trust_for_fav_perf = dict_quantil["fav"]

        freq_of_data = self.get_dict_frequence_of_data()[data_frequency.split(" ")[0]]
        number_of_negotiation_in_period_calcul = freq_of_data * rhp

        # threshold of performance
        value_defav_for_threshold = norm.ppf(threshold_trust_for_defav_perf)
        value_fav_for_threshold = norm.ppf(threshold_trust_for_fav_perf)
        quantil_tension = self.get_quantil_to_calculate_tension_performance(rhp=rhp)
        value_tension_for_threshold = norm.ppf(1-quantil_tension)
        
        # statistic
        mean = self.compute_mean(matrix_yield=matrix_yield)
        volatility = self.compute_volatility(matrix_yield=matrix_yield)
        skewness = self.compute_skewness(matrix_yield=matrix_yield)
        kurtosis = self.compute_kurtosis(matrix_yield=matrix_yield)
        tension_volatility = self.compute_tension_vol(
            log_yield_data=matrix_yield,
            rhp=rhp,
            data_frequency=data_frequency,
            category=category,
            number_of_negotiation_in_period_calcul=\
                number_of_negotiation_in_period_calcul
        )

        # performance
        tension_perf = self.perfomance_tension(
            vol_tension=tension_volatility,
            number_of_negotiation_in_period_calcul=number_of_negotiation_in_period_calcul,
            skewness=skewness,
            kurtosis=kurtosis,
            value_threshold=value_tension_for_threshold
        )
        fav_perf = self.perfomance_favorable_defavorable(
            number_of_negotiation_in_period_calcul=number_of_negotiation_in_period_calcul,
            mean=mean,
            volatility=volatility,
            skewness=skewness,
            kurtosis=kurtosis,
            value_threshold=value_fav_for_threshold
        )
        defav_perf = self.perfomance_favorable_defavorable(
            number_of_negotiation_in_period_calcul=number_of_negotiation_in_period_calcul,
            mean=mean,
            volatility=volatility,
            skewness=skewness,
            kurtosis=kurtosis,
            value_threshold=value_defav_for_threshold
        )
        middle_perf = self.perfomance_middle(
            number_of_negotiation_in_period_calcul=number_of_negotiation_in_period_calcul,
            mean=mean,
            volatility=volatility,
            skewness=skewness
        )

        return fav_perf, defav_perf, middle_perf, tension_perf
    
    def scenario_performance_net_cat_2(self) -> None:
        message_error = f"Method not inplemented : {NotImplementedError}"
        logging.error(message_error)
        ValueError(message_error)
