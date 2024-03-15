from PRIIPS.scenario_performance_4 import ScenarioPerformance4
import numpy as np
import logging

class ScenarioPerformance3(ScenarioPerformance4):

    def __init__(
                self,
                matrix_yield_favorable: np.ndarray,
                matrix_yield_defavorable: np.ndarray,
                dict_parametre_scenario_performance: dict,
                is_logging_activate: bool = False
                ) -> None:
          super().__init__(
                matrix_yield_favorable,
                matrix_yield_defavorable,
                dict_parametre_scenario_performance,
                is_logging_activate=is_logging_activate
                )

    def get_category_mapping_dictionary_for_funct_performance(self) -> None:
            dict_map = {
                "test_function": self.verify_if_all_input_of_cat_4_is_present_in_dict,
                "test_rhp_valide": self.verify_that_rhp_correct_value,
                "performance_funct": self.scenario_performance_cat_3,
                "get_performance_net": self.scenario_performance_net_cat_3,
            }
            self.mapping_funct = dict_map
    
    def get_tension_quantil(self):
        """Acording to the regulation the quantil choosen depend on the rhp"""
        rhp = self.get_rhp()
        if rhp == 1:
            tension_quantil = 99
        elif rhp > 1:
            tension_quantil = 95
        else:
            message_error = f"The rhp is wrong. \
                It should be numerical and higer or egal to 1.\
                The current value is {rhp}."
            logging.error(message_error)
            raise ValueError(message_error)
        return tension_quantil
    
    def get_dict_parametre_quantil_perf(self) -> dict:
        """
        Initialize the dictionary of parametre use to calculate the 
        performance scenario. 
        Update after the reform 2021 
        """
        if "dict_quantil" in self.dict_parametre_scenario_performance or \
            (isinstance(self.dict_parametre_scenario_performance["dict_quantil"], dict) and\
            self.dict_parametre_scenario_performance["dict_quantil"] != {}) :
            dict_parametrage = self.dict_parametre_scenario_performance["dict_quantil"]
        else:
            dict_parametrage = {
                'fav': 95,
                'defav': 10,
                'middle': 5,
                "tension": self.get_tension_quantil()
            }
        return dict_parametrage
    
    def performance_scenario_tension(
            self,
            matrix_rates: np.ndarray
        ) -> float:
        # Brut
        dict_parametre_quantil = self.get_dict_parametre_quantil_perf()
        mnt_investment_initial = self.get_mnt_initial_investement()
        value_tension_for_threshold = np.percentile(
            matrix_rates, dict_parametre_quantil["tension"])
        tension_perf = (value_tension_for_threshold + 1) * mnt_investment_initial
        return tension_perf
    
    def scenario_performance_cat_3(
            self,
        ) -> tuple:
        """            
        Extract and calculate the favorable, defavorable, middle and tension scenario for the
        category 3.
        The matrix : matrix_index as the shape of 
        (number of simulation x number of Projection).
        
        Args : 
        self:
            matrix_yield: 1 ndarray of the yield considerated in the lapse of studied
            dict_parametre_scenario_performance: dict
                rhp: int 1 or higher.
        Indicates the year of projection. It will be impactful to the calcul in the
        tension yield. The rate_chargement_euro will be add to the yield
        for the RHP of 1.
                type_of_support: str can be choose between (Monosupport or Multisupport).
        Indicates if the rate_chargement_euro should be add to the yield.
                mnt_investment_initial: float amount of investment initial.
        The reforme imposed by the reform PRIIPS.
                TMGA: float | int indicates the value of the tmg annuel
                TMF: float | int indicates the value of the tmg 
            
        Returns: 
            tuple of float that contains the 4 performances:
            fav_perf, defav_perf, middle_perf, tension_perf
        """
        matrix_rates_fav = self.get_matrix_rates_fav()
        matrix_net_yield_fav = self.get_served_rates_net_matrix(matrix_rates_fav)
        matrix_cumulated_fav = self.get_cumul_rate_matrix(matrix_net_yield_fav)

        matrix_rates_defav = self.get_matrix_rates_defav()
        matrix_net_yield_defav = self.get_served_rates_net_matrix(
            matrix_rates_defav)
        matrix_cumulated_defav = self.get_cumul_rate_matrix(matrix_net_yield_defav)

        fav_perf = self.performance_scenario_fav(matrix_cumulated_fav)
        defav_perf = self.performance_scenario_defav(matrix_cumulated_defav)
        middle_perf = self.performance_scenario_middle(matrix_cumulated_defav)
        tension_perf = self.performance_scenario_tension(matrix_cumulated_defav)

        return fav_perf, defav_perf, middle_perf, tension_perf
    
    def scenario_performance_net_cat_3(
            self,
        ) -> tuple:
        """
        SHOULD NOT BE USED
        Perform the calcul of performance scenario after deduction of cost. 
        The function use to compute the performance come from 'performance_funct'.

        Args:
            dict_taxes: dict that contains the fees:
                        rate_admission_fees: float. Impact the calcul of the yield.
                        rate_management_fees_on_outstandings: float. Impact the calcul of the yield.
        Returns: 
            tuple of float that contains the 4 performances after deduction of cost:
            fav_perf_net, defav_perf_net, middle_perf_net, tension_perf_net
        """
        fav_perf, defav_perf, middle_perf, tension_perf = \
            self.performance_funct()
        fav_perf_net = self.get_deduction_performance_scenario_fav(
            fav_perf=fav_perf)
        defav_perf_net = self.get_deduction_performance_scenario_defav(
            defav_perf=defav_perf)
        middle_perf_net = self.get_deduction_performance_scenario_middle(
            middle_perf=middle_perf)
        tension_perf_net = self.get_deduction_performance_scenario_tension(
            tension_perf=tension_perf)
        return fav_perf_net, defav_perf_net, middle_perf_net, tension_perf_net