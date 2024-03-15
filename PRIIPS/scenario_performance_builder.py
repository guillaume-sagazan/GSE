from PRIIPS.scenario_performance_1 import ScenarioPerformance1
from PRIIPS.scenario_performance_2 import ScenarioPerformance2
from PRIIPS.scenario_performance_3 import ScenarioPerformance3
from PRIIPS.scenario_performance_4 import ScenarioPerformance4
import numpy as np
import logging

class ScenarioPerformanceManager:

    def __init__(self) -> None:
        pass

    def get_mapping_creation_class(self):
        """
        Associate the goodclass in function of the category
        """
        dict_mapping_class = {
            1: self.initiate_perf_scen_1,
            2: self.initiate_perf_scen_2,
            3: self.initiate_perf_scen_3,
            4: self.initiate_perf_scen_4
        }
        return dict_mapping_class

    def get_rates_matrix(
            self,
            dict_matrix_yield,
            key_tested: str = ''
            ):
        dict_data_rates = dict_matrix_yield
        if len(dict_data_rates) == 1 or key_tested == '':
            return list(dict_data_rates.values())[0]
        else:
            list_value = []
            for key, value in dict_data_rates.items():
                if key == key_tested:
                    list_value.append(True)
                    return value
        if list_value:
            message_error = f"No {key_tested} are present in {dict_data_rates.keys()}."
            raise ValueError(message_error)
    
    @staticmethod
    def check_category(
            dict_parametre: dict
        ) -> float:
        """
        Return False if the category is not a int or including
        in (1, 2, 3, 4) otherwise True is throw
        """
        bool_check = False
        type_category = dict_parametre["category"]
        if type_category in (1, 2, 3, 4):
            bool_check = True
        return bool_check

    @staticmethod
    def adapt_parametre_to_check_dict_taxes(
            dict_parametre: dict
        ) -> dict:
        """Give an default value for the item of dict_taxes"""
        list_taxe_rates = [
            "rate_chargement_euro",
            "rate_admission_fees",
            "rate_management_fees_on_outstandings",
            "rate_fees_on_global_actif",
            "cost_of_transaction",
            "other_cost",
            "cost_fees_linked_to_results",
            "cost_linked_to_incentive_commissions"
            ]
        if "dict_taxes" in dict_parametre:
            dict_taxes = dict_parametre["dict_taxes"]
        else:
            dict_taxes = {}
        for taxe_rates in list_taxe_rates:
            if not taxe_rates in dict_taxes:
                dict_taxes[taxe_rates] = 0
        dict_parametre["dict_taxes"] = dict_taxes
        return dict_parametre

    def adapt_parametre(
            self,
            dict_parametre: dict
        ) -> dict:
        dict_parametre_check_taxes = \
            self.adapt_parametre_to_check_dict_taxes(dict_parametre)
        return dict_parametre_check_taxes
    
    def initiate_perf_scen_1(
            self,
            dict_matrix_yield: np.ndarray | dict,
            dict_of_parametre: dict,
            is_logging_activate: bool = False
        ) -> ScenarioPerformance1:
        matrix_yield_fav = self.get_rates_matrix(dict_matrix_yield)
        object_scen_perf = ScenarioPerformance1(
            matrix_yield_favorable = matrix_yield_fav,
            matrix_yield_defavorable = matrix_yield_fav,
            dict_parametre_scenario_performance = dict_of_parametre,
            is_logging_activate = is_logging_activate
                )
        return object_scen_perf
    
    def initiate_perf_scen_2(
            self,
            dict_matrix_yield: np.ndarray | dict,
            dict_of_parametre: dict,
            is_logging_activate: bool = False
        ) -> ScenarioPerformance1:
        matrix_yield = self.get_rates_matrix(dict_matrix_yield)
        object_scen_perf = ScenarioPerformance2(
            matrix_yield = matrix_yield,
            dict_parametre_scenario_performance=dict_of_parametre,
            is_logging_activate = is_logging_activate
                )
        return object_scen_perf

    def initiate_perf_scen_3(
            self,
            dict_matrix_yield: np.ndarray | dict,
            dict_of_parametre: dict,
            is_logging_activate: bool = False
        ) -> ScenarioPerformance1:
        matrix_yield_fav = self.get_rates_matrix(dict_matrix_yield)
        object_scen_perf = ScenarioPerformance3(
            matrix_yield_favorable = matrix_yield_fav,
            matrix_yield_defavorable = matrix_yield_fav,
            dict_parametre_scenario_performance = dict_of_parametre,
            is_logging_activate = is_logging_activate
                )
        return object_scen_perf
    
    def initiate_perf_scen_4(
            self,
            dict_matrix_yield: np.ndarray | dict,
            dict_of_parametre: dict,
            is_logging_activate: bool = False
        ) -> ScenarioPerformance1:
        matrix_yield_fav = self.get_rates_matrix(
            dict_matrix_yield,
            "favorable")
        matrix_yield_defav = self.get_rates_matrix(
            dict_matrix_yield,
            "defavorable")
        object_scen_perf = ScenarioPerformance4(
            matrix_yield_favorable = matrix_yield_fav,
            matrix_yield_defavorable = matrix_yield_defav,
            dict_parametre_scenario_performance = dict_of_parametre,
            is_logging_activate = is_logging_activate
                )
        return object_scen_perf

class ScenarioPerformanceBuilder:

    @classmethod
    def create_object(
        cls,
        dict_matrix_yield: np.ndarray | dict,
        dict_of_parametre: dict,
        is_logging_activate: bool = False
    ):
        """
        Build an object in function of the parametre category.
        Args:
            dict_matrix_yield: dictionary that contains the array of
        the rates. choose of key (favorable, defavorable)
        Return:
            object ScenarioPerformance like
        """
        manager_sp = ScenarioPerformanceManager()
        dict_of_parametre = manager_sp.adapt_parametre(dict_of_parametre)
        if "category" in dict_of_parametre and \
                manager_sp.check_category(dict_of_parametre):
            type_category = dict_of_parametre["category"]
            dict_mapping_class = manager_sp.get_mapping_creation_class()
            object_scen_perf = dict_mapping_class[type_category](
                dict_matrix_yield, dict_of_parametre, is_logging_activate)
        else:
            message_error = "\
                The object can not be created because the argument\
                category is missing form the dict_of_parametre or don't\
                belongs to (1, 2, 3, 4)."
            logging.error(message_error)
            raise ValueError(message_error)
        return object_scen_perf
    
    

    


