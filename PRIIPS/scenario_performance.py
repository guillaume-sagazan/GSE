from abc import ABC, abstractmethod
import numpy as np
import logging

class ScenarioPerformance(ABC):

    def __init__(
            self,
            matrix_yield: np.ndarray | dict,
            dict_parametre_scenario_performance: dict,
            is_logging_activate: bool = False
        ) -> None:
        self.matrix_yield = matrix_yield
        self.dict_parametre_scenario_performance = dict_parametre_scenario_performance
        if is_logging_activate:
            self.initiate_logging()
    
    def initiate_logging(self):
        logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    filename='PRIIPS\log scenario performance.log')

    @abstractmethod
    def get_category_mapping_dictionary_for_funct_performance(self) -> None:
        """
        all the test function have to be add the dictionary and including the terme test
        performance_funct is a mandatory key to have and must be associated with 
        the function that calculate scenario_performance

        This funct have to initialize the argument mapping_funct
        Return:
            None
        """
        pass
    
    def associate_category_to_type_of_performance(self) -> None:
        dict_map = self.mapping_funct
        if "performance_funct_net" in dict_map: # Optional
            self.performance_funct_net = dict_map["performance_funct_net"]
        if "performance_funct" in dict_map: # Mandatory
            self.performance_funct = dict_map["performance_funct"]
        else:
            message_error = "The key performance_funct is not in mapping_funct.\n\
                Verify that the methode get_category_mapping_dictionary_for_funct_performance \
                    contains the performance_funct key word\n \
                The object can not be created"
            logging.error(message_error)
            raise ValueError(message_error)
        
    def verify_input_for_creation_of_object(self) -> None:
        """
        Execute all the method inside the dictionary that contains the word test.
        If an error is find. An error is rised.
        """
        have_any_test_funct_been_implemented = False
        mapping_funct = self.mapping_funct
        for key in mapping_funct:
            if "test" in key:
                mapping_funct[key]()
                have_any_test_funct_been_implemented = True
        if not have_any_test_funct_been_implemented:
            message_warning = "\
                None of the test has been implemented or initialize in the mapping_funct"
            logging.warning(message_warning)
    
    def verify_that_rhp_correct_value(self) -> None:
        rhp = self.get_rhp()
        input_matrix = self.matrix_yield
        list_shape_projection = []
        if isinstance(input_matrix, dict):
            for key, matrix in input_matrix.items():
                shape_matrix = np.shape(matrix)[-1]
                list_shape_projection.append(shape_matrix)
        if isinstance(input_matrix, np.ndarray):
            shape_matrix = np.shape(input_matrix)[-1]
            list_shape_projection.append(shape_matrix)

        if any(shape_tested < rhp
               for shape_tested in list_shape_projection):
            message_error = f"The shape of the matrix does not fit with the rhp\
                that is equal to {rhp} and the matrix is {list_shape_projection}."
            logging.error(message_error)
            raise ValueError(message_error)
    
    def get_yield_mean(
            self,
            value_perf: float | np.ndarray | tuple
        ) -> float | np.ndarray:
        mnt_init_investement = self.get_mnt_initial_investement()
        rhp = self.get_rhp()
        yield_mean = (np.array(value_perf)/mnt_init_investement) ** (1/rhp) - 1
        return yield_mean
    
    def get_category(self) -> int:
        key = "category"
        if key in self.dict_parametre_scenario_performance:
            return self.dict_parametre_scenario_performance[key]
        else:
            message_error = f"Their is no '{key}' in dict_parametre_scenario_performance.\
                The key in dict_parametre_scenario_performance are \
                    {self.dict_parametre_scenario_performance.keys()}."
            logging.error(message_error)
            raise ValueError(message_error)
    
    def get_rhp(self) -> int:
        if "rhp" in self.dict_parametre_scenario_performance:
            return self.dict_parametre_scenario_performance["rhp"]
        else:
            message_error = f"Their is no 'rhp' in dict_parametre_scenario_performance.\
                The key in dict_parametre_scenario_performance are \
                    {self.dict_parametre_scenario_performance.keys()}."
            logging.error(message_error)
            raise ValueError(message_error)
    
    def set_rhp(
            self,
            new_rhp: float
            ) -> None:
        """Alter dict_parametre_scenario_performance to modify or add the rhp values"""
        self.dict_parametre_scenario_performance["rhp"] = new_rhp
        self.verify_that_rhp_correct_value()

    def get_data_frequency(self):
        if "data_frequency" in self.dict_parametre_scenario_performance:
            return self.dict_parametre_scenario_performance["data_frequency"]
        else:
            message_error = f"Their is no 'data_frequency' in \
                dict_parametre_scenario_performance. The key in \
                dict_parametre_scenario_performance are \
                {self.dict_parametre_scenario_performance.keys()}."
            logging.error(message_error)
            raise ValueError(message_error)
    
    def get_mnt_initial_investement(self):
        key_mnt_init = "mnt_investment_initial"
        if key_mnt_init in self.dict_parametre_scenario_performance:
            return self.dict_parametre_scenario_performance[key_mnt_init]
        else:
            message_error = f"{key_mnt_init} is not in dictionary of the input"
            logging.error(message_error)
            raise ValueError(message_error)
    
    def set_mnt_initial_investement(
            self,
            mnt_investment_initial: float
            ) -> None:
        """Alter dict_parametre_scenario_performance to modify or add the rhp values"""
        self.dict_parametre_scenario_performance["mnt_investment_initial"] =\
            mnt_investment_initial
        
    
    
