from PRIIPS.scenario_performance import ScenarioPerformance
import numpy as np
import re
import logging

class ScenarioPerformance4(ScenarioPerformance):

    def __init__(
            self,
            matrix_yield_favorable: np.ndarray,
            matrix_yield_defavorable: np.ndarray,
            dict_parametre_scenario_performance: dict,
            is_logging_activate: bool = False
        ) -> None:
        super().__init__(
            matrix_yield_favorable,
            dict_parametre_scenario_performance,
            is_logging_activate)
        self.matrix_yield_defavorable = matrix_yield_defavorable
        self.get_category_mapping_dictionary_for_funct_performance()
        self.verify_input_for_creation_of_object()
        self.associate_category_to_type_of_performance()
        self.adapt_dict_parametre_quantil_perf_if_exist()
    
    def get_matrix_rates_fav(self):
        return self.matrix_yield
    
    def get_matrix_rates_defav(self):
        return self.matrix_yield_defavorable
    
    def get_dict_taxes(self):
        if "dict_taxes" in self.dict_parametre_scenario_performance:
            dict_taxes = self.dict_parametre_scenario_performance["dict_taxes"]
        else:
            message_error = "Their is no 'dict_taxes' in the dictionary of parametre"
            logging.error(message_error)
            raise ValueError(message_error)
        return dict_taxes

    def get_management_fees_on_outstanding(self):
        key_dict_taxes = "dict_taxes"
        key_dict_taxes_management_fees = "rate_management_fees_on_outstandings"
        if key_dict_taxes in self.dict_parametre_scenario_performance and \
            key_dict_taxes_management_fees in self.dict_parametre_scenario_performance[key_dict_taxes]:
            return self.dict_parametre_scenario_performance[
                key_dict_taxes][key_dict_taxes_management_fees]
        else:
            message_warning = f"{key_dict_taxes} is not in dictionary of the input or \
                {key_dict_taxes_management_fees} is not in dictionary of taxes"
            logging.warning(message_warning)
            return 0
    
    def get_loading_rates_one_euro(self):
        key_dict_taxes = "dict_taxes"
        key_dict_taxes_loading_rates = "rate_chargement_euro"
        if key_dict_taxes in self.dict_parametre_scenario_performance and \
            key_dict_taxes_loading_rates in self.dict_parametre_scenario_performance[key_dict_taxes]:
            return self.dict_parametre_scenario_performance[
                key_dict_taxes][key_dict_taxes_loading_rates]
        else:
            message_warning = f"{key_dict_taxes} is not in dictionary of the input or \
                {key_dict_taxes_loading_rates} is not in dictionary of taxes"
            logging.warning(message_warning)
            return 0
        
    def get_rate_fees_on_global_actif(self):
        key_dict_taxes = "dict_taxes"
        key_dict_global_fees = "rate_fees_on_global_actif"
        if key_dict_taxes in self.dict_parametre_scenario_performance and \
            key_dict_global_fees in self.dict_parametre_scenario_performance[key_dict_taxes]:
            return self.dict_parametre_scenario_performance[
                key_dict_taxes][key_dict_global_fees]
        else:
            message_warning = f"{key_dict_taxes} is not in dictionary of the input or \
                {key_dict_global_fees} is not in dictionary of taxes."
            logging.warning(message_warning)
            return 0
    
    def get_tmg(self) -> float | int:
        key_dict_contract = "dict_contract"
        if  key_dict_contract in self.dict_parametre_scenario_performance and \
                "TMG" in self.dict_parametre_scenario_performance[key_dict_contract]:
            tmg = self.dict_parametre_scenario_performance[key_dict_contract]["TMG"]
        else:
            tmg = 0
        return tmg
    
    def get_tmga(self) -> float | int:
        key_dict_contract = "dict_contract"
        if  key_dict_contract in self.dict_parametre_scenario_performance and \
                "TMGA" in self.dict_parametre_scenario_performance[key_dict_contract]:
            tmg = self.dict_parametre_scenario_performance[key_dict_contract]["TMGA"]
        else:
            tmg = 0
        return tmg
    
    def get_type_support(self):
        key_dict_contract = "dict_contract"
        if  key_dict_contract in self.dict_parametre_scenario_performance and \
              "type_of_support" in \
                self.dict_parametre_scenario_performance[key_dict_contract]:
            return self.dict_parametre_scenario_performance[
                key_dict_contract]["type_of_support"]
        else:
            message_warning = "\
                type_of_support is not defined in dict_contract \
                    of dict_parametre_scenario_performance"
            logging.warning(message_warning)
            return "Monosupport"
    
    def get_est_cout_fil_du_temps_integre(self):
        if "est_cout_fil_du_temps_integre" in self.dict_parametre_scenario_performance:
            is_cost_fdt_integrated = \
                self.dict_parametre_scenario_performance["est_cout_fil_du_temps_integre"]
        else:
            is_cost_fdt_integrated = False
        return is_cost_fdt_integrated

    def get_admission_fees(self):
        key_dict_taxes = "dict_taxes"
        key_dict_taxes_admission_fees = "rate_admission_fees"
        if key_dict_taxes in self.dict_parametre_scenario_performance and \
            key_dict_taxes_admission_fees in self.dict_parametre_scenario_performance[key_dict_taxes]:
            return self.dict_parametre_scenario_performance[
                key_dict_taxes][key_dict_taxes_admission_fees]
        else:
            message_warning = f"{key_dict_taxes} is not in dictionary of the input or \
                {key_dict_taxes_admission_fees} is not in dictionary of taxes"
            logging.warning(message_warning)
            raise 0
    
    def get_category_mapping_dictionary_for_funct_performance(self) -> None:
        dict_map = {
            "test_function": self.verify_if_all_input_of_cat_4_is_present_in_dict,
            "test_rhp_valide": self.verify_that_rhp_correct_value,
            "performance_funct": self.scenario_performance_cat_4,
            "performance_funct_net": self.scenario_performance_net_cat_4,
        }
        self.mapping_funct = dict_map
    
    def get_dict_parametre_quantil_perf(self) -> dict:
        """
        Initialize the dictionary of parametre use to calculate the 
        performance scenarios. 
        Update after the reform 2021 
        """
        if "dict_quantil" in self.dict_parametre_scenario_performance and \
            isinstance(self.dict_parametre_scenario_performance["dict_quantil"], dict) and\
            self.dict_parametre_scenario_performance["dict_quantil"] != {} :
            dict_parametrage = self.dict_parametre_scenario_performance["dict_quantil"]
        else:
            dict_parametrage = {
                'fav': 99,
                'defav': 10,
                'middle': 50,
            }
        return dict_parametrage
    
    def adapt_dict_parametre_quantil_perf_if_exist(self) -> None:
        """
        If dict_quantil exist in the attribut dict_parametre_scenario_performance
        attribute, check the content to ensure it is correct and 
        make changes if necessary.
        If dict_quantil does not exist, then search in 
        dict_parametre_scenario_performance to see 
        if there are no key that have fav, defav, middle. If they are not 
        present, adapt the dictionnary to normalized it.
        """
        if not "dict_quantil" in self.dict_parametre_scenario_performance:
            dict_quantil_personalised_adapted = {}
            dict_to_check = self.dict_parametre_scenario_performance
        if "dict_quantil" in self.dict_parametre_scenario_performance:
            dict_to_check = self.get_dict_parametre_quantil_perf()
            dict_quantil_personalised_adapted = dict_to_check

        regex_fav = r'\b(\w*)(_{1}|^)(fav)(_{1}|$)(\w*)\b'
        regex_defav = r'\b(\w*)(_{1}|^)(defav)(_{1}|$)(\w*)\b'
        regex_middle = r'\b(\w*)(_{1}|^)(middle)(_{1}|$)(\w*)\b'
        for each_key, values in dict_to_check.items():
            if re.match(regex_fav, each_key) and each_key != "fav":
                dict_quantil_personalised_adapted["fav"] = values
            elif re.match(regex_defav, each_key) and each_key != "defav":
                dict_quantil_personalised_adapted["defav"] = values
            elif re.match(regex_middle, each_key) and each_key != "middle":
                dict_quantil_personalised_adapted["middle"] = values
        if dict_quantil_personalised_adapted == {}:
            message_error = f"At least one quantil argument is lacking. \n\
                At least one quantil must be present or detected"
            logging.error(message_error)
            raise ValueError(message_error)
        self.dict_parametre_scenario_performance["dict_quantil"] = \
            dict_quantil_personalised_adapted
            
    def verify_if_all_input_of_cat_4_is_present_in_dict(self) -> None:
        """
        Verify if all the input of the dictionary dict_parametre_scenario_performance
        are present for the category 2
        Return:
            None: raise an error if any of the key are missing 
        """
        list_of_key_mandatory = [
            "rhp", "mnt_investment_initial", "category"]
        dict_parametre = self.dict_parametre_scenario_performance
        if not all(key_tested in dict_parametre 
               for key_tested in list_of_key_mandatory):
            message_error = f"For the category the element required is \
            {list_of_key_mandatory} but {dict_parametre.keys} is presented."
            logging.error(message_error)
            raise ValueError(message_error)
    
    def extract_submatrix_dim_1(
            self,
            matrix_to_extract: np.ndarray,
            indice_of_shape_to_bootstrap: int,
            indice_boostrap: int
        ) -> np.ndarray:
        if indice_of_shape_to_bootstrap == 0:
            submatrix = matrix_to_extract[indice_boostrap]
        else:
            message_error = "The dimension of indice_of_shape_to_bootstrap doesnot correspond."
            raise ValueError(message_error)
        return submatrix

    def extract_submatrix_dim_2(
            self,
            matrix_to_extract: np.ndarray,
            indice_of_shape_to_bootstrap: int,
            indice_boostrap: int
        ) -> np.ndarray:
        if indice_of_shape_to_bootstrap == 0:
            submatrix = matrix_to_extract[indice_boostrap, :]
        if indice_of_shape_to_bootstrap == 1:
            submatrix = matrix_to_extract[:, indice_boostrap]
        else:
            message_error = "The dimension of indice_of_shape_to_bootstrap doesnot correspond."
            raise ValueError(message_error)
        return submatrix

    def extract_submatrix_dim_3(
            self,
            matrix_to_extract: np.ndarray,
            indice_of_shape_to_bootstrap: int,
            indice_boostrap: int
        ) -> np.ndarray:
        if indice_of_shape_to_bootstrap == 0:
            submatrix = matrix_to_extract[indice_boostrap, :, :]
        if indice_of_shape_to_bootstrap == 1:
            submatrix = matrix_to_extract[:, indice_boostrap, :]
        if indice_of_shape_to_bootstrap == 2:
            submatrix = matrix_to_extract[:, :, indice_boostrap]
        else:
            message_error = "The dimension of indice_of_shape_to_bootstrap doesnot correspond."
            raise ValueError(message_error)
        return submatrix

    def extract_submatrix(
            self,
            matrix_to_extract: np.ndarray,
            indice_of_shape_to_bootstrap: int,
            indice_boostrap: int
        ) -> np.ndarray:
        if len(np.shape(matrix_to_extract)) == 1:
            submatrix = self.extract_submatrix_dim_1(
                matrix_to_extract,
                indice_of_shape_to_bootstrap,
                indice_boostrap)
        if len(np.shape(matrix_to_extract)) == 2:
            submatrix = self.extract_submatrix_dim_2(
                matrix_to_extract,
                indice_of_shape_to_bootstrap,
                indice_boostrap)
        if len(np.shape(matrix_to_extract)) == 3:
            submatrix = self.extract_submatrix_dim_3(
                matrix_to_extract,
                indice_of_shape_to_bootstrap,
                indice_boostrap)
        return submatrix

    def bootstrap(
            self,
            matrix_to_bootstrap: np.ndarray,
            indice_of_shape_to_bootstrap: int,
            number_of_bootstrap: int
        ) -> np.ndarray:
        """
        Return a matrix with same dimension as matrix_to_bootstrap except for the indice
        of the bootstap whose the length will be defined by the parametre number_of_bootstrap.
        On error is rise if the value of the indice_of_shape_to_bootstrap is not conform to
        the input matrix matrix_to_bootstrap and if number_of_boostrap is less than 1.
        """
        if number_of_bootstrap < 1:
            message_error = "The number of bootstap must be higher than 1"
            logging.error(message_error)
            raise ValueError(message_error)
        if not(0 <= indice_of_shape_to_bootstrap and \
            indice_of_shape_to_bootstrap <= len(np.shape(matrix_to_bootstrap))):
            message_error = f"The indice of bootstap must be valide. n\
            The indice is {indice_of_shape_to_bootstrap} but \
            the number of dimension is {len(np.shape(matrix_to_bootstrap))}"
            logging.error(message_error)
            raise ValueError(message_error)
        max_indice_to_boostrap = \
            np.shape(matrix_to_bootstrap)[indice_of_shape_to_bootstrap]
        list_indice_to_bootstrap =  np.random.randint(
            low=0,
            high=max_indice_to_boostrap,
            size=number_of_bootstrap)
        matrix_bootstraped = []
        for indice_bootstrapping in list_indice_to_bootstrap:
            matrix_bootstraped.append(
                self.extract_submatrix(
                    matrix_to_bootstrap,
                    indice_of_shape_to_bootstrap,
                    indice_bootstrapping
                    )
            )
        return np.array(matrix_bootstraped)
    
    def get_init_cost_rates_for_fav_perf(self) -> float:
        """
        Compute the rates that is soustract before the calcul of cumulative rates
        for the favorable scenario 
        """
        dict_taxes = self.get_dict_taxes()
        init_cost_rates = 0
        if "rate_fees_on_global_actif" in dict_taxes:
            init_cost_rates -= self.get_rate_fees_on_global_actif()
        if "rate_chargement_euro" in dict_taxes and \
                self.get_type_support() == "Multisupport":
            init_cost_rates += self.get_loading_rates_one_euro()
        return init_cost_rates

    def get_cout_au_fil_du_temps(self) -> float: 
        """
        Integr d'autre couts si la variable: est_cout_fil_du_temps_integre est True
        """
        dict_taxes = self.get_dict_taxes()
        cost_fdt_rates = 0
        if self.get_est_cout_fil_du_temps_integre():
            for key_cost, value_cost in dict_taxes.items():
                if "cost" in key_cost:   
                    cost_fdt_rates += value_cost
        return cost_fdt_rates
    
    def get_init_cost_rates_for_defav_perf(self) -> float:
        """
        Compute the rates that is soustract before the calcul of cumulative rates
        for the defavorable scenario 
        """
        init_cost_rates = self.get_init_cost_rates_for_fav_perf()
        init_cost_rates += self.get_cout_au_fil_du_temps() # Optional
        return init_cost_rates
    
    def get_served_rates_net_matrix(
            self,
            matrix_yield: np.ndarray,
            rates: float = 0
        ) -> np.ndarray:
        """
        Extract the information from dict_taxe and soustract them to the matrix of yield (rates %)
        Args:
            matrix_yield: np.ndarray Matrix that contains the served rates
            rates: float rate of charges
        Return: One matrix with the same size than self.matrix_yield
        """
        matrix_net_yield = np.copy(matrix_yield)
        matrix_net_yield = matrix_net_yield + rates
        return matrix_net_yield
    
    def get_cumul_rate_matrix(
            self,
            matrix_of_rates_to_cumulate: np.ndarray
        ) -> np.ndarray:
        """
        Construct the matrix of the cumulativ rates for the rhp.
        The matrix matrix_of_rates_to_cumulate have to be an 2-nd array and with the shape
        (number of simulation x year of projection) with year of projection 
        Return: One matrix of dimension (number_of_dimension)
        """
        matrix_cumulate = np.ones(np.shape(matrix_of_rates_to_cumulate)[0])
        rhp = self.get_rhp()
        for year_projected in range(rhp):
            matrix_cumulate = matrix_cumulate * (1 + matrix_of_rates_to_cumulate[:, year_projected])
        matrix_cumulate = matrix_cumulate - 1
        return matrix_cumulate
    
    def get_tmg_tension_rhp_of_1(self) -> float:
        """
        Return float that correspond to the tmg under tension for rhp egal at 1
        """
        rate_tension = self.get_loading_rates_one_euro()
        if self.get_type_support() == "Multisupport":
            rate_tension += self.get_loading_rates_one_euro()
        rate_tension += self.get_tmga() + self.get_tmg()
        return rate_tension

    def get_tmg_tension_rhp_higher_1(self) -> float:
        """
        Return float that correspond to the tmg under tension for rhp higher than 1
        """
        rate_tension = 0
        if self.get_type_support() == "Multisupport":
            rate_tension += self.get_loading_rates_one_euro()
        rate_tension += self.get_tmga() + self.get_tmg()
        return rate_tension
    
    def get_tmg_tension(self):
        """
        Return the yield for tension scenario
        """
        if self.get_rhp() == 1:
            yield_under_tension = self.get_tmg_tension_rhp_of_1()
        else:
            yield_under_tension = self.get_tmg_tension_rhp_higher_1()
        return yield_under_tension

    def performance_scenario_fav(
            self,
            matrix_rates: np.ndarray
        ) -> float:
        # Brut
        dict_parametre_quantil = self.get_dict_parametre_quantil_perf()
        mnt_investment_initial = self.get_mnt_initial_investement()
        value_fav_for_threshold = np.percentile(
            matrix_rates, dict_parametre_quantil["fav"])
        fav_perf = (value_fav_for_threshold + 1) * mnt_investment_initial
        return fav_perf

    def get_deduction_performance_scenario_fav(
            self,
            fav_perf: float
        ) -> float:
        # Net
        rate_net_from_admission_fees = 1 - self.get_admission_fees()
        fav_perf *= rate_net_from_admission_fees
        if self.get_type_support() == "Multisupport":
            rate_global_fees = self.get_rate_fees_on_global_actif()
            fav_perf = fav_perf - rate_global_fees
        return fav_perf

    def performance_scenario_defav(
            self,
            matrix_rates: np.ndarray
        ) -> float:
        # Brut
        dict_parametre_quantil = self.get_dict_parametre_quantil_perf()
        mnt_investment_initial = self.get_mnt_initial_investement()
        value_defav_for_threshold = np.percentile(
            matrix_rates, dict_parametre_quantil["defav"])
        defav_perf = (value_defav_for_threshold + 1) * mnt_investment_initial

        return defav_perf

    def get_deduction_performance_scenario_defav(
            self,
            defav_perf: float
        ) -> float:
        # Net
        rate_net_from_admission_fees = 1 - self.get_admission_fees()
        defav_perf *= rate_net_from_admission_fees
        if self.get_type_support() == "Multisupport":
            rate_global_fees = self.get_rate_fees_on_global_actif()
            defav_perf = defav_perf - rate_global_fees
        return defav_perf

    def performance_scenario_middle(
            self,
            matrix_rates: np.ndarray
        ) -> float:
        # Brut
        dict_parametre_quantil = self.get_dict_parametre_quantil_perf()
        mnt_investment_initial = self.get_mnt_initial_investement()
        value_middle_for_threshold = np.percentile(
            matrix_rates, dict_parametre_quantil["middle"])
        middle_perf = (value_middle_for_threshold + 1) * mnt_investment_initial
        return middle_perf

    def get_deduction_performance_scenario_middle(
            self,
            middle_perf: float
        ) -> float:
        # Net
        rate_net_from_admission_fees = 1 - self.get_admission_fees()
        middle_perf *= rate_net_from_admission_fees
        if self.get_type_support() == "Multisupport":
            rate_global_fees = self.get_rate_fees_on_global_actif()
            middle_perf = middle_perf - rate_global_fees
        return middle_perf
    
    def performance_scenario_tension(self) -> float:
        # Brut
        yield_under_tension = self.get_tmg_tension()
        mnt_investment_initial = self.get_mnt_initial_investement()
        tension_perf = mnt_investment_initial * (1 + yield_under_tension)
        return tension_perf
    
    def get_deduction_performance_scenario_tension(
            self,
            tension_perf: float
        ) -> float:
        # Net
        mnt_investment_initial = self.get_mnt_initial_investement()
        rate_net_from_admission_fees = self.get_admission_fees()
        rate_management_fees = self.get_management_fees_on_outstanding()
        cost_to_deduct = mnt_investment_initial*(rate_net_from_admission_fees + rate_management_fees)
        tension_perf_net = tension_perf - cost_to_deduct
        tension_perf_net = tension_perf_net * (1-rate_management_fees)** (self.get_rhp() - 1)
        return tension_perf_net
     
    def scenario_performance_cat_4(
            self,
        ) -> tuple:
        """            
        Extract and calculate the favorable, defavorable, middle and tension scenario for the
        category 4 that is euro found.
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
                dict_taxes: dict that contains the fees:
                    rate_chargement_euro: float.
        It will be impactful to the yield calculus when Multisupport or rhp is egal to 1 is choose.
                    rate_admission_fees: float. Impact the calcul of the yield.
                    rate_management_fees_on_outstandings: float. Impact the calcul of the yield.
            
        Returns: 
            tuple of float that contains the 4 performances:
            fav_perf, defav_perf, middle_perf, tension_perf
        """
        matrix_rates_fav = self.get_matrix_rates_fav()
        rates_fav = self.get_init_cost_rates_for_fav_perf()
        matrix_net_yield_fav = self.get_served_rates_net_matrix(
            matrix_rates_fav, rates=rates_fav)
        matrix_cumulated_fav = self.get_cumul_rate_matrix(matrix_net_yield_fav)

        matrix_rates_defav = self.get_matrix_rates_defav()
        rates_defav = self.get_init_cost_rates_for_defav_perf()
        matrix_net_yield_defav = self.get_served_rates_net_matrix(
            matrix_rates_defav, rates=rates_defav)
        matrix_cumulated_defav = self.get_cumul_rate_matrix(matrix_net_yield_defav)

        fav_perf = self.performance_scenario_fav(matrix_cumulated_fav)
        defav_perf = self.performance_scenario_defav(matrix_cumulated_defav)
        middle_perf = self.performance_scenario_middle(matrix_cumulated_defav)
        tension_perf = self.performance_scenario_tension()

        return fav_perf, defav_perf, middle_perf, tension_perf

    def scenario_performance_net_cat_4(
            self,
        ) -> tuple:
        """
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
        
