
import pandas as pd
import numpy as np
from CourbeDesTaux import *
from ModelTaux import *
from scipy.stats import norm
import logging
import matplotlib.pyplot as plt
from typing import Union, List, Optional


class Controle:

    def __init__(self,
                 rates_model: ModelTaux,
                 max_maturity_test: float | int,
                 conf_level: float) -> None:
        self.rates_model = rates_model
        self.max_maturity_test = max_maturity_test
        self.conf_level = conf_level
        self._calcul_quantil_normal_law_bilateral()
    
    def get_rates_model(self):
        return self.rates_model
    
    def get_max_maturity_test(self):
        return self.max_maturity_test
    
    def get_conf_level(self):
        return self.conf_level
    
    def _calcul_quantil_normal_law_bilateral(self):
        try:
            if self.conf_level <= 0 and self.conf_level >= 1:
                message_error = "The value of conf_level is incorrect"
                logging.error(message_error)
                raise ValueError(message_error)
            proba =(100-(((1-self.conf_level)*100)/2))/100 
            quantil = norm.ppf(proba)
            self.quantil = quantil
        except Exception as e:
            message_error = f"{e}"
            logging.error(message_error)
            raise e
        
    def martingalite_test_deflator(self):
        """The deflator are test to verify that on average we find the zero coupon prices
            Args:
                self:
                    rates_model
            Returns:
                None
                    Update self and add the attribut deflator_test
        """
        try:
            mean_deflator =  np.mean(self.rates_model.deflator, axis=1)
            maturity = np.arange(1, self.max_maturity_test+1, 1)
            rates_deflator_test = mean_deflator ** (-1/maturity) - 1
            rates_deflator_ref = self.rates_model.cdt.zc_rates_obs[maturity-1]
            rates_deflator_refative_gap = rates_deflator_test/rates_deflator_ref-1

            nbr_simu = np.shape(self.rates_model.deflator)[1]
            interval_conf = self.quantil * np.std(self.rates_model.deflator, axis=1)/np.sqrt(nbr_simu)
            rates_deflator_upper_bound = rates_deflator_ref + interval_conf
            rates_deflator_lower_bound = rates_deflator_ref - interval_conf
            
            self.maturity = maturity
            self.rates_deflator_test = rates_deflator_test
            self.rates_deflator_ref = rates_deflator_ref
            self.rates_deflator_refative_gap = rates_deflator_refative_gap
            self.rates_deflator_upper_bound = rates_deflator_upper_bound
            self.rates_deflator_lower_bound = rates_deflator_lower_bound
        except Exception as e:
            message_error = "Error in the test deflator has been raise \n:"
            message_error += f"{e}"
            logging.error(message_error)
            raise e
    
    def display_martingality_test_graph_deflator(self):
        """Show graph that corresponds to martingality test.
        the method 'martingalite_test_deflator' have to be call before this one
        """
        try:
            if not(all(attribut in dir(self.cdt) for attribut in ['maturity',
                                                                'rates_deflator_test',
                                                                'rates_deflator_ref',
                                                                'rates_deflator_upper_bound',
                                                                'rates_deflator_lower_bound'])):
                    message_error = "the attribut of CourbeDesTaux must contains"
                    message_error += "'maturity', 'rates_deflator_test', 'rates_deflator_ref', 'rates_deflator_upper_bound'"
                    message_error += ", 'rates_deflator_lower_bound'"
                    logging.error(message_error)
            plt.figure(figsize=(8,8))

            plt.plot(self.maturity, self.rates_deflator_test, label="deflator test")
            plt.plot(self.maturity, self.rates_deflator_ref, label="deflator reference")
            plt.plot(self.maturity, self.rates_deflator_upper_bound, label="deflator upper bound")
            plt.plot(self.maturity, self.rates_deflator_lower_bound, label="deflator lower bound")

            plt.title("Test of Martingality with deflator")
            plt.xlabel('Martingality')
            plt.ylabel('deflator')

            plt.legend()

            plt.show()
        except Exception as e:
            message_error = "Error in the test deflator has been raise \n:"
            message_error += f"{e}"
            logging.error(message_error)
        
    

