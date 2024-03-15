import unittest
from ModelTaux_RR import *
import os
import pandas as pd
import numpy as np

class TestModelTaux_RR(unittest.TestCase):

    def setUp(self):
        # parametre de teste
        lambda_t = 0.0355279589
        mu_t = -0.02931970
        sigma = 0.0000040031
        valeur_initial = -0.0032

        nombre_simulation = 1000
        annee_projection = 40
        max_maturity = 40
        freq_of_data = "monthly"
        dt = 1/12

        name_file = "data_taux_court.xlsx"

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
            max_maturity,
            annee_projection,
            nombre_simulation
            )

        cdt_rr = Vasicek_RR(
            parametre_model = dict_parametre_model,
            path_data = name_file,
            parametre_simulation = dict_parametre_simulation,
            random_matrix=matrix_random
        )
        self.cdt_rr = cdt_rr

    def test_get_vector_full_a_value(self):

        vector = self.cdt_rr.get_vector_full_a_value(
            1,
            self.cdt_rr.parametre_simulation["number_of_simulation"]
            )
        self.assertEqual(vector,
                         np.array(
                            [1] * self.cdt_rr.parametre_simulation["number_of_simulation"]
                            )
                        )
         
    def test_simulate_rates(self):

        simulation = self.cdt_rr.simulate_rates()
        self.simulation = simulation
    
    def test_get_simulation_rates(self):
        self.assertAlmostEqual(
            self.cdt_rr.get_simulation_rates(1, 1),
            0.9975308113
            )
    
    def test_get_prices_rr(self):
        initial_value = self.cdt_rr.parametre_model["Initial value"]
        self.assertAlmostEqual(
            self.cdt_rr.get_prices_zc(1, initial_value),
            1.003665896
            )