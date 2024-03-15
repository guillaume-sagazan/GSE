import unittest
from ModelActions_RR import *
import os
import pandas as pd
import numpy as np

class TestModelActions_RR(unittest.TestCase):

    def setUp(self):
        # parametre de teste
        parametre_model = [
        0.00412889777566318, 2.0102719145309, 0.00101047902388654, 
        1.55860487159906E-09, 0.0030588304634895, 0]

        name_data = "data_actions_cours.xlsx"

        initial_value = 100
        number_of_simulation = 1000
        freq_of_time = "monthly"
        step_of_time = 1/12
        year_of_projection = 40

        dict_parametre_simulation = {
            "initial value": initial_value,
            "number of simulation": number_of_simulation,
            "frequence of data": freq_of_time,
            "year of projection": year_of_projection
        }

        matrix_random = np.random.randn(
            int(year_of_projection / step_of_time) + 1,
            number_of_simulation
        )

        m_act_calibre = Merton_RR(
            parametre_model=parametre_model,
            parametre_simulation=dict_parametre_simulation,
            random_matrix=matrix_random,
            path_data=name_data
        )

        m_act_empty = Merton_RR(
            parametre_simulation=dict_parametre_simulation,
            random_matrix=matrix_random,
            path_data=name_data
        )

        self.m_act_calibre = m_act_calibre
        self.m_act_empty = m_act_empty
        self.parametre_model = parametre_model
    
    def test_get_list_of_theoretical_moment(self):
        
        liste_moment_theoric = self.m_act_empty.get_list_of_theoretical_moment(
            parametre_model=parametre_model,
            h=1
            )
        list_reference = [
            0.00412787111786845, 0.0030608830875267, 2.07413353352473E-09,
            2.09586842838137E-12, 2.11783108372558E-15, 2.14002388626505E-18
            ]
        self.assertAlmostEquals(liste_moment_theoric, list_reference)
    
    def test_get_log_yield(self):
        list_data_to_test = [4147.5,4303.92,4032.05,4229.85,4442.84,4313.69,4609.26,4377.87]
        list_log_yield = self.m_act_calibre.get_log_yield(list_data_to_test)
        list_reference = [
            0.037020492243679, -0.0652513038403123, 0.0478915999506938,
            0.0491272801650078, -0.0295001252800666, 0.0662736371949061, 
            -0.0515050187807388
            ]
        self.assertAlmostEquals(list_log_yield, list_reference)
    
    def test_get_list_of_empiric_moment(self):
        data_import = self.m_act_calibre.import_data()
        data_filter = self.m_act_calibre.filter_data(data_import)
        list_log_yield = self.m_act_calibre.get_log_yield(data_filter)
        list_of_empiric_moment = self.m_act_calibre.get_list_of_empiric_moment(
            list_log_yield
        )
        list_reference = [
            9.84431286488174E-12, 1.38188671052961E-09, 0.00010686554300134, 
            0.0000209856652172327, 2.21818064941008E-06, 2.68484030850204E-07
        ]
        self.assertAlmostEquals(list_of_empiric_moment, list_reference)