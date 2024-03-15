import unittest
from ModelTaux import *
import os
import pandas as pd
import numpy as np


class TestModelTaux(unittest.TestCase):

    def setUp(self):
        # parametre - Path
        path = os.getcwd()
        suffix_files_test = "//test files//"
        path_test_files = (path + suffix_files_test).replace('//', '\\')
        path_test_files_initialisation = (path_test_files + "initialisation//").replace('//', '\\')
        pzc = pd.read_csv(path_test_files_initialisation + "PZC30092021" + ".csv", decimal=',', sep=';', header=None)
        maturity_pzc = np.arange(1, len(pzc)+1, 1)
        volatilities_nom = pd.read_csv(path_test_files_initialisation + "VolSwaptionsN30092021" + ".csv", decimal=',', sep=';', 
                                       header=None)
        volatilities_nom = np.array(volatilities_nom)/100
        volatilities_log_nom = pd.read_csv(path_test_files_initialisation + "VolSwaptionsN30092021" + ".csv", decimal=',', sep=';', 
                                           header=None)
        volatilities_log_nom = np.array(volatilities_log_nom)/100
        maturity_vol = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30]
        tenor_vol = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30]

        # parametre - Calcul
        alpha = 0.120275
        ufr = 0.0345
        max_maturity = 60
        parametre = [0.5, 0.005]

        ## initialisation
        # define the rates courbes
        courbe_pzc_s = CourbeDesTaux(maturity_obs=maturity_pzc,
                                     zc=pzc,
                                     is_price=True,
                                     period='half-yearly',
                                     type_of_methode="Smith-Wilson",
                                     max_maturity_to_extrapolate=max_maturity,
                                     ufr=ufr,
                                     alpha=alpha
                                    )

        model_n = HullAndWhiteSwaption(cdt = courbe_pzc_s,
                                       parametre = parametre,
                                       maturity_vol = maturity_vol,
                                       tenor_vol = tenor_vol,
                                       vol = volatilities_nom,
                                       type_vol = "normal",
                                       type_error = 'relatif error')
        model_ln = HullAndWhiteSwaption(cdt = courbe_pzc_s,
                                       parametre = parametre,
                                       maturity_vol = maturity_vol,
                                       tenor_vol = tenor_vol,
                                       vol = volatilities_nom,
                                       type_vol = "log-normal",
                                       type_error = 'absolute error')

        self.model_n = model_n
        self.model_ln = model_ln
        self.path = path_test_files


    def test_compute_relatif_error(self):
        self.assertEqual(self.model_n._compute_relatif_error(10, 11), (1/11)**2 )
        with self.assertRaises(ValueError):
            self.model_n._compute_relatif_error(np.nan, 11)
            self.model_n._compute_relatif_error(1, np.nan)
            self.model_n._compute_relatif_error([1], 11)
            self.model_n._compute_relatif_error(1, [11])
        with self.assertRaises(ZeroDivisionError):
            self.model_n._compute_relatif_error(10, 0)
    
    def test_compute_absolute_error(self):
        self.assertEqual(self.model_ln._compute_absolute_error(10.0, 11), 1.0 )
        with self.assertRaises(ValueError):
            self.model_n._compute_absolute_error(np.nan, 11)
            self.model_n._compute_absolute_error(1, np.nan)
            self.model_n._compute_absolute_error([1], 11)
            self.model_n._compute_absolute_error(1, [11])

    def test_calcul_forward_rate_t0(self):
        self.assertAlmostEqual(self.model_n._calcul_forward_rate_t0(10-0.5, 10, self.model_n.rates, 0.5),
                                                                    0.021321296178254734)
        with self.assertRaises(ValueError):
            self.model_n._calcul_forward_rate_t0(10-0.5, 10, self.model_n.rates, 0)
            self.model_n._calcul_forward_rate_t0(10-0.5, 10, self.model_n.rates, -0.5)
            self.model_n._calcul_forward_rate_t0(10-0.5, 10, [], 0.5)
    
    def test_extraction_swap(self):
        # Importe les valeurs de reference
        matrice_swap_all_maturity = pd.read_csv(self.path + "fichier_test_matrice_swap_all_maturity" + ".csv",
                                                decimal=',', sep=';', header=None)
        matrice_swap_all_maturity = np.array(matrice_swap_all_maturity)

        self.assertAlmostEqual(self.model_n.extraction_swap(), matrice_swap_all_maturity)
        self.assertAlmostEqual(self.model_ln.extraction_swap(), matrice_swap_all_maturity)
        with self.assertRaises(ValueError):
            self.model_n.cdt.zc_prices = self.model_n.cdt.zc_prices[:-2] 
            self.model_n.extraction_swap()
    
    def test_calcul_price(self):
        # Importe les valeurs de reference
        matrice_prix_bach = pd.read_csv(self.path + "fichier_test_prix_bach" + ".csv",
                                                decimal=',', sep=';', header=None)
        matrice_prix_bach = np.array(matrice_prix_bach)
        matrice_prix_black = pd.read_csv(self.path + "fichier_test_prix_black" + ".csv",
                                                decimal=',', sep=';', header=None)
        matrice_prix_black = np.array(matrice_prix_black)

        self.assertAlmostEqual(self.model_n.calcul_price(), matrice_prix_bach)
        self.assertAlmostEqual(self.model_ln.calcul_price(), matrice_prix_black)
        with self.assertRaises(ValueError):
            self.model_n.cdt.zc_prices = self.model_n.cdt.zc_prices[:-2] 
            self.model_n.calcul_price()

    if __name__ == '__main__':
        unittest.main()