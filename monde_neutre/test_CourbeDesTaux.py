import unittest
from CourbeDesTaux import *
import os
import pandas as pd
import numpy as np

class TestCourbesDesTaux(unittest.TestCase):

    def setUp(self):
        path = os.getcwd()
        suffix_files_test = "//test files//"
        path_test_files = (path + suffix_files_test).replace('//', '\\')
        path_test_files_initialisation = (path_test_files + "initialisation//").replace('//', '\\')
        # parametre de teste
        alpha = 0.120275
        ufr = 0.0345
        max_maturity = 120
        pzc = pd.read_csv(path_test_files_initialisation + "PZC30092021" + ".csv", decimal=',', sep=';', header=None)
        rzc = pd.read_csv(path_test_files_initialisation + "RZC30092021" + ".csv", decimal=',', sep=';', header=None)
        maturity = np.arange(1, len(pzc)+1, 1)
        courbe_pzc = CourbeDesTaux(maturity_obs=maturity,
                                    zc=pzc,
                                    is_price=True,
                                    period='annual', # 'half-yearly', 'annual'
                                    type_of_methode="Smith-Wilson",
                                    max_maturity_to_extrapolate=max_maturity,
                                    ufr=ufr,
                                    alpha=alpha)
        courbe_rzc = CourbeDesTaux(maturity_obs=maturity,
                                    zc=rzc,
                                    is_price=False,
                                    period='annual', # 'half-yearly', 'annual'
                                    type_of_methode="Smith-Wilson",
                                    max_maturity_to_extrapolate=max_maturity,
                                    ufr=ufr,
                                    alpha=alpha)
        courbe_pzc_error = CourbeDesTaux(maturity_obs=maturity[1:-1],
                                         zc=rzc,
                                         is_price=False,
                                         period='annual', # 'half-yearly', 'annual'
                                         type_of_methode="Smith-Wilson",
                                         max_maturity_to_extrapolate=max_maturity,
                                         ufr=ufr,
                                         alpha=alpha)
        courbe_pzc_s = CourbeDesTaux(maturity_obs=maturity,
                                    zc=pzc,
                                    is_price=True,
                                    period='half-yearly', 
                                    type_of_methode="Smith-Wilson",
                                    max_maturity_to_extrapolate=max_maturity,
                                    ufr=ufr,
                                    alpha=alpha)
        courbe_pzc_m = CourbeDesTaux(maturity_obs=maturity,
                                     zc=pzc,
                                     is_price=True,
                                     period='monthly',
                                     type_of_methode="Smith-Wilson",
                                     max_maturity_to_extrapolate=max_maturity,
                                     ufr=ufr,
                                     alpha=alpha)
        courbe_pzc_d = CourbeDesTaux(maturity_obs=maturity,
                                    zc=pzc,
                                    is_price=True,
                                    period='daily',
                                    type_of_methode="Smith-Wilson",
                                    max_maturity_to_extrapolate=max_maturity,
                                    ufr=ufr,
                                    alpha=alpha)
        self.courbe_pzc = courbe_pzc
        self.courbe_rzc = courbe_rzc
        self.courbe_zc_error = courbe_pzc_error
        self.courbe_pzc_semestriel = courbe_pzc_s
        self.courbe_pzc_monthly = courbe_pzc_m
        self.courbe_pzc_daily = courbe_pzc_d

        self.path = path_test_files

    def test_calculate_rates(self):
        self.assertAlmostEquals(self.courbe_pzc._calculate_rates(), np.array(self.courbe_rzc.zc).flatten())
        with self.assertRaises(ValueError):
            self.courbe_zc_error._calculate_rates()
    
    def test_calculate_prices(self):
        self.assertAlmostEquals(self.courbe_rzc._calculate_prices(), np.array(self.courbe_pzc.zc).flatten())
        with self.assertRaises(ValueError):
            self.courbe_zc_error._calculate_rates()
    
    def test_fit_smithwilson_rates(self):
        # Initialiser les objects
        self.courbe_pzc.set_up()
        self.courbe_pzc_semestriel.set_up()
        self.courbe_pzc_daily.set_up()

        # Importe les valeurs de reference
        prix_zero_coupon_annuelle = pd.read_csv(self.path + "fichier_test_prix_zero_coupon_annuelle" + ".csv",
                                                 decimal=',', sep=';', header=None)
        prix_zero_coupon_semestriel = pd.read_csv(self.path + "fichier_test_prix_zero_coupon_semestriel" + ".csv",
                                                 decimal=',', sep=';', header=None)
        prix_zero_coupon_quotidien = pd.read_csv(self.path + "fichier_test_prix_zero_coupon_quotidien" + ".csv",
                                                 decimal=',', sep=';', header=None)
        
        # Test
        self.courbe_pzc._fit_smithwilson_rates()
        self.courbe_pzc_semestriel._fit_smithwilson_rates()
        self.courbe_pzc_daily._fit_smithwilson_rates()
        self.assertAlmostEquals(self.courbe_pzc.zc_prices, np.array(prix_zero_coupon_annuelle).flatten())
        self.assertAlmostEquals(self.courbe_pzc_semestriel.zc_prices, np.array(prix_zero_coupon_semestriel).flatten())
        self.assertAlmostEquals(self.courbe_pzc_daily.zc_prices, np.array(prix_zero_coupon_quotidien).flatten())
        with self.assertRaises(ValueError):
            self.courbe_zc_error._fit_smithwilson_rates()

    def test_get_forward_rates(self):
        # Initialiser les objects
        self.courbe_pzc.set_up()
        self.courbe_pzc.extrapolation()
        self.courbe_pzc_monthly.set_up()
        self.courbe_pzc_monthly.extrapolation()
        self.courbe_pzc_daily.set_up()
        self.courbe_pzc_daily.extrapolation()

        # Importe les valeurs de reference
        forward_rate_daily = pd.read_csv(self.path + "fichier_test_forward_pas_quotidien" + ".csv",
                                         decimal=',', sep=';', header=None)
        forward_rate_monthly = pd.read_csv(self.path + "fichier_test_forward_pas_mensuelle" + ".csv",
                                           decimal=',', sep=';', header=None)
        forward_rate_annuelle = pd.read_csv(self.path + "fichier_test_forward_pas_annuelle" + ".csv",
                                            decimal=',', sep=';', header=None)
        # Test
        self.courbe_pzc_daily.get_forward_rate()
        self.courbe_pzc_monthly.get_forward_rate()
        self.courbe_pzc.get_forward_rate()
        self.assertEqual(self.courbe_pzc_daily.fwd_rates, np.array(forward_rate_daily).flatten())
        self.assertEqual(self.courbe_pzc_monthly.fwd_rates, np.array(forward_rate_monthly).flatten())
        self.assertEqual(self.courbe_pzc.fwd_rates, np.array(forward_rate_annuelle).flatten())
        
        with self.assertRaises(ValueError):
            self.courbe_zc_error.get_forward_rate()
    
    def test_get_inst_forward_rates(self):
        # Initialiser les objects
        self.courbe_pzc.set_up()
        self.courbe_pzc.extrapolation()
        self.courbe_pzc_monthly.set_up()
        self.courbe_pzc_monthly.extrapolation()
        self.courbe_pzc_daily.set_up()
        self.courbe_pzc_daily.extrapolation()

        # Importe les valeurs de reference
        forward_rate_daily = pd.read_csv(self.path + "fichier_test_forward_pas_quotidien_instantané" + ".csv",
                                         decimal=',', sep=';', header=None)
        forward_rate_monthly = pd.read_csv(self.path + "fichier_test_forward_pas_mensuelle_instantané" + ".csv",
                                           decimal=',', sep=';', header=None)
        forward_rate_annuelle = pd.read_csv(self.path + "fichier_test_forward_pas_annuelle_instantané" + ".csv",
                                            decimal=',', sep=';', header=None)
        # Test
        self.courbe_pzc_daily.get_instant_forward_rate()
        self.courbe_pzc_monthly.get_instant_forward_rate()
        self.courbe_pzc.get_instant_forward_rate()
        self.assertEqual(self.courbe_pzc_daily.inst_fwd_rates, np.array(forward_rate_daily).flatten())
        self.assertEqual(self.courbe_pzc_monthly.inst_fwd_rates, np.array(forward_rate_monthly).flatten())
        self.assertEqual(self.courbe_pzc.inst_fwd_rates, np.array(forward_rate_annuelle).flatten())
        
        with self.assertRaises(ValueError):
            self.courbe_zc_error.get_forward_rate()

    if __name__ == '__main__':
        unittest.main()