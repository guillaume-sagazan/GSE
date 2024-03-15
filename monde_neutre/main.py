import numpy as np
import pandas as pd
import os
from ModelTaux import *
from CourbeDesTaux import *
from Controle import *

# parametre - Path
path = os.getcwd()
suffix_files_test = "//test files//"
path_test_files = (path + suffix_files_test).replace('//', '\\')
path_test_files_initialisation = (path_test_files + "initialisation//").replace('//', '\\')
pzc = pd.read_csv(path_test_files_initialisation + "PZC30092021" + ".csv", decimal=',', sep=';', header=None)
maturity_pzc = np.arange(1, len(pzc)+1, 1)
volatilities_nom = pd.read_csv(path_test_files_initialisation + "VolSwaptionsN30092021" + ".csv", decimal=',', sep=';', 
                               header=None)
volatilities_nom = volatilities_nom.replace('', np.nan)
volatilities_nom = np.array(volatilities_nom)/100**2
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
courbe_pzc_m = CourbeDesTaux(maturity_obs=maturity_pzc,
                             zc=pzc,
                             is_price=True,
                             period='monthly',
                             type_of_methode="Smith-Wilson",
                             max_maturity_to_extrapolate=max_maturity,
                             ufr=ufr,
                             alpha=alpha
                            )
courbe_pzc_s.set_up()
courbe_pzc_s.extrapolation()
courbe_pzc_s.get_instant_forward_rate()

# define the type of model choose
model_n = HullAndWhiteSwaption(cdt = courbe_pzc_s,
                               parametre = parametre,
                               maturity_vol = maturity_vol,
                               tenor_vol = tenor_vol,
                               vol = volatilities_nom,
                               type_vol = "normal",
                               type_error = 'relatif error')
matrice_swap = model_n.extraction_swap()
matrix_price = model_n.calcul_price()
model_n._calibration_hw_swpt_atm(matrix_price, matrice_swap)
model_n.calibration() # Ne fonctionne pas
model_n.parametre
model_n.prices_calculated
model_n.prices_obs
model_n.project(10)


