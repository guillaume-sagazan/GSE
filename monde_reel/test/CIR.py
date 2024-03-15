import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
from monde_reel.ModelTaux_RR import ModelTaux_RR
# https://www.actuaries.org/lyon2013/papers/AFIR_Choukar_Larrieu_Bonnefoy_Hachicha.pdf

#Methode des Karnel a utiliser lors les taux sont faibles
dt = 1/12
bandwith = 1 #facteur d lissage
# Le module CIR est en travaux.
# Le problème réside dans la détermination de la COV(Xt, Xs) t>s dans la détermination des 
# parametres. La partie génération des simulations est faite mais il faut rajouter 
# la méthode qui permet d'appeler predict.
# Il manque la fonction fit. 
class CIR_RR(ModelTaux_RR):

    def __init__(
            self,
            parametre_simulation: dict,
            random_matrix: np.ndarray,
            path_data: str = "",
            data: pd.DataFrame | np.ndarray = [],
            parametre_model: dict = {},
            ) -> None:
        super().__init__(parametre_simulation, path_data, data)
        self._initiliaze_frequence_of_data()
        self._initiaze_method_calibration()
        self.random_matrix = random_matrix
        self.parametre_model = parametre_model
    
    def _initiaze_method_calibration(self) -> None:
        dict_default_value_method_calibration = {
            "return_to_mean" : "moment",
            "drift" : "moment",
            "volatility" : "no-parametric"
            }
        dict_param_simu = self.get_parametre_simulation()
        if "method_calibration" in dict_param_simu and\
            isinstance(dict_param_simu["method_calibration"], dict):
            dict_method_calibration = dict_param_simu["method_calibration"]
        else:
            dict_method_calibration = dict_param_simu
        
        dict_method_calibration_to_update = {}
        for key_variable_method in dict_default_value_method_calibration:
            if key_variable_method in dict_method_calibration:
                dict_method_calibration_to_update[key_variable_method] =\
                    dict_method_calibration[key_variable_method]
            else:
                dict_method_calibration_to_update[key_variable_method] =\
                    dict_default_value_method_calibration[key_variable_method]
        dict_param_simu["method_calibration"] = \
            dict_method_calibration_to_update
        self.parametre_simulation = dict_param_simu 

    def get_step_of_time(self):
        if "step of time" in self.parametre_simulation:
            step_of_time = self.parametre_simulation["step of time"] 
        else:
            message_error = f"The parametre step_of_time is not present in\
                parametre_simulation."
            logging.error(message_error)
            raise ValueError(message_error)
        return step_of_time
    
    def set_data(
            self,
            data: np.ndarray
            ) -> None:
        self.data = data
    
    def set_random_matrix(
            self,
            random_matrix_new: np.ndarray
            ) -> None:
        self.random_matrix = random_matrix_new

    def compute_volatility_no_param(self) -> float:
        return 1

    def kernel_gaussian(
            self,
            value_evaluation: float,
            value_variable: float,
            bandwith: float
            ) -> float:
        arg_var = (2*np.pi*bandwith*value_evaluation) ** (-0.5)
        arg_exp = (value_variable-bandwith)/(2*bandwith) * \
            (value_evaluation / ( value_variable - bandwith) \
            - 2 + (value_variable - bandwith) / value_evaluation)
        return arg_var * np.exp(-arg_exp)

    def compute_bandwith_theorical_for_normal(
            self,
            std_estimator: float,
            length_sample: int,
            cst_smoothing: float | int,
            *args, **kargs
            ) -> float:
        bandwith = cst_smoothing * std_estimator / length_sample**(2/5)
        return bandwith

    def calcul_estimator_volatilite_no_param(
            self,
            data: pd.DataFrame | pd.Series,
            bandwith: float,
            *args, **kargs
            ) -> float:
        data_shifted = data[1:].reset_index(drop=True)
        data_tronc = data[:-1].reset_index(drop=True)
        data_diff = data_shifted - data_tronc
        list_value_variable = []
        for value_variable in data_tronc:
            data_kernel_compute = \
                self.kernel_gaussian(data_tronc ,value_variable, bandwith)
            df_estimator_vol_for_value_variable = \
                data_diff**2 * data_kernel_compute
            estimator_vol_for_value_variable = \
                df_estimator_vol_for_value_variable.sum() / data_kernel_compute.sum()
            list_value_variable.append(
                (estimator_vol_for_value_variable/value_variable)**0.5
                )
        return np.median(list_value_variable)
    
    def get_data_std_estimator(self) -> float:
        data = self.get_data()
        return float(data.std)

    def compute_estimator_volatilite_no_param(self):
        data = self.get_data()
        bandwith = self.compute_bandwith_theorical_for_normal(
            std_estimator=self.get_data_std_estimator(),
            length_sample=len(data),
            cst_smoothing=3
        )
        volatility_estimator = self.calcul_estimator_volatilite_no_param(
            data=data,
            bandwith=bandwith
        )
        return volatility_estimator

    def calcul_estimator_drift_no_param(
            self,
            data: pd.DataFrame | pd.Series,
            bandwith: float,
            *args, **kargs
            ) -> float:
        data_shifted = data[1:].reset_index(drop=True)
        data_tronc = data[:-1].reset_index(drop=True)
        data_diff = data_shifted - data_tronc
        list_value_variable = []
        for value_variable in data_tronc:
            data_kernel_compute = \
                self.kernel_gaussian(
                    data_tronc,
                    value_variable,
                    bandwith
                    )
            df_estimator_vol_for_value_variable = \
                data_diff * data_kernel_compute
            estimator_drift_for_value_variable = \
                df_estimator_vol_for_value_variable.sum() / \
                    data_kernel_compute.sum()
            list_value_variable.append(
                estimator_drift_for_value_variable
                )
        return np.median(list_value_variable)
    
    def compute_estimator_drift_no_param(self):
        data = self.get_data()
        bandwith = self.compute_bandwith_theorical_for_normal(
            std_estimator=self.get_data_std_estimator(),
            length_sample=len(data),
            cst_smoothing=3
        )
        drift_estimator = self.calcul_estimator_drift_no_param(
            data=data,
            bandwith=bandwith
        )
        return drift_estimator

    def p_cov_var(
            self,
            data_t: pd.DataFrame | pd.Series | np.ndarray,
            data_s: pd.DataFrame | pd.Series | np.ndarray,
            *args, **kargs
            ) -> float:
        """
        data_t et data_s must have the same length
        """
        mat_cov = np.cov(data_t, data_s, bias=True)
        cov_ts = mat_cov[0,1]
        var_s = mat_cov[1,1]
        resultat = -np.log(cov_ts / var_s)
        return resultat

    def calcul_estimator_return_to_mean(
            self,
            data_t: pd.DataFrame,
            data_s: pd.DataFrame,
            frac_year_t: float | int,
            frac_year_s: float | int,
            *args, **kargs
        ) -> float:
        diff_frac_year = frac_year_t - frac_year_s
        log_cov_var = self.p_cov_var(data_t, data_s)
        return log_cov_var / diff_frac_year

    def compute_estimator_return_to_mean(self):
        data = self.get_data()
        list_fract_year = self.get_fract_year()
        nbr_max_elm = len(list_fract_year)
        list_estimator = [
            self.calcul_estimator_return_to_mean(
                data_t = data[nbr_max_elm-time_indice_s:],
                data_s = data[0:time_indice_s],
                frac_year_t = list_fract_year[-1],
                fract_year_s = list_fract_year[time_s]
                )
            for time_indice_s, time_s in enumerate(list_fract_year[:-1])
            ]
        return np.median(list_estimator)

    def calcul_estimator_drift_moment(
            self,
            data_t: pd.DataFrame,
            data_s: pd.DataFrame,
            *args, **kargs
            ) -> float:
        exp_return_to_mean = np.exp(-self.p_cov_var(data_t, data_s))
        numerator = np.mean(data_t) - np.mean(data_s) * exp_return_to_mean
        denominator = 1 - exp_return_to_mean
        resultat = numerator / denominator
        return resultat

    def calcul_estimator_volatilite_moment(
            self,
            data_t: pd.DataFrame,
            data_s: pd.DataFrame,
            frac_year_t: float |int,
            frac_year_s: float | int,
            theta: float,
            *args, **kargs
            ) -> float:
        diff_frac_year = (frac_year_t - frac_year_s) 
        log_cov_var = self.p_cov_var(data_t, data_s)
        return_to_mean = log_cov_var * diff_frac_year
        arg_num = return_to_mean * \
            (np.var(data_t) - np.exp(-2*log_cov_var)*np.var(data_s))

        arg_denominator_1 = np.mean(data_s) *\
            (np.exp(-log_cov_var) - np.exp(-2*log_cov_var)) 
        arg_denominator_2 = theta /2 * (1 - np.exp(-log_cov_var))**2  

        variance = arg_num/(arg_denominator_1 + arg_denominator_2)
        return variance ** 0.5

    def mapping_calibration_volatilite(self) -> dict:
        dict_mapping = {
            "non-parametric": self.calcul_estimator_volatilite_no_param,
            "moment":self.calcul_estimator_volatilite_moment
        }
        return dict_mapping

    def mapping_calibration_return_to_mean(self) -> dict:
        dict_mapping = {
            "moment": self.calcul_estimator_return_to_mean
        }
        return dict_mapping

    def mapping_calibration_drift(self) -> dict:
        dict_mapping = {
            "non-parametric": self.calcul_estimator_drift_no_param,
            "moment": self.calcul_estimator_drift_moment
        }
        return dict_mapping

def generate_taux_discretisation(param, dt, aleas):
    alpha, mu, sigma, initial_value = param

    # Nombre de simulations et nombre de périodes
    N, T = aleas.shape

    # Initialisation x0=0, y0=0, r0
    r = np.zeros((N, T+1))
    r[:, 0] = initial_value
    for i in range(1, T+1):
        # Durée de temps entre 0 et t

        determinist_calcul = r[:, i-1] + alpha*(mu - r[:, i-1])*dt
        aleas_calcul = sigma * np.sqrt(dt) * np.sqrt(r[:, i-1]) * aleas[:, i-1] + \
            sigma**2/4 * dt * (aleas[:, i-1]**2 - 1)
        r[:, i] = determinist_calcul + aleas_calcul
    
    return r

def generate_taux_exact(param, dt, aleas, initial_value=0):
    if len(param) == 4:
        alpha, mu, sigma, initial_value = param
    if len(param) == 3:
        alpha, mu, sigma = param
    c = sigma**2/(4*alpha) * (1 - np.exp(-alpha*dt))
    # Nombre de simulations et nombre de périodes
    if len(aleas.shape) == 2:
        N, T = aleas.shape
    if len(aleas.shape) == 1:
        T = int(aleas.shape[0])
        N=1
    # Initialisation x0=0, y0=0, r0
    r = np.zeros((N, T+1))
    r[:, 0] = initial_value
    for i in range(1, T+1):
        # Durée de temps entre 0 et t

        theta_t_delta = r[:, i-1] * np.exp(-alpha * dt) / c
        kappa = (4*alpha*mu) / sigma**2
        Xi = np.random.noncentral_chisquare(kappa, theta_t_delta, N)
        r[:, i] = c * Xi
    
    return r

def covariance_discrete(X, Y):
    n = len(X)
    m = len(Y)

    # Calcul des moyennes
    mean_X = np.mean(X)
    mean_Y = np.mean(Y)

    # Initialisation de la covariance
    cov_xy = 0
    prob_xy = 1/(m*n)
    # Double boucle pour la somme
    for i in range(n):
        for j in range(m):
            # Somme pondérée par les probabilités
            cov_xy += X[i] * Y[j] * prob_xy

    # Soustraction des produits des moyennes
    cov_xy -= mean_X * mean_Y

    return cov_xy



import os
from scipy.optimize import curve_fit
import scipy.stats as stats

path_input = os.path.join(
    os.getcwd(), "monde_reel", "CDS_France_2024.xlsx"
    )
data = pd.read_excel(path_input)
data["Dernier"] = data["Dernier"] /1e4
data = data.sort_values(by=["Date"], axis=0, ignore_index=True)
data_sorted = np.sort(data["Date"])
rates = data["Dernier"]
sum_mnt = rates.sum()
list_rate = rates.sort_values(ignore_index=True)
fct_rep = [list_rate[:enu].sum() for enu, val in enumerate(list_rate)]
plt.plot(list_rate, fct_rep / sum_mnt)
plt.show()
df = data.copy()
df = data.set_index("Date")
plt.plot(df["Dernier"])
plt.show()

hist, bin_edges = np.histogram(rates, bins=100)
plt.bar(bin_edges[:-1], hist, width=bin_edges[1]-bin_edges[0], color='red', alpha=0.5)
plt.show()

shape, loc, scale = stats.lognorm.fit(rates)
plt.hist(rates, bins=30, density=True, alpha=0.6, color='g', label='Données')
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.lognorm.pdf(x, shape, loc=loc, scale=scale)
plt.plot(x, p, 'k', linewidth=2, label='Loi log-normale ajustée')
plt.xlabel('Valeurs')
plt.ylabel('Densité de probabilité')
plt.title('Ajustement de la loi log-normale aux données')
plt.legend()
plt.show()

donnees = rates
ks_statistic, ks_p_value = stats.kstest(donnees, 'lognorm', args=(shape, loc, scale))
log_likelihood = np.sum(stats.lognorm.logpdf(donnees, shape, loc, scale))
predicted_probs = stats.lognorm.pdf(donnees, shape, loc, scale)
r_squared = 1 - np.sum((donnees - predicted_probs)**2) / np.sum((donnees - np.mean(donnees))**2)


# Fonction de densité de probabilité (PDF) de la loi log-normale
def lognorm_pdf(x, mu, sigma):
    return 1 / (x * sigma * np.sqrt(2 * np.pi)) * np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2))

# Ajustement de la loi log-normale aux données
parametres, cov_matrix = curve_fit(lognorm_pdf, rates, bins=30, p0=[0, 1])
list_cov = []
j = 120
for i in range(3, len(rates)):
    val_cov = covariance_discrete(X = rates.iloc[0:i],Y = rates.iloc[0:j])
    list_cov.append(val_cov)

plt.plot(list_cov)
plt.show()
i = 10
np.cov(rates.iloc[1:i], rates.iloc[1:i])
np.cov(m=rates.iloc[0:i], y=rates.iloc[0:i], bias=False)
/ np.var(rates.iloc[0:i])
mnt_global = rates.sum()
prob_cumulative = np.arange(1, len(data_sorted) + 1) / len(data_sorted)
f_cumul = [rates[:indice].sum() for indice in range(len(rates))] / mnt_global
plt.plot(f_cumul)
plt.show()

plt.plot(rates)
plt.show()

def get_fonct_cumul(data):
    list_data = []
    data_sorted = np.sort(data)
    prob_cumulative = np.arange(1, len(data_sorted) + 1) / len(data_sorted)

    for each_data in data:
        list_data.append()

param = [0.1037, 5.9279, 0.7490, 4.8323]
alpha, mu, sigma, value_init = param
test = 2*alpha*mu > sigma**2
dt = 1/12
aleas = np.random.normal(size=(100, 8))

t_disc = generate_taux_discretisation(param, dt, aleas)
t_exact = generate_taux_exact(param, dt, aleas)

t_mean_disc = np.mean(t_disc, axis=0)
t_mean_exact = np.mean(t_exact, axis=0)

plt.plot(t_mean_disc)
plt.plot(t_mean_exact)
plt.show()

std_data = float(data.std().iloc[0])
length_sample = len(data)
cst_smoothing = 3
bandwith = compute_bandwith_theorical_for_normal(
        std_estimator = std_data,
        length_sample =length_sample,
        cst_smoothing = cst_smoothing 
        )

list_estimator = calcul_estimator_drift_no_param(
        data=abs(data["taux"]),
        bandwith=bandwith
        )

kernel_gaussian(
        value_evaluation=abs(data['taux']),
        value_variable=float(data.iloc[0]),
        bandwith=bandwith
        )

plt.plot(list_estimator)
plt.show()