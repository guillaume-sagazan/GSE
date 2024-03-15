import os
import pandas as pd
import numpy as np
from typing import Iterable, NoReturn, Any
import matplotlib.pyplot as plt
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
log_normal_pdf = stats.lognorm.pdf(x, shape, loc=loc, scale=scale)
# Ajuster une loi de Fréchet aux données
shape_frechet, loc_frechet, scale_frechet = stats.invweibull.fit(rates)
plt.hist(rates, bins=30, density=True, alpha=0.6, color='g', label='Données')
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
frechet_pdf = stats.invweibull.pdf(x, shape_frechet, loc=loc_frechet, scale=scale_frechet)

plt.plot(x, frechet_pdf, 'k', linewidth=2, label='Loi Frechet ajustée')
plt.plot(x, log_normal_pdf, 'k', linewidth=2, label='Loi log-normale ajustée')
plt.xlabel('Valeurs')
plt.ylabel('Densité de probabilité')
plt.title('Ajustement de la loi log-normale aux données')
plt.legend()
plt.show()

donnees = rates


def plot_cdf(
        data_to_plot: Iterable[tuple[pd.Series | np.ndarray]],
        command_parameter: dict[str]
        ) -> None:
    """
    Fonction use to displays a graph for all data present in the data_to_plot.
    That is defined as a collection of tuple where the first element
    is associated to the abscisse and the value of cdf in the second elment.
    Return Nothing and displays a graph that contains all data defined in 
    data_to_plot adjusted for the commentary defined in command_parametre.
    Argument:
        data_to_plot: Iterable of tuple where the element is a Series or a numpy 
        array that contains in first argument the absciss and the second
        the ordered.
        Dict that contains data to describe each plot.
        The arguments feasible are :
            - category: plot for curve or hist for histogramme
            - color for the color of the data
    """
    category = "plot"
    color = 'k'
    for name_model, data_point in zip(command_parameter, data_to_plot):
        if 'category' in command_parameter[name_model]:
            category = command_parameter[name_model]['category']
        if "color" in command_parameter[name_model]:
            color = command_parameter[name_model]['color']
        if category == "plot":
            plt.plot(data_point[0], data_point[1], color=color, linewidth=2, label=name_model)
        else:
            plt.hist(data_point, bins=30, density=True, alpha=0.6, color=color, label=name_model)
    plt.xlabel('Valeurs')
    plt.ylabel('Densité de probabilité')
    plt.title('Ajustement de la loi log-normale aux données')
    plt.legend()
    plt.show()
    return None


data_to_plot = [(x, log_normal_pdf), (rates), (x, frechet_pdf)]
dict_of_param = {
    "log-normal": {
        "category": "plot",
        "color": "k"
        },
    "ecdf": {
        "category": "hist",
        "color": "green"
        },
    "frechet": {
        "category": "plot",
        "color": "y"
    }
    }

def generate_metric_calibration(
        data_to_measure: pd.Series | np.ndarray,
        list_name_distribution: str
        ) -> pd.DataFrame:
    dict_mapping_name_to_obj_dist = {
        "lognorm": stats.lognorm,
        "invweibull": stats.invweibull
        }
    list_of_data = []
    for name_dist in list_name_distribution:
        list_of_row_data = []
        if name_dist in dict_mapping_name_to_obj_dist:
            obj_dist = dict_mapping_name_to_obj_dist[name_dist]
            param_fit = obj_dist.fit(data_to_measure)
            dict_fit = {
                    "shape": param_fit[0],
                    "loc": param_fit[1],
                    "scale": param_fit[2]
                    }
            ks_statistic, ks_p_value = get_kstest(
                data = data_to_measure,
                name_dist = name_dist,
                param_fit = dict_fit
                )
            log_likelihood = get_log_likelihood(
                data = data_to_measure,
                obj_model = obj_dist,
                param_fit = dict_fit
                )
            r_square = get_r_squared(
                data = data_to_measure,
                obj_model = obj_dist,
                param_fit = dict_fit
                )
            list_of_row_data = \
                [
                    name_dist, ks_statistic,
                    ks_p_value, log_likelihood,
                    r_square
                ]
            
        if len(list_of_row_data) > 0:
            list_of_data.append(list_of_row_data)  
    df_data_stat = pd.DataFrame(
            list_of_data,
            columns=[
                "name distribution", "statistique KS", "p-values KS",
                "log likelihood", "r square"
                ]
                )
    return df_data_stat

def get_kstest(
        data: pd.Series | np.ndarray,
        name_dist: str,
        param_fit: dict,
        *args,
        **kargs
        ) -> tuple[float]:
    """Compute the Kolmogorov-Smirnov test"""
    try:
        shape, loc, scale = param_fit["shape"], param_fit["loc"],\
            param_fit["scale"]
    except Exception as e:
        raise ValueError(e)
    ks_statistic, ks_p_value = stats.kstest(
        data,
        name_dist,
        args=(shape, loc, scale)
        )
    return ks_statistic, ks_p_value

def get_log_likelihood(
        data: pd.Series | np.ndarray,
        obj_model: stats.rv_continuous,
        param_fit: dict,
        *args,
        **kargs
        ) -> float:
    try:
        shape, loc, scale = param_fit["shape"], param_fit["loc"],\
            param_fit["scale"]
    except Exception as e:
        raise ValueError(e)
    try:
        log_likelihood = np.sum(obj_model.logpdf(data, shape, loc, scale))
    except Exception:
        log_likelihood = -np.inf
    return log_likelihood

def get_r_squared(
        data: pd.Series | np.ndarray,
        obj_model: stats.rv_continuous,
        param_fit: dict,
        *args,
        **kargs
        ) -> float:
    try:
        shape, loc, scale = param_fit["shape"], param_fit["loc"],\
            param_fit["scale"]
    except Exception as e:
        raise ValueError(e)
    try:
        predicted_probs = obj_model.pdf(data, shape, loc, scale)
        r_squared = 1 - np.sum((data - predicted_probs)**2) / np.sum((data - np.mean(data))**2)
    except Exception:
        r_squared = -np.inf
    return r_squared
    

plot_cdf(
    data_to_plot = data_to_plot,
    command_parameter = dict_of_param
    )

df = generate_metric_calibration(
        data_to_measure = rates,
        list_name_distribution = ["lognorm", "invweibull"]
        ) 


list_data_1 = [1, 2, 3, 4, 5]
list_data_2 = [4, 1, 2, 5, 1]

np.cov(list_data_1, list_data_2)