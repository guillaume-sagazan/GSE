from monde_reel.ModelActions_RR import *
from monde_reel.ModelTaux_RR import *
from monde_reel.ModelTaux_black_karinski_RR import *
from monde_reel.ModelTaux_Inflation import *
from typing import Callable
import os
import numpy as np
from scipy.linalg import cholesky
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt

class Projection_RR:

    def __init__(
            self,
            parametre_simulation: dict,
            parametre_output: dict,
            matrix_random: np.ndarray
            ) -> None:
        self.parametre_simulation = parametre_simulation
        self.parametre_output = parametre_output
        self.matrix_random = matrix_random
        self.init_output()
        self.verify_information_parametre_simulation()
        self.verify_information_parametre_output()
        self.check_parametre_output()
    
    def get_parametre_simulation(self) -> dict:
        return self.parametre_simulation

    def get_parametre_output(self) -> dict:
        return self.parametre_output
    
    def get_format_of_output(self) -> str:
        parametre_output = self.get_parametre_output()
        return parametre_output["format_output"]

    def get_data(
            self,
            index_serie: str
        ) -> np.ndarray | pd.Series:
        data = "data" 
        dict_parametre = self.get_parametre_simulation()
        if index_serie in dict_parametre and \
            data in dict_parametre[index_serie]:
            return dict_parametre[index_serie][data]
        else:
            message_error = f"There is no {index_serie} or 'data' in \
                {dict_parametre[index_serie]}."
            logging.error(message_error)
            raise ValueError(message_error)
    
    def get_modeling_parametre(
            self,
            index_serie: str
            ) -> dict:
        modeling_parametre = "modeling_parametre" 
        dict_parametre = self.get_parametre_simulation()
        if index_serie in dict_parametre and \
            modeling_parametre in dict_parametre[index_serie]:
            return dict_parametre[index_serie][modeling_parametre]
        else:
            message_error = f"There is no {index_serie} or 'modeling_parametre' in \
                {dict_parametre[index_serie]}."
            logging.error(message_error)
            raise ValueError(message_error)
    
    def get_path_output(self) -> str:
        path_output = "path_output" 
        dict_parametre = self.get_parametre_output()
        if path_output in dict_parametre:
            return dict_parametre[path_output]
        else:
            message_error = f"There is no'{path_output}' in {dict_parametre} in \
                argument."
            logging.error(message_error)
            raise ValueError(message_error)
    
    def get_type_of_output(self) -> dict:
        type_of_output = "type_of_output" 
        dict_parametre_output = self.get_parametre_output()
        if type_of_output in dict_parametre_output:
            return dict_parametre_output[type_of_output]
        else:
            message_error = f"There is no'{type_of_output}' in {dict_parametre_output} in \
                {dict_parametre_output}."
            logging.error(message_error)
            raise ValueError(message_error)
    
    def get_parametre_model(
            self,
            index_serie: str
        ) -> dict:
        parametre_model = "parametre_model" 
        name_dict_hyperparametre = "hyperparametre"
        dict_parametre = self.get_parametre_simulation()
        if index_serie in dict_parametre and \
            name_dict_hyperparametre in dict_parametre[index_serie] and \
            parametre_model in dict_parametre[
                index_serie][name_dict_hyperparametre]:
            parametre_model_input = dict_parametre[
                index_serie][name_dict_hyperparametre][parametre_model]
        else:
            parametre_model_input = {}
        return parametre_model_input
    
    def get_type_product(
            self,
            index_serie: str
        ) -> str:
        type_product = "type_product" 
        name_dict_hyperparametre = "hyperparametre"
        dict_parametre = self.get_parametre_simulation()
        if index_serie in dict_parametre and \
            name_dict_hyperparametre in dict_parametre[index_serie] and \
            type_product in dict_parametre[
                index_serie][name_dict_hyperparametre]:
            return dict_parametre[
                index_serie][name_dict_hyperparametre][type_product]
        else:
            message_error = f"There is no {index_serie} or \
                '{name_dict_hyperparametre}' or '{type_product}'."
            logging.error(message_error)
            raise ValueError(message_error)
    
    def get_name_model(
            self,
            index_serie: str
        ) -> str:
        name_model = "name_model" 
        name_dict_hyperparametre = "hyperparametre"
        dict_parametre = self.get_parametre_simulation()
        if index_serie in dict_parametre and \
            name_dict_hyperparametre in dict_parametre[index_serie] and \
            name_model in dict_parametre[index_serie][
                name_dict_hyperparametre]:
            return dict_parametre[index_serie][
                name_dict_hyperparametre][name_model]
        else:
            message_error = f"There is no {index_serie} or \
                '{name_dict_hyperparametre}' or '{name_model}'."
            logging.error(message_error)
            raise ValueError(message_error)
    
    def get_rate_model(
            self,
            index_serie: str
            ) -> dict:
        rate_model = "name_rate_model"
        name_dict_hyperparametre = "hyperparametre"
        dict_parametre = self.get_parametre_simulation()
        if index_serie in dict_parametre and \
            name_dict_hyperparametre in dict_parametre[index_serie] and \
            rate_model in dict_parametre[index_serie][
                name_dict_hyperparametre]:
            name_rate_model = dict_parametre[index_serie][
                name_dict_hyperparametre][rate_model]
            object_rate_model = self.get_mapping_model_taux_RR()[name_rate_model]
            return object_rate_model
        else:
            message_error = f"There is no {index_serie} or 'rate_model' in \
                {dict_parametre[index_serie]} or '{name_dict_hyperparametre}."
            logging.error(message_error)
            raise ValueError(message_error)
    
    def get_mapping_model_taux_RR(self):
        dict_mapping_rate = {
            "Vasicek": Vasicek_RR
        }
        return dict_mapping_rate
    
    def get_mapping_model_action_RR(self):
        dict_mapping_action = {
            "Merton": Merton_RR,
            "Black-Scholes": Black_Scholes_RR
        }
        return dict_mapping_action
    
    def get_mapping_model_immobilier_RR(self):
        dict_mapping_immobilier = {
            "Merton": Merton_RR,
            "Black-Scholes": Black_Scholes_RR
        }
        return dict_mapping_immobilier
    
    def get_mapping_model_inflation_RR(self):
        dict_mapping_inflation = {
            "inflation": ModelInflation_RR
        }
        return dict_mapping_inflation
    
    def get_mapping_class_object(self) -> dict:
        dict_mapping = {
            'taux': self.get_mapping_model_taux_RR(),
            'action': self.get_mapping_model_action_RR(),
            'immobilier': self.get_mapping_model_immobilier_RR(),
            'inflation': self.get_mapping_model_inflation_RR()
        }
        return dict_mapping
    
    def get_class_object(
            self,
            index_serie: str
            ):
        dict_mapping_class = self.get_mapping_class_object()
        if self.get_type_product(index_serie) in dict_mapping_class and \
            self.get_name_model(index_serie) in \
                dict_mapping_class[project.get_type_product(index_serie)]:
            class_object = dict_mapping_class[
                self.get_type_product(index_serie)][self.get_name_model(index_serie)]
            return class_object
        else:
            message_error = f"There is no {self.get_type_product(index_serie)} \
                or {self.get_name_model(index_serie)} in parametre_simulation."
            logging.error(message_error)
            raise ValueError(message_error)
    
    def set_type_of_output(
            self,
            type_of_output: dict
            ) -> None:
        self.parametre_output["type_of_output"] = type_of_output 
    
    def init_output(self) -> None:
        try:
            dict_output = {}
            for index_serie in self.get_parametre_simulation():
                dict_output[index_serie] = {}
        except Exception as e:
            raise e
    
    @staticmethod
    def get_diff_list(self, list_1: list, list_2: list) -> list:
        set_1 = set(list_1)
        set_2 = set(list_2)
        return list(set_1 - set_2)

    def verify_information_parametre_simulation(self) -> None:
        list_key_admissible = [
            "modeling_parametre",
            "data",
            "hyperparametre"
        ]
        for index_serie, dict_value in self.get_parametre_simulation().items():
            if not all(key_admissible in dict_value 
                   for key_admissible in list_key_admissible):
                message_error = f"The index of the Series '{index_serie}'\
                    does not contains the value: \
                    {self.get_diff_list(list_key_admissible, dict_value.keys())}"
                logging.error(message_error)
                raise ValueError(message_error)
    
    def verify_information_parametre_output(self) -> None:
        list_key_admissible = [
            "path_output",
            "type_of_output"
        ]
        if not all(key_admissible in self.get_parametre_output().keys() 
            for key_admissible in list_key_admissible):
            message_error = f"The parametre_output does not contains the value: \
                {self.get_diff_list(list_key_admissible, self.get_parametre_output().keys())}"
            logging.error(message_error)
            raise ValueError(message_error)
    
    def check_validity_type_path(self) -> None:
        path_output = self.get_path_output()
        if not isinstance(path_output, str):
            message_error = f"The type of path_output is not str.\
                The type is {type(path_output)}"
            logging.error(message_error)
            raise ValueError(message_error)

    def check_validity_content_type_of_output(self) -> None:
        type_output = self.get_type_of_output()
        data_of_output_default = self.get_mapping_type_output()
        if not isinstance(type_output, dict):
            message_error = f"The type of type_output is not dict.\
                The type is {type(type_output)}"
            logging.error(message_error)
            raise ValueError(message_error)
        for key_content, is_content_compute in type_output.items():
            is_warning = False
            if key_content in data_of_output_default:
                if not isinstance(is_content_compute, bool):
                    message_warning = f"The type of content of \
                        type_of_output is not valide.\
                        The type waiting is bool \
                        and the real value of {key_content} is\
                        {type(is_content_compute)}.\
                        The value associated will be True"
                    logging.warning(message_warning)
                    is_warning = True
            if not(key_content in data_of_output_default):
                type_output[key_content] = self.do_nothing
            if is_warning or is_content_compute:
                type_output[key_content] = data_of_output_default[key_content]
        self.set_type_of_output(type_output)
    
    def get_dict_of_empiric_moment(
            self,
            list_yield: np.ndarray | pd.Series
            ) -> dict:
        mean_yield = np.mean(list_yield)
        std_yield = np.std(list_yield)
        skew_yield = skew(list_yield)
        kurtosis_yield = kurtosis(list_yield)
        dict_statistic = {
            "Moyenne": mean_yield,
            "Ecart-type": std_yield,
            "Skewness": skew_yield,
            "Kurtosis": kurtosis_yield
        }
        return dict_statistic
    
    def get_dict_of_quantil(
            self,
            list_yield: np.ndarray | pd.Series
            ) -> dict:
        centile_1 = np.percentile(list_yield, 1)
        centile_5 = np.percentile(list_yield, 5)
        centile_10 = np.percentile(list_yield, 10)
        quantil_inf = np.percentile(list_yield, 25)
        median = np.percentile(list_yield, 50)
        quantil_sup = np.percentile(list_yield, 75)
        centile_90 = np.percentile(list_yield, 90)
        centile_95 = np.percentile(list_yield, 95)
        centile_99 = np.percentile(list_yield, 99)
        dict_quantil = {
            "1er centile": centile_1,
            "5e centile": centile_5,
            "10e centile": centile_10,
            "Quartile inférieur": quantil_inf,
            "Médiane": median,
            "Quartile supérieur": quantil_sup,
            "90 centile": centile_90,
            "95e centile": centile_95,
            "99e centile": centile_99,
        }
        return dict_quantil
    
    def do_nothing(
        self,
        list_yield: np.ndarray
        ) -> None:
        print("Nothing is doing")
        return(list_yield)
    
    def get_dict_simulation(
            self,
            list_yield: np.ndarray
            ) -> np.ndarray | pd.Series:
        yield_simulation = np.transpose(list_yield)
        return yield_simulation
    
    def get_dict_by_maturity_for_statistical_method(
            self,
            array_simulation: np.ndarray,
            nbr_maturity: int,
            method_to_compute: Callable,
            ) -> dict:
        dict_quantil_maturity = {}
        for maturity in range(nbr_maturity):
            dict_quantil_tmp = method_to_compute(array_simulation[maturity, :])
            dict_quantil_maturity[maturity+1] = dict_quantil_tmp
        return dict_quantil_maturity

    def  get_dict_for_statistical_method(
            self,
            array_simulation: np.ndarray,
            method_to_compute: Callable,
            type_product: str
            ) -> dict:
        dict_statistic_simulation = {}
        if type_product == "taux":
            shape_simu = np.shape(array_simulation)
            for tenor in range(shape_simu[1]):
                dict_statistic_simulation[tenor+1] = \
                    self.get_dict_by_maturity_for_statistical_method(
                        array_simulation=array_simulation[:, tenor, :],
                        nbr_maturity=np.shape(array_simulation)[0],
                        method_to_compute=method_to_compute
                    )
        else:
            dict_statistic_simulation = \
                self.get_dict_by_maturity_for_statistical_method(
                    array_simulation=array_simulation,
                    nbr_maturity=np.shape(array_simulation)[0],
                    method_to_compute=method_to_compute
                )
        return dict_statistic_simulation
    
    def get_mapping_type_output(self) -> dict:
        mapping_type_output = {
            "quantil": self.get_dict_of_quantil,
            "statistique": self.get_dict_of_empiric_moment,
            "simulation": self.get_dict_simulation
        }
        return mapping_type_output

    def check_parametre_output(self) -> None:
        self.check_validity_type_path()
        self.check_validity_content_type_of_output()

    def build_model(self) -> dict:
        dict_model_init = {}
        for index_serie, parametre_simulation in self.get_parametre_simulation().items():
            object_creat = self.get_class_object(index_serie=index_serie)
            if not isinstance(object_creat, ModelInflation_RR):
                object_init = object_creat(
                    parametre_simulation = self.get_modeling_parametre(index_serie),
                    random_matrix = self.matrix_random,
                    data = self.get_data(index_serie),
                    parametre_model=self.get_parametre_model(index_serie))
            else:
                object_init = object_creat(
                    rate_model_nominaux = self.get_rate_model(index_serie),
                    parametre_simulation = self.get_modeling_parametre(index_serie),
                    random_matrix = self.matrix_random,
                    data = self.get_data(index_serie),
                    )                    
            dict_model_init[index_serie] = object_init
        return dict_model_init
    
    def fit_model(
            self,
            dict_model: dict
            ) -> dict:
        """
        Update the dictionnary 'dict_model' that contains the object after fit.
        Arg:
            self:
            dict_model: dict. dictionnary of model that contains
            the object. Each object must have the fit method.
        Return:
            dictionnary that contains the object of dict_model
            after fitting.
        """
        for each_model, model in dict_model.items():
            model.fit()
        return dict_model

    def predict_model(
            self,
            dict_model: dict
            ) -> dict:
        dit_simulation = {}
        for each_model, model in dict_model.items():
            dit_simulation[each_model] = model.predict()
        return dit_simulation
    
    def product_dict_output(
            self,
            dict_simulation: dict,
            ) -> dict:
        mapping_method_output = self.get_mapping_type_output()
        dict_output = {type_output: {} 
            for type_output in mapping_method_output.keys()}
        for index_serie, matrix_simulation in dict_simulation.items():
            for type_output, method_to_applied in mapping_method_output.items():
                type_product = self.get_type_product(index_serie)
                dict_output[type_output][index_serie] = \
                    self.get_dict_for_statistical_method(
                        array_simulation=matrix_simulation,
                        method_to_compute=method_to_applied,
                        type_product=type_product
                    )
        return dict_output
    
    def prepared_data_output(
            self,
            data: dict | np.ndarray
            ) -> list[pd.DataFrame]:
        try:
            if any(isinstance(data_content, dict)
                   for key, data_content in data.items()):
                list_df_data = [pd.DataFrame.from_dict(data)]
            if any(isinstance(data_content, np.ndarray)
                   for key, data_content in data):
                list_df_data = [pd.DataFrame.from_dict(data[key])
                           for key in data]
            return list_df_data
        except Exception as e:
            message_error = f"the data has not the right type or \
                an anthor issue is occured. {e}"
            logging.error(message_error)
            raise e
    
    def save_file(
            self,
            list_dataframe_of_output: list[pd.DataFrame],
            path_output_completed: str,
            information_dataframe: str = ""
            ):
        for df_data in list_dataframe_of_output:
            with open(path_output_completed, mode='a') as file:
                file.write(information_dataframe)
            df_data.to_csv(path_output_completed, mode='a', sep=";")

    def export_to_file(
            self,
            dict_to_export: dict,
            name_of_dict: str
            ) -> None:
        path_output = self.get_path_output()
        path_output_completed = os.path.join(path_output, name_of_dict + ".csv")
        for index_serie, data in dict_to_export.items():
            list_dataframe_to_output = self.prepared_data_output(data)
            self.save_file(
                list_dataframe_to_output,
                path_output_completed,
                information_dataframe=index_serie
                )
    
    def export_all_simulations(
            self,
            dict_output_prepared: dict
            ) -> None:
        for name_type_output, dict_data in dict_output_prepared.items():
            self.export_to_file(dict_data, name_of_dict=name_type_output)
    
    def run_simulation(self):
        dict_model = self.build_model()
        dict_model_fit = self.fit_model(dict_model=dict_model)
        dict_model_predict = self.predict_model(dict_model=dict_model_fit)
        dict_output_prepared = self.product_dict_output(
            dict_simulation=dict_model_predict)
        self.export_all_simulations(dict_output_prepared=dict_output_prepared)
    
    def export_simulation_to_file(
            self,
            data: dict,
            index_serie: str
            ) -> None:
        path_output = self.get_path_output()
        path_output_completed = os.path.join(path_output, "simulation" + ".csv")
        self.save_file(
            data,
            path_output_completed,
            information_dataframe=index_serie
            )

    def add_col_indexation(
        self,
        df_simulation,
        nbr_maturity: int = 1,
        nbr_simu: int = 1
        ) -> pd.DataFrame:
        list_col = list(df_simulation.columns)
        columns_add = []
        list_indexation_simulation = [i for i in range(1, nbr_simu+1) for _ in range(nbr_maturity)]
        df_simulation["simulation"] = list_indexation_simulation
        columns_add.append("simulation")
        if nbr_maturity != 1:
            list_indexation_maturity = [i for i in range(1, nbr_maturity+1)] * nbr_simu
            df_simulation["maturity"] = list_indexation_maturity
            columns_add.append("maturity")
        return df_simulation[columns_add + list_col]

    def formate_dataframe_simulation(
            self,
            df_to_adapt: pd.DataFrame,
            nbr_maturity: int = 1,
            nbr_simu: int = 1
            ) -> pd.DataFrame:
        df_adapted = df_to_adapt.copy()
        df_adapted.columns = [col + 1 for col in df_to_adapt.columns]
        df_adapted = self.add_col_indexation(df_adapted, nbr_maturity, nbr_simu)
        return df_adapted

    def adapt_data_to_dataframe_dim_2(
            self,
            data: pd.DataFrame | np.ndarray
            ) -> pd.DataFrame:
        shape_data = np.shape(data)
        is_data_formated = False

        if len(shape_data) == 2:
            nbr_maturity = shape_data[-1]
            nbr_simu = shape_data[0]
            data_formated = pd.DataFrame(data)
            data_formated = self.formate_dataframe_simulation(data_formated, nbr_simu=nbr_simu)
            is_data_formated = True
        if len(shape_data) == 3:
            nbr_maturity = shape_data[0]
            nbr_simu = shape_data[-1]
            nbr_tenor = shape_data[1]
            new_data = [data[:, :, num_simu] for num_simu in range(nbr_simu)]
            data_formated = np.resize(new_data, new_shape=(nbr_maturity*nbr_simu,nbr_tenor)) 
            data_formated = pd.DataFrame(data_formated) 
            data_formated = self.formate_dataframe_simulation(data_formated, nbr_maturity, nbr_simu)
            is_data_formated = True
        if not is_data_formated:
            message_error = "The data can not be adapted to format dim=2"
            logging.error(message_error)
            raise ValueError(message_error)
        return data_formated
    
    def get_mapping_function_to_save_file(self):
        dict_function_to_save = {
            "csv": self.save_file_to_csv,
            "excel": self.save_file_to_xlsx
        }
        return dict_function_to_save
    
    def get_mapping_create_object_to_save_file(self) -> dict:
        dict_function_mapping_to_save = {
            "csv": self.create_object_to_save_csv,
            "excel": self.create_object_to_save_xlsx
        }
        return dict_function_mapping_to_save
    
    def get_mapping_particule_format_output(self) -> dict:
        dict_format_part_output = {
            "csv": ".csv",
            "excel": ".xlsx"
        }
        return dict_format_part_output
    
    def get_particule_format_output(self) -> str:
        dict_format_part_output = self.get_mapping_particule_format_output()
        type_format_output = self.get_format_of_output()
        part_type_format_output = dict_format_part_output[type_format_output]
        return part_type_format_output
    
    def create_object_to_save_csv(
            self,
            path_output: str
            ):
        return open(path_output, mode='a')
    
    def create_object_to_save_xlsx(
            self,
            path_output: str
            ):
        return pd.ExcelWriter(path_output, mode='a')
    
    def save_file_to_csv(
            self,
            data: np.ndarray,
            path_output: str,
            index_serie: str,
            file
            ) -> None:
        file.write(index_serie)
        data.to_csv(path_output, mode='a', sep=";")
    
    def save_file_to_xlsx(
            self,
            data: np.ndarray,
            path_output: str,
            index_serie: str,
            file
            ) -> None:
        data.to_excel(file, sheet_name=index_serie, index=False)
    
    def save_file_to_good_format(
            self,
            df_output: np.ndarray,
            path_output: str,
            index_serie: str
            ) -> None:
        mapping_funct_to_save = self.get_mapping_function_to_save_file()
        type_format_output = self.get_format_of_output()
        funct_to_save = mapping_funct_to_save[type_format_output]
        object_to_save = \
            self.get_mapping_create_object_to_save_file()[type_format_output]
        with object_to_save(path_output) as file:
            funct_to_save(df_output, path_output, index_serie, file)
    
    def get_path_output_completed(
            self,
            name_file:str
        ) -> str:
        path_output = self.get_path_output()
        part_format_output = self.get_particule_format_output()
        path_output_completed = os.path.join(
            path_output, name_file + part_format_output)
        return path_output_completed

    def save_files_simulation(
            self,
            data: np.ndarray,
            index_serie: str
            ) -> None:
        path_output_completed = self.get_path_output_completed("simulation")
        df_output = self.adapt_data_to_dataframe_dim_2(data)
        self.save_file_to_good_format(
            df_output=df_output,
            path_output=path_output_completed,
            index_serie=index_serie)

    def export_simulations(
            self,
            dict_data_simulation: dict
            ) -> None:
        for index_serie, data_by_actif in dict_data_simulation.items():
            self.save_files_simulation(
                data=data_by_actif,
                index_serie=index_serie)

def correlation_brownien(
        year_of_projection,
        number_of_simulation,
        correlation_matrix,
        step_of_time,
        number_of_browniens
        ) -> np.ndarray:
    W_bro = np.random.randn( 
        int(year_of_projection * (1 / step_of_time)) + 1,
        int(number_of_browniens),
        int(number_of_simulation)
        )
    for j in range(int(year_of_projection * (1 / step_of_time))+1):
        W_bro[j, :, :] = np.dot(
            cholesky(correlation_matrix, lower=True), W_bro[j, :, :]
        )
    return W_bro

def get_brownien_correlated(
        W_bro: list | np.ndarray
        ) -> list:
    list_of_brownien_correlated = []
    dimension_correlation_brownien = np.shape(W_bro)
    for indice_brownien in range(dimension_correlation_brownien[1]):
        sub_brownien = W_bro[:, indice_brownien, :]
        brownien_correlated = np.column_stack((sub_brownien, -sub_brownien))
        list_of_brownien_correlated.append(brownien_correlated)
    return list_of_brownien_correlated

# Must have
initial_value_stock = 100
initial_value_rates = -0.0032
number_of_simulation = 500
step_of_time = 1/12 # Not required in dict_parametre_simulation
freq_of_data = "monthly" #frequence of data
year_of_projection = 40
max_maturity = 40 # rates only
start_time = "2009/01/01" # Optional
finished_time = "2023/12/31" # Optional

folder_name_monde_reel = "monde_reel"
name_file_rates = os.path.join(folder_name_monde_reel, "data_taux_court.xlsx")
name_file_infla = os.path.join(folder_name_monde_reel, "data_inflation_france.xlsx")
name_file_stock = os.path.join(folder_name_monde_reel, "data_actions_cours.xlsx")

dict_parametre_simulation_stock = {
    "initial value": initial_value_stock,
    "number of simulation": number_of_simulation * 2,
    "frequence of data": freq_of_data,
    "year of projection": year_of_projection,
    "studied period start": start_time,
    "finished period start": finished_time
}

dict_parametre_simulation_rates = {
    "initial value": initial_value_rates,
    "number of simulation": number_of_simulation * 2,
    "frequence of data": freq_of_data,
    "maturity maximal": max_maturity,
    "year of projection": year_of_projection,
    "studied period start": start_time,
    "finished period start": finished_time
}

dict_parametre_simulation_inflation = {
    "initial value": 2/100,
    "number of simulation": number_of_simulation * 2,
    "frequence of data": freq_of_data,
    "maturity maximal": max_maturity,
    "year of projection": year_of_projection,
    "studied period start": start_time,
    "finished period start": finished_time
}

# rate | action | immo
Crr = np.array(
    [[1, -0.08394538, 0.75497063],
    [-0.08394538, 1, 0.02903245],
    [0.75497063, 0.02903245, 1]]
    )

# parametre model Action / Immoblier
parametre_model_stock = [0.004128898,2.0103,0.0010,0.0000,0.0031,0.0057]
parametre_rates = [0.1, 0.02, 0.01]

W_bro = correlation_brownien(
            year_of_projection=year_of_projection,
            number_of_simulation=number_of_simulation,
            correlation_matrix=Crr,
            step_of_time=step_of_time,
            number_of_browniens=3
            )
list_of_brownien_correlated = get_brownien_correlated(W_bro)
mat_random_rates = list_of_brownien_correlated[0]
mat_random_action = list_of_brownien_correlated[1]
mat_random_immo = list_of_brownien_correlated[2]

m_act_BS = Black_Scholes_RR(
    parametre_simulation=dict_parametre_simulation_stock,
    random_matrix=mat_random_action,
    path_data=name_file_stock,
    parametre_model=[-0.0032]
)

m_act = Merton_RR(
    parametre_simulation=dict_parametre_simulation_stock,
    random_matrix=mat_random_action,
    path_data=name_file_stock,
    parametre_model=parametre_model_stock
)

m_rates =  Vasicek_RR(
    parametre_simulation = dict_parametre_simulation_rates,
    path_data=name_file_rates,
    random_matrix=mat_random_rates
    )

m_infla = ModelInflation_RR(
    rate_model_nominaux = m_rates,
    parametre_simulation = dict_parametre_simulation_inflation,
    path_data_input = name_file_infla
)

m_infla.fit()
m_infla.predict()
data_inflation = m_infla.data

from monde_reel.ModelTaux_black_karinski_RR import *
m_rates_BK =  Black_Karinski_RR(
    parametre_simulation = dict_parametre_simulation_rates,
    path_data = name_file_rates,
    random_matrix = mat_random_rates,
    #parametre_model = parametre_rates
    )

m_act.fit()
matrix_action_simulate = m_act.predict()
data_act = m_act.import_data()

m_rates.fit()
#ma = m_rates.simulate_rates()

data_rates = m_rates.import_data()
list_matrix_rates_simulate = m_rates.predict()
m_rates.parametre_simulation["number of simulation"]

m_rates_BK.fit()
m_rates_BK.evaluate_validity_calibration()

m_act_BS = Black_Scholes_RR(
    parametre_simulation=dict_parametre_simulation_stock,
    random_matrix=mat_random_action,
    path_data=name_file_stock,
    parametre_model=[0.0032]
)
m_act_BS.get_parametre_model()
m_act_BS.get_frequence_of_data()
m_act_BS.get_parametre_simulation()
data = m_act_BS.import_data()
data["Action"].plot()
plt.show()
m_act_BS.fit()
mBS = m_act_BS.predict()
pd.DataFrame(mBS.mean(axis=0)).plot()
plt.show()

data_rates = m_rates.import_data()
data_act = m_act.import_data()
new_data = [list_matrix_rates_simulate[:, :, num_simu] for num_simu in range(1000)]

parametre_simulation = {
    "Obligation": {
        "modeling_parametre": dict_parametre_simulation_rates,
        "data": data_rates,
        "hyperparametre": {
            "type_product": "taux",
            "name_model": "Vasicek",
        }
    },
    "Action": {
        "modeling_parametre": dict_parametre_simulation_stock,
        "data": data_act,
        "hyperparametre": {
            "type_product": "action",
            "name_model": "Merton",
            "parametre_model": parametre_model_stock
        }
    },
    "Inflation": {
        "modeling_parametre": dict_parametre_simulation_inflation,
        "data": data_inflation,
        "hyperparametre": {
            "type_product": "inflation",
            "name_model": "inflation",
            "name_rate_model": "Vasicek",
        }
    }
}

parametre_output = {
    "path_output": os.path.join(os.getcwd(), "monde_reel", "test"),
    "format_output": "excel",
    "type_of_output": {
        "statistique": False,
        "quantil": False,
        "simulation": True
    }
}

project = Projection_RR(
    parametre_simulation=parametre_simulation,
    parametre_output=parametre_output,
    matrix_random=mat_random_action
)

dict_model = project.build_model()
dict_model_fit = project.fit_model(dict_model=dict_model)
dict_model_predict = project.predict_model(dict_model=dict_model_fit)
project.export_simulations(dict_model_predict)

project.get_path_output()
dict_model_predict["Action"].shape

data_act
