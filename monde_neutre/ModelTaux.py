import pandas as pd
import numpy as np
from CourbeDesTaux import *
import logging
from scipy import optimize
from scipy.stats import norm
from typing import Union, List, Optional

class ModelTaux:

    def __init__(self, 
                 cdt: CourbeDesTaux,
                 parametre: np.ndarray | List[float],
                 maturity_vol: np.ndarray | List[float],
                 tenor_vol: np.ndarray | List[float],
                 vol: np.ndarray | List[float],
                 type_vol: str,
                 type_error: str # "absolute error" ou "relatif error"
                 ):
        # Configuring logging
        logging.basicConfig(filename='log.txt', level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s')
        self.cdt = cdt
        self.parametre = parametre
        self.maturity_vol = maturity_vol
        self.tenor_vol = tenor_vol
        self.vol = vol
        self.type_vol = type_vol
        self.type_model = 'Modèle de taux court'
        self.type_error = type_error
    
    def get_courbe_des_taux(self):
        return self.cdt

    def get_maturity_zc(self):
        return self.cdt.maturity
    
    def get_maturity_vol(self):
        return self.maturity_vol

    def get_tenor_vol(self):
        return self.tenor_vol

    def get_rates(self):
        return self.cdt.zc_rates
    
    def get_prices_zero_coupon(self):
        return self.cdt.zc_prices

    def get_parametre(self):
        return self.parametre
    
    def get_vol(self):
        return self.vol
    
    def get_type_vol(self):
        return self.type_vol
    
    def get_type_error(self):
        return self.type_error
    
    def calculation_error(
            self,
            value_to_compare: float | int,
            reference_value: float | int
            ) -> float:
        pass

    def fit(self):
        pass
    
    def project(self):
        pass

class ModelTauxCourt(ModelTaux):
    
    def __init__(self, 
                 cdt: CourbeDesTaux,
                 parametre: np.ndarray | List[float],
                 maturity_vol: np.ndarray | List[float],
                 tenor_vol: np.ndarray | List[float],
                 vol: np.ndarray | List[float],
                 type_vol: str,
                 type_error: str):
        super().__init__(cdt, parametre, maturity_vol, tenor_vol, vol, type_vol, type_error)
        self._get_methode_calculation_error() # initiate the attribut compute_error

    def _compute_relatif_error(self,
                               value_to_compare: float | int,
                               reference_value: float | int
                               ) -> float:
        """Compute the relatif error"""
        logging.info("Initialisation")
        if reference_value == 0:
            raise ZeroDivisionError
        if np.isnan(reference_value) or np.isnan(value_to_compare):
            message_warning = "nan value is computed in _compute_relatif_error."
            message_warning+= f" reference_value equal {reference_value} and value_to_compare equal {value_to_compare}"
            logging.warning(message_warning)
            raise ValueError(message_warning)
        try:
            relatif_error = (value_to_compare - reference_value) / reference_value
        except Exception as e:
            logging.error(e)
            raise ValueError(e)
        return relatif_error**2
    
    def _compute_absolute_error(self,
                                value_to_compare: float | int,
                                reference_value: float | int
                                ) -> float:
        """Compute the absolute_error of two values"""
        logging.info("Initialisation")
        if np.isnan(reference_value) or np.isnan(value_to_compare):
            message_warning = "nan value is computed in _compute_relatif_error."
            message_warning+= f" reference_value equal {reference_value} and value_to_compare equal {value_to_compare}"
            logging.warning(message_warning)
            raise ValueError(message_warning)
        try:
            absolute_error = value_to_compare - reference_value
        except Exception as e:
            logging.error(e)
            raise ValueError(e)
        return absolute_error**2
    
    def _get_methode_calculation_error(self):
        """Add the methode attribut for calculate the error use in the calibration"""
        logging.info("Initialisation")
        try:
            type_error_map = {"absolute error": self._compute_absolute_error,
                            "relatif error": self._compute_relatif_error}
            if self.type_error in type_error_map.keys():
                self.compute_error = type_error_map[self.type_error]
            else:
                message_error = f"The atribute type_error : {self.type_error} are incorrect. Only 'absolute error' and 'relatif error' are eligible"
                logging.error(message_error)
                raise ValueError(message_error)
        except Exception as e:
            logging.error(e)
            raise e
        
    def calculation_error(self,
                          value_to_compare: float | int,
                          reference_value: float | int
                          ) -> float:
        """Returns the value of the compute error for the inputs"""
        logging.info("Initialisation")
        try:
            error_mnt = self.compute_error(value_to_compare, reference_value)
            logging.info("Finished")
        except Exception as e:
            logging.error(e)
            raise e 
        return error_mnt
        
    def _calcul_forward_rate_t0(self,
                               t1: float,
                               t2: float
                               ) -> float:
        """Calcul the forward rates uniquely for time t=0. This function is use for
        the evaluation black and bach formula for cap or floor
            Args:
                self:
                    cdt: CourbesDesTaux object that contains:
                        zc_prices: 1d array of the zero coupon prices of dimension higher than int(n+m/tau)
                        freq: float defines as the steps (in the fraction of the year) of the forward rate
                            ie: if tau = 0.5 then it means semestrial
                t1: value of the maturity
                t2: value of the maturity such as t1 < t2
            Returns:
                float:
                    value of the forward at time 0
        """
        logging.info("Initialisation")
        try:
            # initialisation
            pzc = self.cdt.zc_prices
            freq = self.cdt.freq

            if freq <= 0:
                raise ValueError(f"The value of tau is {freq}. It can not be negative or nul")
            if len(pzc) == 0:
                raise ValueError(f"The length of price zero coupon is nul")
            forward_t0 = pzc[int(t1/freq)]/pzc[int(t2/freq)] - 1
            logging.info("Finished")
        except Exception as e:
            logging.error(e)
            raise e 
        return forward_t0/(t2-t1)
    
    def extraction_swap(self) -> np.ndarray:
        """comptue for each maturity et tenor the swap for the steps
        considerate
            Args:
                self:
                    cdt: CourbesDesTaux object that contains:
                        zc_prices: 1d array of the zero coupon prices of dimension higher than int(n+m/tau)
                        freq: float defines as the steps (in the fraction of the year) of the forward rate
                            ie: if tau = 0.5 then it means semestrial
                maturity_vol: array of the product's maturity of dimension (m) 
                tenor_vol: array of the product's tenor of dimension (n)
            Returns:
                np.array:
                    Array of dimension (m, n) that contains the value of swap for the freq choosen
        """
        logging.info("Initialisation")
        try:
            # initialisation
            pzc = self.cdt.zc_prices
            freq = self.cdt.freq
            
            max_maturity = int(np.max(self.maturity_vol))
            max_tenor = int(np.max(self.tenor_vol))
            array_swap = np.zeros( (max_maturity, max_tenor) )

            for each_tenor in range(1, max_tenor+1):
                for each_maturity in range(1, max_maturity+1):
                    diff_pzc = pzc[int(each_maturity/freq)-1] - pzc[int((each_maturity + each_tenor) / freq)-1]
                    frac_swap = freq*np.sum(
                        pzc[int(each_maturity/freq):int((each_maturity + each_tenor)/freq)]
                                            )
                    array_swap[each_maturity-1, each_tenor-1] = diff_pzc / frac_swap
            logging.info("Finished")
        except Exception as e:
            logging.error(e)
            raise e 
        return array_swap

    def calcul_strike_ATM(self,
                          maturity: np.ndarray | List[float]
                          ) -> np.ndarray:
        """Compute the value of the strike at ATM for values of price zero coupon
        after extrapolation
            Args:
                self:
                    cdt: CourbesDesTaux object that contains:
                        zc_prices: 1d array of the zero coupon prices of dimension higher than int(n+m/tau)
                    maturity: 1d array of the maturity of dimension higher than int(n+m/tau)
            Returns:
                np.array:
                    array of strike value for each maturity available with a dimension (n) 
        """
        logging.info("Initialisation")
        try:
            # initialisation
            pzc = self.cdt.zc_prices
            maturity = self.cdt.maturity

            max_maturity = int(np.max(maturity))
            array_strike_atm = np.empty(max_maturity)
            for each_maturity in range(max_maturity):
                delta_pzc = pzc[0] - pzc[2*each_maturity]
                frac_strike = np.sum(pzc[1:(2*each_maturity)])
                array_strike_atm[each_maturity] = delta_pzc/frac_strike
            logging.info("Finished")
        except Exception as e:
            logging.error(e)
            raise e 
        return array_strike_atm


class HullAndWhite(ModelTauxCourt):


    def __init__(self,
                 cdt: CourbeDesTaux,
                 parametre: np.ndarray | List[float], 
                 maturity_vol: np.ndarray | List[float],
                 tenor_vol: np.ndarray | List[float],
                 vol: np.ndarray | List[float], 
                 type_vol: str,
                 type_error: str
                 ):
        super().__init__(cdt, parametre, maturity_vol, tenor_vol, vol, type_vol, type_error)
        self.model_taux = 'Hull and White'
    
    def _calcul_coef_alpha(self, 
                           forward: float
                           ) -> float:
        """Calcul le coefficiant alpha nécessaire pour la détermination dans la diffusion du
            modèle HW
            Args:
                self: 
                    parametre: list containing the parameters (alpha et sigma)
                    cdt: CourbesDesTaux object that contains:
                        maturity: 1d array of the maturity of dimension higher than int(n+m/tau) 

                forward: Observed forward rate for maturity
            Returns:
                Float :
                    coefficiant alpha(t, t + dt) use in the diffusion
        """
        logging.info("Initialisation")
        try:
            maturity = self.cdt.maturity
            alpha = forward + (self.parametre[1]/self.parametre[0])**2*0.5
            alpha *= (1 - np.exp(-self.parametre[0])*maturity)**2
            logging.info("Finished")
        except Exception as e:
            logging.error(e)
            raise e 
        return alpha
    
    def _calcul_coef_V(self,
                       t: float | int,
                       t_dt: float | int
                       ) -> float:
        """Calcul le coefficiant alpha nécessaire pour la détermination dans la diffusion du
            modèle HW
            Args:
                self: 
                    parametre: list containing the parameters (alpha et sigma)
                t: Observed time to maturity 
                t_dt: Observed time to maturity shifted
            Returns:
                Float :
                    coefficiant V(t, t + dt) use in the diffusion
        """
        logging.info("Initialisation")
        try:
            a = self.parametre[0]
            sigma = self.parametre[1]

            V = t_dt - t + 2*np.exp(-a*(t_dt - t))/a - np.exp(-2*a*(t_dt-t))/(2*a)- 3/(2*a)
            V *= (sigma/a)**2
            logging.info("Finished")
        except Exception as e:
            logging.error(e)
            raise e 
        return V
    
    def _calcul_B(self,
                  t: float | int,
                  t_dt: float | int
                  ) -> float:
        """Calcul le coefficiant alpha nécessaire pour la détermination dans la diffusion du
            modèle HW
            Args:
                self: 
                    parametre: list containing the parameters (alpha et sigma)
                t: Observed time to maturity 
                t_dt: Observed time to maturity shifted
            Returns:
                Float :
                    coefficiant B(t, t + dt) use in the diffusion
        """
        logging.info("Initialisation")
        try:
            a = self.parametre[0]
            B = (1-np.exp(-a*(t_dt -t)))/a
            logging.info("Finished")
        except Exception as e:
            logging.error(e)
            raise e 
        return B
    
    def _diffusion_HW_reccurent(self,
                                r_t: float,
                                t: float | int,
                                t_dt: float | int,
                                forward_t: float,
                                forward_t_dt: float | int,
                                vectBrownian: np.ndarray | List[float]
                                ) -> np.ndarray | List[float]:
        """Calcul le taux à l'étape du contenu de l'intégrale suivante pour l'ensemble des éléments
            contenue dans le vecteur brownian
            Args:
                self: 
                    parametre: list containing the parameters (alpha et sigma)
                r_t: value of rate for the maturity t
                t: Observed time to maturity 
                t_dt: Observed time to maturity shifted of dt
                forward_t: Observed forward rate for t 
                forward_t_dt: Observed forward rate for t_dt
                vectBrowian: array of mouvement brownian with a dimension of (n)
            Returns:
                np.ndarray | List[float] :
                    list of rates values for the maturity t_dt of dimension (n)
        """
        logging.info("Initialisation")
        try:
            # extraction parametre
            a = self.parametre[0]
            sigma = self.parametre[1]

            U_t = ( ( 1 - np.exp(-2*a*t_dt) ) * sigma**2/(2*a) )**0.5

            r_t_dt = r_t * np.exp(-a*t_dt) + self._calcul_coef_alpha(t_dt, forward_t_dt) -\
                    self._calcul_coef_alpha(t, forward_t)*np.exp(-a*t_dt) + U_t * vectBrownian
            logging.info("Finished")
        except Exception as e:
            logging.error(e)
            raise e 
        return r_t_dt
    
    def _calcul_integral_recurring(self,
                                   r_t: float,
                                   t: float | int,
                                   dt: float | int,
                                   pzc_t: float,
                                   pzc_t_dt: float,
                                   forward_t: float,
                                   vectBrownian: np.ndarray | List[float]
                                   ) -> np.ndarray | List[float]:
        """Determinate the value of the integrale at the step t for each value of the vector Brownian
            Args:
                self: 
                    parametre: list containing the parameters (alpha et sigma)
                r_t: value of rate for the maturity t
                t: Observed time to maturity 
                dt: Difference of time to maturity shifted
                pzc_t: Price zero-coupon for the maturity t
                pzc_t_dt: Price zero-coupon for the maturity t + dt
                forward_t: Observed forward rate for t 
                vectBrowian: array of mouvement brownian with a dimension of (n)
            Returns:
                np.ndarray | List[float] :
                    list of rates values for the maturity t_dt of dimension (n)
        """
        logging.info("Initialisation")
        try:
            # calculus intermediaire
            B_t_dt = self._calcul_B(t, t + dt)
            alpha_t = self._calcul_coef_alpha(maturity=t, forward=forward_t)
            V_0_t = self._calcul_coef_V(0, t)
            V_0_t_dt = self._calcul_coef_V(0, t + dt)
            V_t_dt = self._calcul_coef_V(t, t + dt)

            # Calculus  
            value_int_r_t_dt = B_t_dt * (r_t - alpha_t) + np.log(pzc_t/pzc_t_dt) +\
                (V_0_t_dt - V_0_t)/2 + V_t_dt**2 * vectBrownian
            logging.info("Finished")
        except Exception as e:
            logging.error(e)
            raise e 
        return value_int_r_t_dt
    
    def _diffusion_short_rate_hw(self,
                                 vectBrownian: np.ndarray | List[float]
                                 ):
        """Diffuse the Hull and White method.
        Compute the short rates terms
            Args:
                self:
                    cdt:
                        zc_prices: dimension (n)
                        fwd:
                vectBrownian: vector of brownian of dimension (n x m)
            Returns:
                None:
                    Update self to add short_rates (annual view) and short_rates_monthly (monthly view)
        """
        logging.info("Initialisation")
        try:
            if not(all(attribut in dir(self.cdt) for attribut in ['maturity', 'zc_prices', 'fwd'])):
                message_error = "the attribut of CourbeDesTaux must contains 'maturity', 'zc_prices', 'fwd'"
                logging.error(message_error)
            if self.cdt.period != "monthly":
                logging.warning("This method should only be used with a monthly time step.")
            if isinstance(vectBrownian, list):
                vectBrownian = np.array(vectBrownian)
            
            short_rates = np.zeros((len(self.cdt.maturity)+1, np.shape(vectBrownian)[1] ))
            short_rates[0,] = self.cdt.inst_fwd_rates
            for indice_maturity, each_maturity in enumerate(self.cdt.maturity, start=1):
                short_rates[indice_maturity,] = self._diffusion_HW_reccurent(short_rates[indice_maturity-1],
                                                                             each_maturity,
                                                                             each_maturity + self.cdt.freq,
                                                                             self.cdt.inst_fwd_rates_monthly[indice_maturity-1],
                                                                             self.cdt.inst_fwd_rates_monthly[indice_maturity],
                                                                             vectBrownian[indice_maturity-1,]
                                                                             )
            available_annual_maturity = np.arange(0, len(self.cdt.maturity), 12)
            self.short_rates = short_rates[available_annual_maturity,]
            self.short_rates_monthly = short_rates
        except Exception as e:
            logging.error(e)
            raise e

    def _calcul_matrix_zero_coupon_prices_hw(self):
        """Diffuse the Hull and White method.
        Compute the matrix of zero coupon prices
            Args:
                self:
                    cdt:
                        zc_prices: dimension (n)
                        maturity: dimension (n)
                        fwd:
                    short_rates
            Returns:
                None:
                    Update self to add zero_coupon_prices (annual view) with the zero coupon price array
                    of dimension (n, n+1, number of simulation)
        """
        logging.info("Initialisation")
        try:
            if not(all(attribut in dir(self.cdt) for attribut in ['maturity', 'zc_prices', 'fwd'])):
                message_error = "the attribut of CourbeDesTaux must contains 'maturity', 'zc_prices', 'fwd'"
                logging.error(message_error)
            if not(all(attribut in dir(self) for attribut in ['short_rates'])):
                message_error = "the attribut of Hull and White must contains 'short_rates'. '_diffusion_short_rate_hw' have to be call before"
                logging.error(message_error)
            
            zero_coupon_prices = np.zeros(
                    (
                        np.shape(self.short_rates)[0],
                        np.shape(self.short_rates)[0] + 1,
                        np.shape(self.short_rates)[1]
                        ) 
                                  )
            zero_coupon_price_annual_extented = np.insert(self.cdt.zc_prices_obs, 0, 1)
            for index_simu in range(np.shape(self.short_rates)[1]):
                for indice_tenor in range(np.shape(self.short_rates)[0]+1):
                    for indice_maturity in range(np.shape(self.short_rates)[0]):
                        if indice_tenor + indice_maturity < len(self.cdt.zc_prices_obs):
                            zero_coupon_prices[indice_maturity, indice_tenor, index_simu] = \
                                self._calcul_price_zero_coupon(
                                    self.short_rates[indice_tenor, index_simu],
                                    indice_tenor-1,
                                    indice_maturity,
                                    zero_coupon_price_annual_extented[indice_tenor],
                                    zero_coupon_price_annual_extented[indice_tenor + indice_maturity],
                                    self.cdt.inst_fwd_rates_annual[indice_tenor]
                                    )

            self.zero_coupon_prices = zero_coupon_prices
        except Exception as e:
            logging.error(e)
            raise e    

    def _calcul_matrix_zero_coupon_rates_hw(self):
        """Diffuse the Hull and White method.
        Compute the matrix of zero coupon rates
            Args:
                self:
                    zero_coupon_rates: array of zero coupon prices resulting from the simulation of the Hull and White (n, n+1, number of simulation)
            Returns:
                None:
                    Update self to add zero_coupon_rates (annual view) with the zero coupon rates array
                    of dimension (n, n+1, number of simulation)
        """
        logging.info("Initialisation")
        try:
            if not(all(attribut in dir(self.cdt) for attribut in ['maturity', 'zc_prices', 'fwd'])):
                message_error = "the attribut of CourbeDesTaux must contains 'maturity', 'zc_prices', 'fwd'"
                logging.error(message_error)
            if not(all(attribut in dir(self) for attribut in ['zero_coupon_prices'])):
                message_error = "the attribut of Hull and White must contains 'zero_coupon_prices'. '_diffusion_short_rate_hw' have to be call before"
                logging.error(message_error)
            
            zero_coupon_rates = np.zeros_like(self.zero_coupon_prices)
            for indice_tenor in range(np.shape(self.zero_coupon_rates)[1]):
                tenor = indice_tenor + 1
                zero_coupon_rates[:, indice_tenor,:] = self.zero_coupon_prices[:, indice_tenor, :]**(-1/tenor)
            self.zero_coupon_rates = zero_coupon_rates*100
        except Exception as e:
            logging.error(e)
            raise e  
        
    def _calcul_matrix_deflator_hw(self):
        """
        Diffuse the Hull and White method.
        Compute the matrix of deflator from zero coupon rates
            Args:
                self:
                    deflator: array of deflator from short rates resulting from the simulation of the Hull and White
                    (number of simulation, number of year project)
            Returns:
                None:
                    Update self to add deflator (annual view) with a of dimension (number of simulation, number of year project = n)
        """
        logging.info("Initialisation")
        try:
            if not(all(attribut in dir(self.cdt) for attribut in ['maturity', 'zc_prices', 'fwd'])):
                message_error = "the attribut of CourbeDesTaux must contains 'maturity', 'zc_prices', 'fwd'"
                logging.error(message_error)
            if not(all(attribut in dir(self) for attribut in ['short_rates_monthly'])):
                message_error = "the attribut of Hull and White must contains 'short_rates_monthly'.\
                    '_calcul_matrix_deflator_hw' have to be call before"
                logging.error(message_error)
            deflator = np.exp(-np.apply_along_axis(
                np.cumsum, 2,
                self.short_rates_monthly * self.cdt.freq))

            maturity_available_annual = np.arange(int(1/self.cdt.freq),
                                                  np.shape(self.short_rates_monthly)[0]-1,
                                                  int(1/self.cdt.freq))
            deflator_annual = deflator[maturity_available_annual]
            
            self.deflator = deflator_annual
        except Exception as e:
            logging.error(e)
            raise e 
    
    def _calcul_biais_taux_nominal(self,
                                   t: float,
                                   t_dt: float) -> float:
        """
        Compute the biais of nominal model
            Args:
                self: 
                    parametre: list containing the parameters (alpha et sigma)
                t: Observed time to maturity
                t_dt: Observed time to maturity shifted
            Returns:
                Float:
                    value of the biais
        """
        logging.info("Initialisation")
        try:
            # extraction parametre
            a = self.parametre[0]
            sigma = self.parametre[1]

            # calculus intermediaire
            B_t_dt = self._calcul_B(t, t_dt)
            arg_1 = -sigma**2/(2*a) * (1 - np.exp(-2*a*t))*B_t_dt**2
            arg_2 = -sigma**2/(2*a**2) * (1 - np.exp(-a*t))**2*B_t_dt

            # Calculus
            biais = 1 - np.exp(arg_1 + arg_2)
            logging.info("Finished")
        except Exception as e:
            logging.error(e)
            raise e 
        return biais
    
    def _calcul_price_zero_coupon(self,
                                  r_t: float,
                                  t: float | int,
                                  dt: float | int,
                                  pzc_t: float,
                                  pzc_t_dt: float | int,
                                  forward_t: float
                                  ) -> float:
        """
        Compute the value of the price of zero coupon in Hull and White model
            Args:
                self:
                    parametre: list containing the parameters (alpha et sigma)
                r_t: value of rate for the maturity t
                t: Observed time to maturity 
                dt: Difference of time to maturity shifted
                pzc_t: Price zero-coupon for the maturity t
                pzc_t_dt: Price zero-coupon for the maturity t + dt
                forward_t: Observed forward rate for t 
                vectBrowian: array of mouvement brownian with a dimension of (n)
                                    
            Returns:
                Float:
                    value of the price of the zero coupon in Hull and White model
        """
        logging.info("Initialisation")
        try:
            if pzc_t == 0:
                message_error = "The value of zero coupon price is null. This can not be possible."
                message_error += f"\n {ZeroDivisionError} raised"
                logging.error(message_error)
                raise ZeroDivisionError
            # extraction parametre
            a = self.parametre[0]
            sigma = self.parametre[1]

            # calculus intermediaire
            B_t_dt = self._calcul_B(t, t + dt)
            arg_1 = B_t_dt * (-r_t + forward_t)
            arg_2 = -sigma**2/(4*a) * (1 - np.exp(-2*a*t))*B_t_dt**2

            # Calculus
            zc_t_dt = pzc_t_dt / pzc_t * np.exp(arg_1) * np.exp(arg_2)
            logging.info("Finished")
        except Exception as e:
            logging.error(e)
            raise e 

        return zc_t_dt 
    
    def project(self, nbr_simulation: int):
        """Project the nombre of simulation for all the maturity to 1 at max_maturity_to_extrapolate 
            Update the object to add the attributs : 
                deflator
                zero_coupon_rates
                zero_coupon_prices
                short_rates_monthly
        """
        vectBrownian = np.random.random_sample( (self.cdt.max_maturity_to_extrapolate, ) ) 
        self._diffusion_short_rate_hw(vectBrownian)
        self._calcul_matrix_zero_coupon_prices_hw()
        self._calcul_matrix_zero_coupon_rates_hw()
        self._calcul_matrix_deflator_hw()
    
    def _bach(self,
              strike: float,
              fwd: float,
              vol: float,
              t: float,
              w: float
              ) -> float:
        """Compute the bachelier function
            Args: 
                self:
                strike: value of the strike
                fwd: value of the forward
                vol: value of the volatility
                t: value of the maturity
                w: value indicate if the amount d_1 is positif or negatif
            Returns:
                float:
                    Value of the bachelier function
        """
        logging.info("Initialisation")
        try:
            signe_d_1 = np.sign(w)
            vol_temp = vol * np.sqrt(t)
            d_1 = signe_d_1 * (fwd - strike) / vol_temp
            value_bach = vol_temp * (d_1*norm.cdf(d_1) + norm.pdf(d_1))
            logging.info("Finished")
        except Exception as e:
            logging.error(e)
            raise e 
        return(value_bach)
        
class HullAndWhiteSwaption(HullAndWhite):

    def __init__(self,
                 cdt: CourbeDesTaux,
                 parametre: np.ndarray | List[float], 
                 maturity_vol: np.ndarray | List[float],
                 tenor_vol: np.ndarray | List[float],
                 vol: np.ndarray | List[float], 
                 type_vol: str,
                 type_error: str
                 ):
        super().__init__(cdt, parametre, maturity_vol, tenor_vol, vol, type_vol, type_error)
        self._get_methode_calculation_price()
        self._get_methode_calculation_vol()

    def _get_methode_calculation_price(self):
        logging.info("Initialisation")
        mapping_distribution = {"log-normal": self.swn_black_atm,
                                "normal": self.swn_bach_atm
                                }
        try:
            if not(self.type_vol in mapping_distribution.keys()):
                message_error = f"the value of type_vol does not valide. The value type_vol is {self.type_vol}"
                message_error += " only log-normal or normal are available"
                raise ValueError(message_error)
            self.method_price_atm = mapping_distribution[self.type_vol]
        except Exception as e:
            raise e
    
    def _get_methode_calculation_vol(self):
        logging.info("Initialisation")
        mapping_distribution = {"log-normal": self.vol_swn_black_atm,
                                "normal": self.vol_swn_bach_atm
                                }
        try:
            if not(self.type_vol in mapping_distribution.keys()):
                message_error = f"the value of type_vol does not valide. The value type_vol is {self.type_vol}"
                message_error += " only log-normal or normal distributions are available"
                raise ValueError(message_error)
            self.method_vol_atm = mapping_distribution[self.type_vol]
        except Exception as e:
            raise e
    
    def calcul_price(self):
        """Compute the price matrix for the hull and white methode associated with the type vol during the initialisation
            Args:
                self:
                    maturity_vol: 1-array of float of maturity with a dimension of (m)
                    tenor_vol: 1-array of float of tenors with a dimension of (n)
                    vol: 1-array of float of volatility with a dimension of (m x n)
            Returns:
                array:
                    Matrix of price with a dimension of (m x n), same dimension than volatilities
        """
        try:
            volatilities = self.vol
            maturities = self.maturity_vol
            tenors = self.tenor_vol
            matrix_price = np.full( np.shape(volatilities), np.nan )

            for indice_maturity, each_maturity in enumerate(maturities):
                for indice_tenor, each_tenor in enumerate(tenors):
                    vol = volatilities[indice_maturity, indice_tenor]
                    matrix_price[indice_maturity, indice_tenor] = self.method_price_atm(each_maturity,
                                                                                        each_maturity + each_tenor,
                                                                                        vol)
            self.matrix_price = matrix_price
            return matrix_price
        except Exception as e:
            logging.error(e)
            raise ValueError(e)

    def calcul_vol(self,
                   prices: np.ndarray | List[float]
                   ):
        """Compute the volatilities matrix for the hull and white methode associated with the type vol during the initialisation
            Args:
                self:
                    maturities: 1-array of float of maturity with a dimension of (m)
                    tenors: 1-array of float of tenors with a dimension of (n)
                prices: 1-array of float of price with a dimension of (m x n)
            Returns:
                array:
                    Matrix of price with a dimension of (m x n), same dimension than volatilities
        """
        try:
            maturities = self.maturity_vol
            tenors = self.tenor_vol
            matrix_vol = np.full(np.shape(prices), np.nan )

            for indice_maturity, each_maturity in enumerate(maturities):
                for indice_tenor, each_tenor in enumerate(tenors):
                    price = prices[indice_maturity, indice_tenor]
                    matrix_vol[indice_maturity, indice_tenor] = self.method_vol_atm(each_maturity, each_tenor, price)
            self.matrix_vol = matrix_vol
        except Exception as e:
            logging.error(e)
            raise ValueError(e)

    def swn_black_atm(self,
                      maturity: float | int,
                      tenor: float | int,
                      vol: float
                      ) -> float:
        """Compute the Black formula of a ATM swaption at t=0
            Args:
                self:
                cdt: CourbesDesTaux object that contains:
                        zc_prices: 1d array of the zero coupon prices of dimension higher than int(n+m/tau)
                        freq: float defines as the steps (in the fraction of the year) of the forward rate
                            ie: if tau = 0.5 then it means semestrial
                maturity: Observed product's time to maturity 
                tenor: Observed product's time to tenor 
                vol: Observed value of volatility  
            Returns:
                float:
                    price of a swaption in the Black's model 
        """ 
        logging.info("Initialisation")
        try:
            pzc = self.cdt.zc_prices
            freq = self.cdt.freq

            int_t = int(maturity/freq) - 1
            int_s = int(tenor/freq) - 1
            price_swaption = (2*norm.cdf(0.5*vol*np.sqrt(maturity))-1)*(pzc[int_t]-pzc[int_s])
            logging.info("Finished")
        except Exception as e:
            logging.error(e)
            raise e 
        return(price_swaption)

    def vol_swn_black_atm(self,
                          maturity: float | int,
                          tenor: float | int,
                          price_swaption: float
                          ) -> float:
        """Compute the volatility with the Black formula of a ATM swaption at t=0
            Args:
                self:
                cdt: CourbesDesTaux object that contains:
                        zc_prices: 1d array of the zero coupon prices of dimension higher than int(n+m/tau)
                        freq: float defines as the steps (in the fraction of the year) of the forward rate
                            ie: if tau = 0.5 then it means semestrial
                maturity: Observed product's time to maturity 
                tenor: Observed product's time to tenor 
                price_swaption: Observed value of price  
            Returns:
                float:
                    volatility of a swaption in the Black's model 
        """ 
        logging.info("Initialisation")
        try:
            pzc = self.cdt.zc_prices
            freq = self.cdt.freq

            int_t = int(maturity / freq) - 1
            int_s = int(tenor / freq) - 1
            
            arg_swp = 1 + price_swaption/(pzc[int_t]-pzc[int_s])
            vol_swaption = 2/np.sqrt(maturity)*norm.ppf((arg_swp)/2)
            logging.info("Finished")
        except Exception as e:
            logging.error(e)
            raise e 
        return(vol_swaption)
    
    def swn_bach_strike_k(self,
                          maturity: float | int,
                          tenor: float | int,
                          strike: float,
                          swap: float,
                          vol: float
                          ) -> float:
        """Compute the Bachelier formula for a swaption at t=0 and with a strike
            Args:
                self: 
                cdt: CourbesDesTaux object that contains:
                        zc_prices: 1d array of the zero coupon prices of dimension higher than int(n+m/tau)
                        freq: float defines as the steps (in the fraction of the year) of the forward rate
                            ie: if tau = 0.5 then it means semestrial
                maturity: Observed product's time to maturity 
                tenor: Observed product's time to tenor
                strike: value of the strike
                swap: value of the swap
                vol: Observed value of volatility  
            Returns:
                float:
                    price of a swaption in the Bachelier's model 
        """ 
        logging.info("Initialisation")
        try:
            pzc = self.cdt.zc_prices
            freq = self.cdt.freq

            int_t = int(maturity/freq) + 1
            int_s = int(tenor/freq)
            list_indice = np.sort([int_t, int_s])

            res_bach_function = self._bach(strike, swap, vol*np.sqrt(maturity), 1)
            price_swaption = res_bach_function * freq * np.sum(pzc[list_indice[0] - 1: list_indice[1]])
            logging.info("Finished")
        except Exception as e:
            logging.error(e)
            raise e 
        return(price_swaption)
    
    def swn_bach_atm(self,
                     maturity: float | int,
                     tenor: float | int,
                     vol: float
                     ) -> float:
        """Compute the Bach formula of a ATM swaption at t=0
            Args:
                self: 
                    cdt: CourbesDesTaux object that contains:
                        zc_prices: 1d array of the zero coupon prices of dimension higher than int(n+m/tau)
                        freq: float defines as the steps (in the fraction of the year) of the forward rate
                            ie: if tau = 0.5 then it means semestrial
                t: Observed time to maturity 
                tenor: Observed time to tenor   
                vol: Observed value of volatility  
            Returns:
                float:
                    price of a swaption in the Bach's model 
        """ 
        logging.info("Initialisation")
        try:
            pzc = self.cdt.zc_prices
            freq = self.cdt.freq

            int_t = int(maturity/freq) + 1
            int_s = int(tenor/freq)
            list_indice = np.sort([int_t, int_s])
            
            price_swaption = vol * np.sqrt(maturity/(2*np.pi)) * freq * np.sum(pzc[list_indice[0] - 1: list_indice[1]])
            logging.info("Finished")
        except Exception as e:
            logging.error(e)
            raise e 
        return(price_swaption)

    def vol_swn_bach_atm(self,
                         maturity: float | int,
                         tenor: float | int,
                         price_swaption: float
                         ) -> float:
        """Compute the volatility with the Bach formula of a ATM swaption at t=0
            Args:
                self:
                    cdt: CourbesDesTaux object that contains:
                            zc_prices: 1d array of the zero coupon prices of dimension higher than int(n+m/tau)
                            freq: float defines as the steps (in the fraction of the year) of the forward rate
                                ie: if tau = 0.5 then it means semestrial
                maturity: Observed time to maturity a
                tenor: Observed time to tenor 
                price_swaption: Observed value of price  
            Returns:
                float:
                    volatility of a swaption in the Bachelier's model 
        """ 
        logging.info("Initialisation")
        try:
            pzc = self.cdt.zc_prices
            freq = self.cdt.freq

            int_t = int(maturity/freq) + 1
            int_s = int(tenor/freq)
            list_indice = np.sort([int_t, int_s])
            
            vol_swaption = price_swaption * np.sqrt(2*np.pi/maturity)/(freq * np.sum(pzc[list_indice[0] - 1: list_indice[1]]))
            logging.info("Finished")
        except Exception as e:
            logging.error(e)
            raise e 
        return(vol_swaption)
    
    def _calcul_coef_A(self,
                       maturity: float | int, 
                       tenor: float | int,
                       fwd_t: float
                       ) -> float:
        """Compute the coefficiant A use in the calibration of Hull and White with swaptions
            Args:
                self: 
                    parametre: list containing the parameters (alpha et sigma)
                    cdt: CourbesDesTaux object that contains:
                            zc_prices: 1d array of the zero coupon prices of dimension higher than int(n+m/tau)
                            freq: float defines as the steps (in the fraction of the year) of the forward rate
                                ie: if tau = 0.5 then it means semestrial
                maturity: Observed time to maturity of product
                tenor: Observed time to tenor
                dt: step of time 
                fwd_t: value of the forward rate at t
            Returns:
                float :
                    coefficiant A(maturity, tenor) use in the diffusion
        """
        logging.info("Initialisation")
        try:
            # extraction parametre
            a = self.parametre[0]
            sigma = self.parametre[1]
            pzc = self.cdt.zc_prices
            freq = self.cdt.freq

            int_t = int(maturity / freq) - 1
            int_t_dt = int(tenor / freq) - 1

            B_t_dt = self._calcul_B(maturity, tenor)
            L_t = sigma**2 * (1 - np.exp(-2*a*maturity)) / (4*a)
            fract_pzc = pzc[int_t_dt] / pzc[int_t]

            coef_A =  fract_pzc*np.exp(B_t_dt*fwd_t - L_t*B_t_dt**2)
            logging.info("Finished")
        except Exception as e:
            logging.error(e)
            raise e 
        return(coef_A)
    
    def _calcul_zero_bond_put(self,
                              maturity: float | int,
                              tenor: float | int,
                              strike: float
                              ) -> float:
        """Calculate the Zero Bond price 
            Args:
                self: 
                    parametre: list containing the parameters (alpha et sigma)
                    cdt: CourbesDesTaux object that contains:
                            zc_prices: 1d array of the zero coupon prices of dimension higher than int(n+m/tau)
                            freq: float defines as the steps (in the fraction of the year) of the forward rate
                                ie: if tau = 0.5 then it means semestrial
                maturity: Observed time to maturity of product
                tenor: Observed time to tenor
                strike: value of the strike
            Returns:
                float:
                    the put value for a zero coupon for a specific strike 
        """
        logging.info("Initialisation")
        try:
            # extraction parametre
            a = self.parametre[0]
            sigma = self.parametre[1]
            pzc = self.cdt.zc_prices
            freq = self.cdt.freq
            
            int_t = int(maturity/freq) - 1
            int_t_dt = int(tenor/freq) - 1

            B_t_td = self._calcul_B(maturity, tenor)
            arg_sqrt = (1 - np.exp(-2*a*maturity))/( 2*a )
            sigma_modified = sigma * np.sqrt(arg_sqrt) * B_t_td
            h = np.log( pzc[int_t_dt] / pzc[int_t] * strike )/sigma_modified + sigma_modified/2 

            valeur_put = strike * pzc[int_t]*norm.cdf(-h + sigma_modified) - pzc[int_t_dt]*norm.cdf(-h)
            logging.info("Finished")
        except Exception as e:
            logging.error(e)
            raise e 
        return(valeur_put)
    
    def _c1(self,
            maturity: float | int,
            tenor: float | int,
            matrice_swap: np.ndarray | List[float]
            ) -> np.ndarray:
        """Calculate the coefficiant c_i use in the calcul of the price of Swaption
            Args:
                self:
                     cdt: CourbesDesTaux object that contains:
                            freq: float defines as the steps (in the fraction of the year) of the forward rate
                                ie: if tau = 0.5 then it means semestrial
                maturity: Observed time to maturity of product
                tenor: Observed time to tenor of product
                matrice_swap: array of dimention (n x n) that contains swap 
            Returns:
                Array:
                    The vector of coefficiant C_i, with a dimension of n
        """
        logging.info("Initialisation")
        try:
            freq = self.cdt.freq

            length_vect_ci = int((tenor - maturity)/freq)
            vector_ci = [0]*length_vect_ci
            vector_ci[-1] = 1 + freq * matrice_swap[int(maturity) - 1, int(tenor-maturity-1) ]
            if(length_vect_ci>1):
                vector_ci[0:-1] = [freq * matrice_swap[int(maturity-1), int(tenor-maturity-1)]] * (length_vect_ci-1)
            logging.info("Finished")
        except Exception as e:
            logging.error(f"{e}. Les parametres sont maturity : {maturity}, tenor : {tenor}, difference : {int(tenor-maturity)}")
            raise e  
        return(np.array(vector_ci))
    
    def _price_hw_atm(self,
                      maturity: float | int,
                      tenor: float | int,
                      matrice_swap: np.ndarray | List[float],
                      r_optimal_default: float = 0.04
                      ) -> np.ndarray:
        """Compute the zero coupon price in the Hull and White model 
            Args:
                self: 
                    parametre: list containing the parameters (alpha et sigma) that must be
                    higher than 0
                    cdt: CourbesDesTaux object that contains:
                            freq: float defines as the steps (in the fraction of the year) of the forward rate
                                ie: if tau = 0.5 then it means semestrial
                            inst_fwd_rates: 1d array of the instant forward rates
                maturity: Observed time to maturity (m)
                tenor: Observed time to tenor (n)
                pzc: array of zero-coupon prices
                tau: step of time 
                matrice_swap: array of dimention (m x n) that contains swap 
            Returns:
                float:
                    the zero coupon price of the input.
        """
        logging.info("Initialisation")
        try:
            freq = self.cdt.freq

            vect_payment_swp = np.arange(maturity + freq, tenor + freq, freq)
            length_payment = len(vect_payment_swp)

            vect_ci = self._c1(maturity, tenor, matrice_swap)

            f_A = lambda indice: self._calcul_coef_A(maturity, vect_payment_swp[indice], self.cdt.inst_fwd_rates[int(maturity)])
            vect_A = np.array( [f_A(indice) for indice in range(length_payment)] )

            f_B = lambda indice: self._calcul_B(maturity, vect_payment_swp[indice])
            vect_B = np.array( [f_B(indice) for indice in range(length_payment)] )

            f_optimize = lambda rate: abs( np.sum(vect_ci * vect_A * np.exp(-vect_B*rate)) - 1)
            rate_optimal = optimize.minimize(f_optimize, r_optimal_default, method="BFGS").x

            strike_optimal = vect_A * np.exp(-vect_B*rate_optimal)

            f_zcp = lambda indice: self._calcul_zero_bond_put(maturity, vect_payment_swp[indice], strike_optimal[indice])
            vector_price_zero_coupon = np.array( [f_zcp(indice) for indice in range(length_payment)] )
            price_zero_coupon = np.sum(vector_price_zero_coupon*vect_ci)
            logging.info("Finished")
        except Exception as e:
            logging.error(f"{e}, maturity: {maturity}, tenor: {tenor}")
            raise e  
        return(price_zero_coupon)

    def _calcul_mse_hw_atm(self,
                           parametre: np.ndarray | List[float],
                           vect_maturity: np.ndarray | List[float],
                           vect_tenor: np.ndarray | List[float],
                           price_swpt: np.ndarray | List[float],
                           matrice_swap: np.ndarray | List[float]
                           ) -> float:
        """Compute the least squares error use in calibration
            Args:
                self: 
                parametre: list containing the parameters (alpha et sigma) that must be higher than 0
                vect_maturity: vector of observed time to maturity (m)
                vect_tenor: vector of observed time to tenor (n)
                price_swpt: array of dimention (m x n) that swaption prices
                matrice_swap: array of dimention (m x n) that contains swap 
            Returns:
                float:
                    The least squares error of the input.
                    If parametres do not fit the conditions then the output value is 10**20
        """
        logging.info("Initialisation")
        try:
            # initialisation
            self.parametre = parametre
            a = self.parametre[0]
            sigma = self.parametre[1]
            
            length_vect_maturity = len(vect_maturity)
            length_vect_tenor = len(vect_tenor)

            mse_hw_atm = 10**20 # default value
            mat_price_gap = np.zeros( (length_vect_maturity, length_vect_tenor) )

            if(a > 0 and sigma > 0):
                for indice_maturity, each_maturity in enumerate(vect_maturity):
                    for indice_tenor, each_tenor in enumerate(vect_tenor):
                        price_computed = self._price_hw_atm(each_maturity, each_maturity + each_tenor, matrice_swap)
                        if( not(any(np.isnan([price_swpt[indice_maturity, indice_tenor], price_computed]))) ):
                            mat_price_gap[indice_maturity, indice_tenor] = self.compute_error(
                                                                            price_computed,
                                                                            price_swpt[indice_maturity, indice_tenor])
                mse_hw_atm = np.sum(mat_price_gap) * 10**10
            
            message = f"Objectif : {round(mse_hw_atm, 6)} \n Ecart relatif (en %) : {np.mean(np.sqrt(mse_hw_atm))}"
            print(message)
            logging.info(message)
            logging.info("Finished")
        except Exception as e:
            logging.error(e)
            raise e  
        return mse_hw_atm
    
    def _evaluation_price_hw_atm(self,
                                 matrice_swap: np.ndarray | List[float]
                                 ) -> float:
        """Compute the matrix prices to add to the object 
            Args:
                self: 
                    parametre: list containing the parameters (alpha et sigma) that must be higher than 0
                    vect_maturity: vector of observed time to maturity (m)
                    vect_tenor: vector of observed time to tenor (n)
                matrice_swap: array of dimention (m x n) that contains swap 
            Returns:
                float:
                    Update the object with the add of prices_calculated attribut
                    If parametres do not fit the conditions then a warninf is print and nothing is created
        """
        logging.info("Initialisation")
        try:
            # initialisation
            a = self.parametre[0]
            sigma = self.parametre[1]

            vect_maturity = self.maturity_vol
            vect_tenor = self.tenor_vol

            mat_price = np.zeros( (len(vect_maturity), len(vect_tenor)) )

            if(a > 0 and sigma > 0):
                for indice_maturity, each_maturity in enumerate(vect_maturity):
                    for indice_tenor, each_tenor in enumerate(vect_tenor):
                        price_computed = self._price_hw_atm(each_maturity, each_maturity + each_tenor, matrice_swap)
                        if( not(any(np.isnan([price_computed]))) ):
                            mat_price[indice_maturity, indice_tenor] = price_computed
            else:
                message_error = "The values of the parametre does not fit. The resultat can not be created"
                print( message_error )
                logging.warning(message_error)
            self.prices_calculated = mat_price
            logging.info("Finished")
        except Exception as e:
            logging.error(e)
            raise e  

    def _evaluation_mse_hw_atm(self,
                               vect_maturity: np.ndarray | List[float],
                               vect_tenor: np.ndarray | List[float],
                               price_swpt: np.ndarray | List[float],
                               matrice_swap: np.ndarray | List[float]
                               ) -> float:
        """Compute the least squares error for the object 
            Args:
                self: 
                    parametre: list containing the parameters (alpha et sigma) that must be higher than 0
                vect_maturity: vector of observed time to maturity (m)
                vect_tenor: vector of observed time to tenor (n)
                price_swpt: array of dimention (m x n) that swaption prices
                matrice_swap: array of dimention (m x n) that contains swap 
            Returns:
                float:
                    The least squares error of the input.
                    If parametres do not fit the conditions then the output value is 10**20
        """
        logging.info("Initialisation")
        try:
            # initialisation
            a = self.parametre[0]
            sigma = self.parametre[1]

            mse_hw_atm = 10**20 # default value
            mat_price_gap = np.zeros( (len(vect_maturity), len(vect_tenor)) )

            if(a > 0 and sigma > 0):
                for indice_maturity, each_maturity in enumerate(vect_maturity):
                    for indice_tenor, each_tenor in enumerate(vect_tenor):
                        price_computed = self._price_hw_atm(each_maturity, each_maturity + each_tenor, matrice_swap)
                        if( not(any(np.isnan([price_swpt[indice_maturity, indice_tenor], price_computed]))) ):
                            mat_price_gap[indice_maturity, indice_tenor] = self.compute_error(
                                                                            price_computed,
                                                                            price_swpt[indice_maturity, indice_tenor])
                mse_hw_atm = np.sum(mat_price_gap) * 10**10
            
            message = f"Objectif : {round(mse_hw_atm, 6)} \n Ecart relatif (en %) : {np.mean(np.sqrt(mse_hw_atm))}"
            print(message)
            logging.info(message)
            logging.info("Finished")
        except Exception as e:
            logging.error(e)
            raise e  

        return mse_hw_atm

    def _calibration_hw_swpt_atm(self,
                                 price_swpt: np.ndarray | List[float],
                                 matrice_swap: np.ndarray | List[float]
                                 ) -> float:
        """solves the optimization problem for swaption in Hull and White model.
        Changes the value of the attribut in the current object: parametre
            Args:
                self: 
                    parametre: list containing the parameters (alpha et sigma) that must be
                    higher than 0
                    cdt: CourbesDesTaux object that contains:
                            zc_prices: 1d array of the zero coupon prices of dimension higher than int(n+m/tau)
                            freq: float defines as the steps (in the fraction of the year) of the forward rate
                                ie: if tau = 0.5 then it means semestrial
                            inst_fwd_rates: 1d array of the instant forward rates
                    maturity_vol: vector of observed time to maturity (m)
                    tenor_vol: vector of observed time to tenor (n)
                price_swpt: array of dimention (m x n) that swaption prices
                matrice_swap: array of dimention (m x n) that contains swap 
            Returns:
                None:
                    Update self with new optimal parametre      
        """
        logging.info("Initialisation")
        try:
            vect_maturity = self.maturity_vol
            vect_tenor = self.tenor_vol
            initial_parametre = self.parametre

            f_optimize = lambda parametre: self._calcul_mse_hw_atm(
                parametre, vect_maturity, vect_tenor, price_swpt, matrice_swap)
            parametre_optimal = optimize.minimize(f_optimize, initial_parametre, method="BFGS").x
            self.parametre = parametre_optimal

            message_info = f"The resultat of the optimisation is : {parametre_optimal}"
            print(message_info)
            logging.info(message_info)
            logging.info("Finished")
        except Exception as e:
            logging.error(e)
            raise e
    
    def calibration(self):
        """Compute the calibration of the model.
            Update the object to modificate the value of the parametre within
        """
        matrice_swap = self.extraction_swap()
        matrix_price = self.calcul_price()
        self._calibration_hw_swpt_atm(matrix_price, matrice_swap)

        self._evaluation_price_hw_atm(matrice_swap)
        self.prices_obs = matrix_price