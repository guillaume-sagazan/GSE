import pandas as pd
import numpy as np
import logging
from scipy import optimize
from scipy.stats import norm
from typing import Union, List, Optional

class ModelTaux:

    def __init__(self,
                 maturity:  Union[np.ndarray, List[float]],
                 rates: Union[np.ndarray, List[float]], 
                 parametre: Union[np.ndarray, List[float]]
                 ):
        self.maturity = maturity
        self.rates = rates
        self.parametre = parametre

    def get_maturity(self):
        return self.maturity
    
    def get_rates(self):
        return self.rates
    
    def get_parametre(self):
        return self.parametre
    
    def fit(self):
        return None
    
    def predict(self):
        return None

class ModelTauxCourt(ModelTaux):
    
    def __init__(self, 
                 maturity: np.ndarray | List[float],
                 rates: np.ndarray | List[float],
                 parametre: np.ndarray | List[float],
                 vol: np.ndarray | List[float],
                 type_vol: str,
                 type_error: str):
        super().__init__(maturity, rates, parametre)
        self.vol = vol
        self.type_vol = type_vol
        self.type_model = 'Modèle de taux court'
        self.type_error = type_error

    def get_vol(self):
        return self.vol
    
    def get_type_vol(self):
        return self.type_vol
    
    def compute_relatif_error(self,
                               value_to_compare: float | int,
                               reference_value: float | int
                               ) -> float:
        """Compute the relatif error"""
        if reference_value == 0:
            raise ZeroDivisionError
        return( np.isnan(reference_value), np.isnan(value_to_compare) )
        if np.isnan(reference_value) or np.isnan(value_to_compare):
            raise ValueError("nan value is computed")
        try:
            relatif_error = (value_to_compare - reference_value) / reference_value
        except Exception as e:
            raise ValueError(e)
        return relatif_error**2
    
    def _compute_absolute_error(self,
                               value_to_compare: float | int,
                               reference_value: float | int
                               ) -> float:
        """Compute the absolute_error of two values"""
        absolute_error = value_to_compare - reference_value
        return absolute_error**2
    
    def get_forward_rate(pzc: np.ndarray | List[float],
                         tau: float
                         ) -> np.ndarray | List[float]:
        """Compute the forward rate transformation in function of the steps
        for each value of the zero coupon price in input
            Args:
                pzc: vector contains the price of zero coupon
                tau: float defines as the steps (in the fraction of the year) of the forward rate
                    ie: if tau = 0.5 then it means semestrial
            Returns:
                np.array:
                    Array contains the value of the forward rates
        """
        p_pzc = -np.log(pzc)
        length_pzc = len(pzc)
        forward_rate = (p_pzc[2:length_pzc] - p_pzc[1:length_pzc - 1])/tau
        return forward_rate
    
    def calcul_forward_rate_t0(t1: float,
                               t2: float,
                               pzc: np.ndarray | List[float],
                               tau: float
                               ) -> float:
        """Calcul the forward rates uniquely for time t=0
            Args:
                t1: value of the maturity
                t2: value of the maturity such as t1 < t2
                pzc: array of the maturity of dimension higher than int(n+m/tau)
                tau: float defines as the steps (in the fraction of the year) of the forward rate
                    ie: if tau = 0.5 then it means semestrial
            Returns:
                float:
                    value of the forward at time 0
        """
        forward_t0 = pzc[int(t1/tau)]/pzc[int(t2/tau)] - 1
        return forward_t0/(t2-t1)
    
    def extraction_swap(maturity: np.ndarray | List[float],
                        tenor: np.ndarray | List[float],
                        pzc: np.ndarray | List[float],
                        tau: float) -> np.ndarray:
        """comptue for each maturity et tenor the swap for the steps
        considerate
            Args:
                maturity: array of the maturity of dimension (n)
                tenor: array of the tenor of dimension (m)
                pzc: array of the value of price zero coupon of dimension higher than int(n+m/tau)
                tau: float defines as the steps (in the fraction of the year) of the forward rate
                    ie: if tau = 0.5 then it means semestrial
            Returns:
                np.array:
                    Array of dimension (m, n) that contains the value of
                    swap for the step choosen
        """
        max_maturity = int(np.max(maturity))
        max_tenor = int(np.max(tenor))
        array_swap = np.zeros( (max_maturity, max_tenor) )
        for each_tenor in range(1, max_tenor):
            for each_maturity in range(1, max_maturity):
                diff_pzc = pzc[int(each_maturity/tau)] - pzc[int(each_maturity + each_tenor) /tau]
                frac_swap = tau*np.sum(
                    pzc[int(each_maturity/tau + 1):int((each_maturity+each_tenor)/tau)]
                                        )
                array_swap[each_maturity, each_tenor] = diff_pzc / frac_swap
        return array_swap

    def calcul_strike_ATM(self,
                          pzc: np.ndarray | List[float],
                          maturity: np.ndarray | List[float]) -> np.ndarray:
        """Compute the value of the strike at ATM for values of price zero coupon
        after extrapolation
            Args:
                self:
                pzc: array of the value of price zero coupon of dimension after extrapolation (n)
                maturity: array of the maturity of dimension after extrapolation (n)
            Returns:
                np.array:
                    array of strike value for each maturity available with a dimension (n) 
        """
        max_maturity = int(np.max(maturity))
        array_strike_atm = np.empty(max_maturity)
        for each_maturity in range(max_maturity):
            delta_pzc = pzc[0] - pzc[2*each_maturity]
            frac_strike = np.sum(pzc[1:(2*each_maturity)])
            array_strike_atm[each_maturity] = delta_pzc/frac_strike
        return frac_strike


class HullAndWhite(ModelTauxCourt):


    def __init__(self,
                 maturity: np.ndarray | List[float],
                 rates: np.ndarray | List[float], 
                 parametre: np.ndarray | List[float], 
                 vol: np.ndarray | List[float], 
                 type_vol: str,
                 type_error: str
                 ):
        super().__init__(maturity, rates, parametre, vol, type_vol, type_error)
        self.model_taux = 'Hull and White'
    
    def _calcul_coef_alpha(self, 
                           maturity: float | int, 
                           forward: float
                           ) -> float:
        """Calcul le coefficiant alpha nécessaire pour la détermination dans la diffusion du
            modèle HW
            Args:
                self: 
                    parametre: list containing the parameters (alpha et sigma)
                    maturity: Observed time to maturity 
                forward: Observed forward rate for maturity
            Returns:
                Float :
                    coefficiant alpha(t, t + dt) use in the diffusion
        """
        alpha = forward + (self.parametre[1]/self.parametre[0])^2*0.5
        alpha *= (1 - np.exp(-self.parametre[0])*maturity)^2
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
        a = self.parametre[0]
        sigma = self.parametre[1]
        V = t_dt - t + 2*np.exp(-a*(t_dt - t))/a - np.exp(-2*a*(t_dt-t))/(2*a)- 3/(2*a)
        V *= (sigma/a)^2
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
        a = self.parametre[0]
        B = (1-np.exp(-a*(t_dt -t)))/a
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
        # extraction parametre
        a = self.parametre[0]
        sigma = self.parametre[1]

        U_t = ( ( 1 - np.exp(-2*a*t_dt) ) * sigma^2/(2*a) )^0.5

        r_t_dt = r_t * np.exp(-a*t_dt) + self._calcul_coef_alpha(t_dt, forward_t_dt) -\
                 self._calcul_coef_alpha(t, forward_t)*np.exp(-a*t_dt) + U_t * vectBrownian
        return r_t_dt
    
    def _calcul_integral_recurring(self,
                                   r_t: float,
                                   t: float | int,
                                   dt: float | int,
                                   pzc_t: float,
                                   pzc_t_dt: float | int,
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
        # calculus intermediaire
        B_t_dt = self._calcul_B(t, t + dt)
        alpha_t = self._calcul_coef_alpha(maturity=t, forward=forward_t)
        V_0_t = self._calcul_coef_V(0, t)
        V_0_t_dt = self._calcul_coef_V(0, t + dt)
        V_t_dt = self._calcul_coef_V(t, t + dt)

        # Calculus  
        value_int_r_t_dt = B_t_dt * (r_t - alpha_t) + np.log(pzc_t/pzc_t_dt) +\
            (V_0_t_dt - V_0_t)/2 + V_t_dt^2 * vectBrownian
        
        return value_int_r_t_dt

    def _calcul_biais_taux_nominal(self, t, t_dt) -> float:
        """Compute the biais of nominal model
            Args:
                self: 
                    parametre: list containing the parameters (alpha et sigma)
                r_t: value of rate for the maturity t
                t: Observed time to maturity
            Returns:
                Float:
                    value of the biais
        """
        # extraction parametre
        a = self.parametre[0]
        sigma = self.parametre[1]

        # calculus intermediaire
        B_t_dt = self._calcul_B(t, t_dt)
        arg_1 = -sigma^2/(2*a) * (1 - np.exp(-2*a*t))*B_t_dt^2
        arg_2 = -sigma^2/(2*a^2) * (1 - np.exp(-a*t))^2*B_t_dt

        # Calculus
        biais = 1 - np.exp(arg_1 + arg_2)
        return biais
    
    def _calcul_price_zero_coupon(self,
                                  r_t: float,
                                  t: float | int,
                                  dt: float | int,
                                  pzc_t: float,
                                  pzc_t_dt: float | int,
                                  forward_t: float
                                  ) -> float:
        """Compute the value of the price of zero coupon 
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
        # extraction parametre
        a = self.parametre[0]
        sigma = self.parametre[1]

        # calculus intermediaire
        B_t_dt = self._calcul_B(t, t + dt)
        arg_1 = B_t_dt * (-r_t + forward_t)
        arg_2 = -sigma^2/(4*a) * (1 - np.exp(-2*a*t))*B_t_dt^2

        # Calculus
        zc_t_dt = pzc_t_dt / pzc_t * np.exp(arg_1) * np.exp(arg_2)

        return zc_t_dt 
    
    def _bach(self,
              k: float,
              fwd: float,
              vol: float,
              t: float,
              w: float
              ) -> float:
        """Compute the bachelier function
            Args: 
                self:
                k: value of the strike
                fwd: value of the forward
                vol: value of the volatility
                t: value of the maturity
                w: value indicate if the amount d_1 is positif or negatif
            Returns:
                float:
                    Value of the bachelier function
        """
        signe_d_1 = np.sign(w)
        vol_temp = vol * np.sqrt(t)
        d_1 = signe_d_1 * (fwd - k) / vol_temp
        value_bach = vol_temp * (d_1*norm.cdf(d_1) + norm.pdf(d_1))
        return(value_bach)
        
class HullAndWhiteSwaption(HullAndWhite):

    def __init__(self,
                 maturity: np.ndarray | List[float],
                 rates: np.ndarray | List[float],
                 parametre: np.ndarray | List[float], 
                 vol: np.ndarray | List[float], 
                 type_vol: str,
                 type_error: str
                 ):
        super().__init__(maturity, rates, parametre, vol, type_vol, type_error)

    def swn_black_atm(self,
                      maturity: float | int,
                      tenor: float | int,
                      tau: float,
                      pzc: np.ndarray | List[float],
                      vol: float
                      ) -> float:
        """Compute the Black formula of a ATM swaption at t=0
            Args:
                self:
                maturity: Observed time to maturity 
                tenor: Observed time to tenor 
                tau: float defines as the steps (in the fraction of the year) of the forward rate
                    ie: if tau = 0.5 then it means semestrial
                pzc: array of zero-coupon prices
                vol: Observed value of volatility  
            Returns:
                float:
                    price of a swaption in the Black's model 
        """ 
        int_t = int(maturity/tau)
        int_s = int(tenor/tau)
        price_swaption = (2*norm.cdf(0.5*vol*np.sqrt(maturity))-1)*(pzc[int_t]-pzc[int_s])
        return(price_swaption)

    def vol_swn_black_atm(self,
                          maturity: float | int,
                          tenor: float | int,
                          tau: float,
                          pzc: np.ndarray | List[float],
                          price_swaption: float
                          ) -> float:
        """Compute the volatility with the Black formula of a ATM swaption at t=0
            Args:
                self:
                maturity: Observed time to maturity 
                tenor: Observed time to tenor 
                tau: float defines as the steps (in the fraction of the year) of the forward rate
                    ie: if tau = 0.5 then it means semestrial
                pzc: array of zero-coupon prices
                price_swaption: Observed value of price  
            Returns:
                float:
                    volatility of a swaption in the Black's model 
        """ 
        int_t = int(maturity/tau)
        int_s = int(tenor/tau)
        
        arg_swp = 1+price_swaption/(pzc[int_t]-pzc[int_s])
        vol_swaption = 2/np.sqrt(maturity)*norm.ppf((arg_swp)/2)
        return(vol_swaption)
    
    def swn_bach_strike_k(self,
                          maturity: float | int,
                          tenor: float | int,
                          tau: float,
                          k: float,
                          swap: float,
                          pzc: np.ndarray | List[float],
                          vol: float
                          ) -> float:
        """Compute the Bachelier formula for a swaption at t=0 and with a strike of k
            Args:
                self:
                maturity: Observed time to maturity 
                tenor: Observed time to tenor 
                tau: float defines as the steps (in the fraction of the year) of the forward rate
                    ie: if tau = 0.5 then it means semestrial
                k: value of the strike
                swap: value of the swap
                pzc: array of zero-coupon prices
                vol: Observed value of volatility  
            Returns:
                float:
                    price of a swaption in the Bachelier's model 
        """ 
        int_t = int(maturity/tau) + 1
        int_s = int(tenor/tau)

        res_bach_function = self._bach(k, swap, vol*np.sqrt(maturity), 1)
        price_swaption = res_bach_function * tau * np.sum(pzc[int_t:int_s])
        return(price_swaption)
    
    def swn_bach_atm(self,
                     maturity: float | int,
                     tenor: float | int,
                     tau: float,
                     pzc: np.ndarray | List[float],
                     vol: float
                     ) -> float:
        """Compute the Bach formula of a ATM swaption at t=0
            Args:
                self:
                t: Observed time to maturity 
                tenor: Observed time to tenor 
                tau: float defines as the steps (in the fraction of the year) of the forward rate
                    ie: if tau = 0.5 then it means semestrial
                pzc: array of zero-coupon prices
                vol: Observed value of volatility  
            Returns:
                float:
                    price of a swaption in the Bach's model 
        """ 
        int_t = int(maturity/tau) + 1
        int_s = int(tenor/tau)
        
        price_swaption = vol * np.sqrt(maturity/(2*np.pi)) * tau * np.sum(pzc[int_t: int_s])
        return(price_swaption)

    def vol_swn_black_atm(self,
                          maturity: float | int,
                          tenor: float | int,
                          tau: float,
                          pzc: np.ndarray | List[float],
                          price_swaption: float
                          ) -> float:
        """Compute the volatility with the Bach formula of a ATM swaption at t=0
            Args:
                self:
                maturity: Observed time to maturity 
                tenor: Observed time to tenor 
                tau: float defines as the steps (in the fraction of the year) of the forward rate
                    ie: if tau = 0.5 then it means semestrial
                pzc: array of zero-coupon prices
                price_swaption: Observed value of price  
            Returns:
                float:
                    volatility of a swaption in the Bachelier's model 
        """ 
        int_t = int(maturity/tau) + 1
        int_s = int(tenor/tau)
        
        vol_swaption = price_swaption * np.sqrt(2*np.pi/maturity)/(tau * np.sum(pzc[int_t: int_s]))
        return(vol_swaption)
    

    def _calcul_coef_A(self,
                       maturity: float | int, 
                       tenor: float | int,
                       tau: float,
                       fwd_t: float,
                       pzc: np.ndarray | List[float]
                       ) -> float:
        """Compute the coefficiant A use in the calibration of Hull and White with swaptions
            Args:
                self: 
                    parametre: list containing the parameters (alpha et sigma)
                maturity: Observed time to maturity 
                tenor: Observed time to tenor
                dt: step of time 
                fwd_t: value of the forward rate at t
                pzc: array of zero-coupon prices
            Returns:
                float :
                    coefficiant A(maturity, tenor) use in the diffusion
        """
        # extraction parametre
        a = self.parametre[0]
        sigma = self.parametre[1]

        int_t = int(maturity/tau) 
        int_t_dt = int(tenor/tau)
        B_t_dt = self._calcul_B(maturity, tenor)
        L_t = sigma^2 * (1 - np.exp(-2*a*maturity)) / (4*a)
        fract_pzc = pzc[int_t_dt] / pzc[int_t]

        coef_A =  fract_pzc*(B_t_dt*fwd_t - L_t*B_t_dt^2)
        return(coef_A)
    
    def _calcul_zero_bond_put(self,
                              maturity: float | int,
                              tenor: float | int,
                              tau: float,
                              pzc: np.ndarray | List[float],
                              strike: float
                              ) -> float:
        """Calculate the Zero Bond price 
            Args:
                self: 
                    parametre: list containing the parameters (alpha et sigma)
                maturity: Observed time to maturity 
                tenor: Observed time to tenor
                tau: step of time 
                pzc: array of zero-coupon prices
                strike: value of the strike
            Returns:
                float:
                    the put value for a zero coupon for a specific strike 
        """
        # extraction parametre
        a = self.parametre[0]
        sigma = self.parametre[1]
        
        int_t = int(maturity/tau) 
        int_t_dt = int(tenor/tau)
        B_t_td = self._calcul_B(maturity, tenor)
        arg_sqrt = (1 - np.exp(-2*a*maturity))/( 2*a )
        sigma_modified = sigma * np.sqrt(arg_sqrt) * B_t_td
        h = np.log( pzc[int_t_dt] / pzc[int_t] * strike )/sigma_modified + sigma_modified/2 

        valeur_put = strike * pzc[int_t]*norm.cdf(-h + sigma_modified) - pzc[int_t_dt]*norm.cdf(-h)
        return(valeur_put)
    
    def _c1(self,
            maturity: float | int,
            tenor: float | int,
            tau: float,
            matrice_swap: np.ndarray | List[float]
            ) -> np.ndarray:
        """Calculate the coefficiant c_i use in the calcul of the price of Swaption
            Args:
                self:
                maturity: Observed time to maturity 
                tenor: Observed time to tenor
                tau: step of time  
                matrice_swap: array of dimention (n x n) that contains swap 
            Returns:
                Array:
                    The vector of coefficiant C_i, with a dimension of n
        """
        length_vect_ci = int((tenor - maturity)/tau)
        vector_ci = np.array( [0]*length_vect_ci )
        vector_ci[length_vect_ci] = 1 + tau*matrice_swap[int(maturity), int(tenor-maturity)]
        if(length_vect_ci>1):
            vector_ci[0:-2] = tau * matrice_swap[int(maturity), int(tenor-maturity)]
        return(vector_ci)
    
    def _price_hw_atm(self,
                      maturity: float | int,
                      tenor: float | int,
                      tau: float,
                      pzc: np.ndarray | List[float],
                      fwd: np.ndarray | List[float],
                      matrice_swap: np.ndarray | List[float],
                      r_optimal_default: float = 0.04) -> np.ndarray:
        """Compute the zero coupon price in the Hull and White model 
            Args:
                self: 
                    parametre: list containing the parameters (alpha et sigma) that must be
                    higher than 0
                maturity: Observed time to maturity (m)
                tenor: Observed time to tenor (n)
                pzc: array of zero-coupon prices
                tau: step of time 
                matrice_swap: array of dimention (m x n) that contains swap 
            Returns:
                float:
                    the zero coupon price of the input.
        """
        vect_payment_swp = np.arange(maturity + tau, tenor, tau)
        length_payment = len(vect_payment_swp)

        vect_ci = self._c1(maturity, tenor, tau, matrice_swap)

        f_A = lambda indice: self._calcul_coef_A(maturity, vect_payment_swp[indice], tau,  fwd[int(maturity*360)+1], pzc)
        vect_A = np.array( [f_A(indice) for indice in range(length_payment)] )

        f_B = lambda indice: self._calcul_B(maturity, vect_payment_swp[indice])
        vect_B = np.array( [f_B(indice) for indice in range(length_payment)] )

        f_optimize = lambda rate: abs( np.sum(vect_ci * vect_A * np.exp(-vect_B*rate)) - 1)
        rate_optimal = optimize.minimize(f_optimize, r_optimal_default, method="BFGS").x

        strike_optimal = vect_A * np.exp(-vect_B*rate_optimal)

        f_zcp = lambda indice: self._calcul_zero_bond_put(maturity, vect_payment_swp[indice], tau, pzc, strike_optimal[indice])
        vector_price_zero_coupon = np.array( [f_zcp(indice) for indice in range(length_payment)] )
        price_zero_coupon = np.sum(vector_price_zero_coupon*vect_ci)
        return(price_zero_coupon)

    def _calcul_mse_hw_atm(self,
                           parametre: np.ndarray | List[float],
                           vect_maturity: np.ndarray | List[float],
                           vect_tenor: np.ndarray | List[float],
                           tau: float,
                           pzc: np.ndarray | List[float],
                           price_swp: np.ndarray | List[float],
                           matrice_swap: np.ndarray | List[float]
                           ) -> float:
        """Compute the least squares error
            Args:
                self: 
                parametre: list containing the parameters (alpha et sigma) that must be
                    higher than 0
                vect_maturity: vector of observed time to maturity (m)
                vect_tenor: vector of observed time to tenor (n)
                tau: step of time 
                pzc: array of zero-coupon prices
                price_swp: array of dimention (m x n) that swaption prices
                matrice_swap: array of dimention (m x n) that contains swap 
            Returns:
                float:
                    The least squares error of the input.
                    If parametres do not fit the conditions then
                    the output value is 10^20
        """
        # extraction parametre
        a = parametre[0]
        sigma = parametre[1]
        length_vect_maturity = len(vect_maturity)
        length_vect_tenor = len(vect_tenor)

        mse_hw_atm = 10^20 # default value
        mat_price_gap = np.zeros( (length_vect_maturity, length_vect_tenor) )
        #funct_type_error = self._compute_relatif_error

        if(a > 0 and sigma > 0):
            for each_maturity in length_vect_maturity:
                for each_tenor in length_vect_tenor:
                    if( np.isnan(price_swp[each_maturity, each_tenor]) ):
                        mat_price_gap[each_maturity, each_tenor] = self.compute_relatif_error(
                            self._calcul_mse_hw_atm(each_maturity, each_tenor, tau, pzc, matrice_swap),
                            price_swp[each_maturity, each_tenor])
            mse_hw_atm = np.sum(mat_price_gap)*10**10
        
        message = f"Objectif : {round(mse_hw_atm, 6)} \n Ecart relatif (en %) : {np.mean(np.sqrt(mse_hw_atm))}"
        print(message)

        return mse_hw_atm
    
    def calibration(self,
                    vect_maturity: np.ndarray | List[float],
                    vect_tenor: np.ndarray | List[float],
                    tau: float,
                    pzc: np.ndarray | List[float],
                    price_swp: np.ndarray | List[float],
                    matrice_swap: np.ndarray | List[float]
                    ) -> float:
        """solves the optimization problem for swaption in Hull and White model.
        Changes the value of the attribut in the current object: parametre
            Args:
                self: 
                    parametre: list containing the parameters (alpha et sigma) that must be
                    higher than 0
                vect_maturity: vector of observed time to maturity (m)
                vect_tenor: vector of observed time to tenor (n)
                tau: step of time 
                pzc: array of zero-coupon prices
                price_swp: array of dimention (m x n) that swaption prices
                matrice_swap: array of dimention (m x n) that contains swap 
            Returns:
                None       
        """
        initial_parametre = self.parametre
        f_optimize = lambda parametre: self._calcul_mse_hw_atm(parametre, vect_maturity, vect_tenor,
                                                               tau, pzc, price_swp, matrice_swap)
        parametre_optimal = optimize.minimize(f_optimize, initial_parametre, method="BFGS").x
        self.parametre = parametre_optimal

        return None
