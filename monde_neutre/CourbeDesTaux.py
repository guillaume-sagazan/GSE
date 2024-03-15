import pandas as pd
import numpy as np
import logging
from scipy import optimize
from typing import Union, List, Optional
import copy

class CourbeDesTaux:
    
    def __init__(
            self,
            maturity_obs: np.ndarray | List[float],
            zc: np.ndarray,
            is_price: bool,
            period: str,
            max_maturity_to_extrapolate: int | float,
            type_of_methode: str,
            ufr: float = None,
            alpha: float = None
            ) -> None:
        # Configuring logging
        logging.basicConfig(
            filename='log.txt',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
        # Parametrisation
        self.maturity_obs = maturity_obs # array of maturity for Zero-coupon prices observe
        self.zc = zc # array of Zero-coupon (rates or prices) observe
        # boolean that indicates if the zero coupon is a price (True) or a rate (False)
        self.is_price = is_price 
        #type of method uses to extrapolate the Zero-coupon price (Smith-Wilson, MPL)
        self.type_of_methode = type_of_methode 
                                               
        self.period = period # choose between (annual, quarterly, half-yearly, daily)
        # indicate the maturity maximal to extrapolat
        self.max_maturity_to_extrapolate = max_maturity_to_extrapolate 
        self.is_setup = False # indicate if the object has use the setup methode
        self.ufr = ufr
        self.alpha = alpha

    def get_maturity_obs(self):
        return self.maturity_obs
    
    def get_zc(self):
        return self.zc
    
    def get_is_price(self):
        return self.is_price
    
    def get_period(self):
        return self.period
    
    def get_is_setup(self):
        return self.is_setup
    
    def get_max_maturity_to_extrapolate(self):
        return self.max_maturity_to_extrapolate

    def get_zc_rates_extrapolate(self):
        return self.zc_rates_extrapolate
    
    def get_zc_prices_extrapolate(self):
        return self.zc_prices_extrapolate
    
    def get_maturity_extrapolate(self):
        return self.maturity_extrapolate
    
    def get_type_of_methode(self):
        return self.type_of_methode
    
    def get_ufr(self):
        return self.ufr
    
    def get_alpha(self):
        return self.alpha
    
    def _initiate_period(self) -> None:
        logging.info("Initialisation")
        period_map = {
            "annual": 1.0,
            "half-yearly": 0.5,
            "quarterly": 0.25,
            "monthly": 1/12,
            "daily": 1/360
            }
        if self.period in period_map.keys():
            self.freq = period_map[self.period]
            logging.info("Finished")
        else:
            logging.error(f"There is not such period {self.period}")
            raise ValueError(f"There is not such period {self.period}")
    
    def _initiate_extrapolation_method(self) -> None:
        logging.info("Initialisation")
        extrapolation_method_map = {
            "Smith-Wilson": self._fit_smithwilson_rates,
            "MPL": NotImplemented
            }
        if self.type_of_methode in extrapolation_method_map.keys():
            self.extrapolation_method = extrapolation_method_map[self.type_of_methode]
            logging.info("Finished")
        else:
            message_error = f"There is not such extrapolation" +\
                f" method {self.type_of_methode}."
            logging.error(message_error)
            raise ValueError(message_error)
    
    def _initiate_maturity_to_extrapolate(self):
        logging.info("Initialisation")
        if self.max_maturity_to_extrapolate > 0:
            self.maturity_extrapolate = np.arange(
                self.freq,
                self.max_maturity_to_extrapolate + self.freq, self.freq)
            self.maturity_extrapolate = np.array(self.maturity_extrapolate).reshape((-1, 1))
            logging.info("Finished")
        else:
            message_error = f"Wrong choice of maximal maturity" +\
                f" = {self.max_maturity_to_extrapolate}." + \
                "It can not be null or negative."
            logging.error(message_error)
            raise ValueError(message_error)

    def _calculate_prices(self) -> np.ndarray:
        """Calculate prices from zero-coupon rates
        Args:
            self :
                zc: zero-coupon rates vector of length n
                maturity: time to maturity vector (in years) of length n
        Returns:
            Prices as vector of length n
        """
        logging.info("Initialisation")
        try:
            rates = np.array(self.zc).flatten()
            maturity = np.array(self.maturity_obs).flatten()
            # Convert list into numpy array
            if len(rates) != len(maturity):
                message_error = 'The courbe of prices does not have" +\
                    " the same length as the maturities.'
                logging.error(message_error)
                raise ValueError(message_error)
            prices_array = np.power(1 + rates, -maturity)
            logging.info("Finished")
        except Exception as e:
            message_error = f"fonction appelée _calculate_prices"
            message_error += f"\n {e}"  
            logging.error(message_error)
            raise ValueError(message_error)
        return prices_array
    
    def _calculate_prices_extrapolate(self) -> np.ndarray:
        """Calculate prices from zero-coupon rates extrapolates
        Args:
            self :
                zc_rates_extrapolate: zero-coupon rates vector of length n
                maturity_extrapolate: time to maturity vector of length n
        Returns:
            Prices as vector of length n
        """
        logging.info("Initialisation")
        try:
            # Convert list into numpy array
            rates = np.array(self.zc_rates_extrapolate).flatten()
            maturity = np.array(self.maturity_extrapolate).flatten()
            if len(rates) != len(maturity):
                message_error = 'The courbe of prices does not have" +\
                    " the same length as the maturities.'
                logging.error(message_error)
                raise ValueError(message_error)
            prices_array = np.power(1 + rates, -maturity)
            logging.info("Finished")
        except Exception as e:
            message_error = f"fonction appelée _calculate_prices"
            message_error += f"\n {e}"  
            logging.error(message_error)
            raise ValueError(message_error)
        return prices_array
    
    def _calculate_rates(self) -> np.ndarray:
        """Calculate rates from zero-coupon prices
        Args:
            self :
                zc: zero-coupon prices vector of length n
                maturity: time to maturity vector (in years) of length n
            is_extrapolate : indicate if the data to calcul is the data extrapolate (True) or observe (False)
        Returns:
            rates as vector of length n
        """
        logging.info("Initialisation")
        try:
            # Convert list into numpy array
            prices = np.array(self.zc).flatten()
            maturity = np.array(self.maturity_obs).flatten()
            if len(prices) != len(maturity):
                message_error = 'The courbe of prices does not have" +\
                    " the same length as the maturities.'
                logging.error(message_error)
                raise ValueError(message_error)
            rates_array = np.power(prices, -1/maturity) - 1
            logging.info("Finished")
        except Exception as e:
            message_error = f"fonction appelée _calculate_prices"
            message_error += f"\n {e}"  
            logging.error(message_error)
            raise e
        return rates_array

    def set_up(self) -> None:
        """Initiate the object in order to give specific method"""
        logging.info("Initialisation")
        try:
            self.zc = np.array(self.zc).reshape((-1, 1))
            self.maturity_obs = np.array(self.maturity_obs).reshape((-1, 1))
            
            if self.is_price:
                self.zc_prices_obs = self.zc
                self.zc_rates_obs = self._calculate_rates()
            else:
                self.zc_rates_obs = self.zc
                self.zc_prices_obs = self._calculate_prices()
            self._initiate_period()
            self._initiate_extrapolation_method()
            self._initiate_maturity_to_extrapolate()
            # Set default value to attribut
            self.zc_rates = self.zc_rates_obs.flatten()
            self.zc_prices = self.zc_prices_obs.flatten()
            self.maturity = self.maturity_obs.flatten()
            self.is_setup = True
            logging.info("Finished")
        except Exception as e:
            logging.error(f"There is an error: {e}")
            raise ValueError(e)

    def spline_funct(self,
                     value_to_extrapolate: np.ndarray | List[float],
                     maturity_to_extrapolate: np.ndarray | List[float],
                     new_maturity_after_extrapolation: np.ndarray | List[float]
                     )-> np.ndarray:
        """Extrapolate values from 'value_to_extrapolate' for the maturity 'maturity_to_extrapolate'
        for the new values of maturity : new_maturity_after_extrapolation according the spline cubic
        method.
            Args:
                value_to_extrapolate: array of values to extrapolate dimension (n)
                maturity_to_extrapolate: array of maturity associated to value to extrapolate dimension (n)
                new_maturity_after_extrapolation: array of maturity of new values of maturity (m) such as
                m >= n
            Returns:
                np.array:
                    Values of the new value after extrapolation with the same dimension as 
                    new_maturity_after_extrapolation (m)
        """
        logging.info("Initialisation")
        from scipy.interpolate import splrep, splev
        tck = splrep(x=maturity_to_extrapolate, y=value_to_extrapolate, s=0)
        new_value_after_extrapolation = splev(new_maturity_after_extrapolation, tck, der=0)
        logging.info("Finished")
        return new_value_after_extrapolation
    
    def _ufr_discount_factor(self,
                            maturity: Union[np.ndarray, List[float]]
                            ) -> np.ndarray:
        """Calculate Ultimate Forward Rate (UFR) discount factors.
        Takes the UFR with a vector of maturities and returns for each of the
        maturities the discount factor
            d_UFR = e^(-UFR * t)
        Note that UFR is expected to be annualy compounded and that
        this function performs the calculation of the log return prior
        to applying the formula above.
        Args:
            ufr: Ultimate Forward Rate (annualized/annual compounding)
            maturity_obs: time to maturity vector (in years) of length n
        Returns:
            UFR discount factors as vector of length n
        """
        logging.info("Initialisation")
        try:
            # Convert annualized ultimate forward rate to log-return
            log_ufr = np.log(1 + self.ufr)
            # Convert list into numpy array
            maturity_array = np.array(maturity)
            logging.info("Finished")
        except Exception as e:
            message_error = f"fonction appelée _ufr_discount_factor"
            message_error += f"\n {e}"  
            logging.error(message_error)
            raise ValueError(message_error)
        return np.exp(-log_ufr * maturity_array)

    def _wilson_function(
            self,
            t1: Union[np.ndarray, List[float]],
            t2: Union[np.ndarray, List[float]]
            ) -> np.ndarray:
        """Calculate matrix of Wilson functions
        The Smith-Wilson method requires the calculation of a series of Wilson
        functions. The Wilson function is calculated for each maturity combination
        t1 and t2. If t1 and t2 are scalars, the result is a scalar. If t1 and t2
        are vectors of shape (m, 1) and (n, 1), then the result is a matrix of
        Wilson functions with shape (m, n) as defined on p. 16:
            W = e^(-UFR * (t1 + t2)) * (α * min(t1, t2) - 0.5 * e^(-α * max(t1, t2))
                * (e^(α * min(t1, t2)) - e^(-α * min(t1, t2))))
        Source: EIOPA QIS 5 Technical Paper; Risk-free interest rates – Extrapolation method;
        Args:
            self : 
                alpha: Convergence speed parameter
                ufr: Ultimate Forward Rate (annualized/annual compounding)
            t1: time to maturity vector of length m
            t2: time to maturity vector of length n
        Returns:
            Wilson-matrix of shape (m, n) as numpy matrix
        """

        # Take time vectors of shape (nx1) and (mx1) and turn them into matrices of shape (mxn).
        # This is achieved by repeating the vectors along the axis 1. The operation is required
        # because the Wilson function needs all possible combinations of maturities to construct
        # the Wilson matrix
        logging.info("Initialisation")
        try:
            length_t1 = len(t1)
            length_t2 = len(t2)
            t1_Mat = np.repeat(t1, length_t2, axis=1)
            t2_Mat = np.repeat(t2, length_t1, axis=1).T

            # Calculate the minimum and maximum of the two matrices
            min_t = np.minimum(t1_Mat, t2_Mat)
            max_t = np.maximum(t1_Mat, t2_Mat)

            # Calculate the UFR discount factor - p.16
            ufr_disc = self._ufr_discount_factor(maturity=(t1_Mat + t2_Mat))
            W = ufr_disc * (self.alpha * min_t - 0.5 * np.exp(-self.alpha * max_t) * \
                (np.exp(self.alpha * min_t) - np.exp(-self.alpha * min_t)))
            logging.info("Finished")
        except Exception as e:
            message_error = f"fonction appelée _wilson_function"
            message_error += f"\n {e}"  
            logging.error(message_error)
            raise ValueError(e)
        return W
    
    def _fit_parameters(self) -> np.ndarray:
        """Calculate Smith-Wilson parameter vector ζ
        Given the Wilson-matrix, vector of discount factors and prices,
        the parameter vector can be calculated as follows:
            ζ = W^-1 * (μ - P)
        Source: EIOPA QIS 5 Technical Paper; Risk-free interest rates – Extrapolation method
        Args:
            self :
                zc : Observed zero-coupon rates or price vector of length n
                maturity: Observed time to maturity vector (in years) of length n
                alpha: Convergence speed parameter
                ufr: Ultimate Forward Rate (annualized/annual compounding)
                is_price: indicate if the zc is a Price (True) or a Rate
        Returns:
            Wilson-matrix of shape (m, n) as numpy matrix
        """
        # Calcualte square matrix of Wilson functions, UFR discount vector and price vector
        # The price vector is calculated with zero-coupon rates and assumed face value of 1
        # For the estimation of zeta, t1 and t2 correspond both to the observed maturities
        logging.info("Initialisation")
        try:
            W = self._wilson_function(t1=self.maturity_obs, t2=self.maturity_obs)
            mu = self._ufr_discount_factor(maturity=self.maturity_obs)
            Pzc = self.zc_prices_obs
            # Calculate vector of parameters
            # To invert the Wilson-matrix, conversion to type matrix is required
            zeta = np.matrix(W).I * (mu - Pzc)
            zeta = np.array(zeta)     # Convert back to more general array type
            logging.info("Finished")
        except Exception as e:
            message_error = f"fonction appelée _fit_parameters"
            message_error += f"\n {e}"  
            logging.error(message_error)
            raise ValueError(e)
        return zeta

    def _fit_smithwilson_rates(self):
        """Calculate zero-coupon yields with Smith-Wilson method based on observed rates.
        This function expects the rates and initial maturity vector to be
        before the Last Liquid Point (LLP). The targeted maturity vector can
        contain both, more granular maturity structure (interpolation) or terms after
        the LLP (extrapolation).
        The Smith-Wilson method calculated first the Wilson-matrix:
            W = e^(-UFR * (t1 + t2)) * (α * min(t1, t2) - 0.5 * e^(-α * max(t1, t2))
                * (e^(α * min(t1, t2)) - e^(-α * min(t1, t2))))
        Given the Wilson-matrix, vector of discount factors and prices,
        the parameter vector can be calculated as follows (p.17):
            ζ = W^-1 * (μ - P)
        With the Smith-Wilson parameter and Wilson-matrix, the zero-coupon bond
        prices can be represented as in matrix notation:
            P = e^(-t * UFR) - W * zeta
        In the last case, t can be any maturity vector
        Source: EIOPA QIS 5 Technical Paper; Risk-free interest rates – Extrapolation method
        Args:
            self
                zc : Initially observed zero-coupon rates vector before LLP of length n
                maturity : Initially observed time to maturity vector (in years) of length n
                ufr: Ultimate Forward Rate (annualized/annual compounding)
                alpha: Convergence speed parameter. If not provided estimated using
                maturity_extrapolate: Targeted maturity vector (in years) with
                    interpolated/extrapolated terms
                the `fit_convergence_parameter()` function
        Returns:
            add the attributs zc_rates_extrapolate and zc_price_extrapolate at self
            and update the attributs zc_rates and zc_prices of self
        """

        # Convert list to numpy array and use reshape to convert from 1-d to 2-d array
        # E.g. reshape((-1, 1)) converts an input of shape (10,) with second dimension
        # being empty (1-d vector) to shape (10, 1) where second dimension is 1 (2-d vector)
        logging.info("Initialisation")
        try:
            if self.is_setup:
                zeta = self._fit_parameters()
                ufr_disc = self._ufr_discount_factor(maturity=self.maturity_extrapolate)
                W = self._wilson_function(t1=self.maturity_extrapolate, t2=self.maturity_obs)

                # Price vector - equivalent to discounting with zero-coupon yields 1/(1 + r)^t
                # for prices where self.maturity = self.maturity_extrapolate.
                # All other matuirites are interpolated or extrapolated
                result_prices_vector = ufr_disc - W @ zeta     # '@' in numpy is the dot product of two matrices
                # Transform price vector to zero-coupon rate vector (1/P)^(1/t) - 1
                result_rates_vector = np.power(1 / result_prices_vector, 1 / self.maturity_extrapolate) - 1
                # Set attribut value of extrapolation and thus who are used
                self.zc_rates_extrapolate = result_rates_vector.flatten()
                self.zc_prices_extrapolate = result_prices_vector.flatten()

                self.zc_rates = np.copy(self.zc_rates_extrapolate)
                self.zc_prices = np.copy(self.zc_prices_extrapolate)
                self.maturity = np.copy(self.maturity_extrapolate)

                logging.info("Finished")
            else:
                message_error = "The object have to be set up, " +\
                    "with set_up method, before to execute: extrapolation"
                logging.error(message_error)
                raise ValueError(message_error)
        except Exception as e:
            message_error = f"fonction appelée _fit_smithwilson_rates"
            message_error += f"\n {e}"  
            logging.error(message_error)
            raise ValueError(e)
        
    
    def extrapolation(self) -> None:
        """execute the extrapolation method"""
        logging.info("Initialisation")
        try:
            self.extrapolation_method()
            logging.info("Finished")
        except Exception as e:
            message_error = f"fonction appelée extrapolation"
            message_error += f"\n {e}"  
            logging.error(message_error)
            raise ValueError(e)
    
    def get_forward_rate(self):
        """Compute the forward rate transformation in function of the steps
        for each value of the zero coupon price in input.
        Using the attribute "zc_prices" implies that in order to obtain
        the forward rates of extrapolated zero coupon, 
        you have to execute "extrapolate" first, and conversely, use the "set_up" method.
            Args:
                self:
                    zc_prices: vector contains the price of zero coupon
                    freq: float defines as the steps 
                    (in the fraction of the year) of the forward rate
                        ie: if tau = 0.5 then it means semestrial
            Returns:
                Update and add the attribut fwd_rates at self
        """
        try:
            if self.is_setup:
                if self.freq <= 0:
                    message_error = f" the value of tau is {self.freq}." +\
                        " It can not be negative or nul."
                    logging.error(message_error)
                    raise ValueError(message_error)
                if len(self.zc_prices) <= 1:
                    message_error = f" the length of price zero coupon" +\
                        " is nul or with one element."
                    logging.error(message_error)
                    raise ValueError(message_error)
                if type(self.zc_prices) is list:
                    self.zc_prices = np.array(self.zc_prices)
                p_pzc = -np.log(self.zc_prices)
                length_pzc = len(self.zc_prices)
                forward_rate = (p_pzc[1:length_pzc] - p_pzc[0:length_pzc - 1])/self.freq
                self.fwd_rates = forward_rate
            else:
                message_error = "The object have to be set up, " +\
                    "with set_up method, before to execute: get_forward_rate"
                logging.error(message_error)
                raise ValueError(message_error)
        except Exception as e:
            message_error = f"fonction appelée get_forward_rate"
            message_error += f"\n {e}"  
            logging.error(message_error)
            raise ValueError(e)

    def get_instant_forward_rate(self):
        """Compute the instant forward rate transformation in function of the steps
        for each value of the zero coupon price in input
        Using the attribute "zc_prices" implies that in order to obtain
        the forward rates of extrapolated zero coupon, 
        you have to execute "extrapolate" first, and conversely, use the "set_up" method.
            Args:
                self:
                    zc_prices: vector contains the price of zero coupon
            Returns:
                Update and add the attributs at self :
                    inst_fwd_rates that correspond in terms of attributs 'period', 
                    inst_fwd_rates_annual for annual step,
                    inst_fwd_rates_monthly for monthly step, 
                    inst_fwd_rates_daily for daily step
        """
        try:
            if self.is_setup:
                step = int(self.freq * 360)
                maturity_to_take_account = np.arange(
                    int(0),
                    int(self.max_maturity_to_extrapolate * 360),
                    step)
                maturity_annual = np.arange(
                    int(0),
                    int(self.max_maturity_to_extrapolate * 360), 360)
                maturity_monthly = np.arange(
                    int(0),
                    int(self.max_maturity_to_extrapolate * 360), 30)

                self_instant = copy.copy(self)
                self_instant.period = "daily"
                self_instant.set_up()
                self_instant.extrapolation()
                self_instant.get_forward_rate()

                self.inst_fwd_rates = self_instant.fwd_rates[maturity_to_take_account]
                self.inst_fwd_rates_annual = self_instant.fwd_rates[maturity_annual]
                self.inst_fwd_rates_monthly = self_instant.fwd_rates[maturity_monthly]
                self.inst_fwd_rates_daily = self_instant.fwd_rates
            else:
                message_error = "The object have to be set up, with set_up method," +\
                    " before to execute : get_instant_forward_rate"
                logging.error(message_error)
                raise ValueError(message_error)
        except Exception as e: 
            logging.info( f"{maturity_annual}" )
            logging.error(e)
            raise ValueError(e)
                

