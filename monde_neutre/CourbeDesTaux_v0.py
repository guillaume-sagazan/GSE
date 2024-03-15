import pandas as pd
import numpy as np
import logging
from scipy import optimize
from scipy.interpolate import splrep, splev, BSpline
from typing import Union, List, Optional

class CourbeDesTaux:
    
    def __init__(self, maturity, zc, is_price) -> None:
        self.maturity = maturity # array of maturity for Zero-coupon prices observe
        self.zc = zc # array of Zero-coupon (rates or prices) observe 
        self.is_price = is_price # boolean that indicates if the zero coupon is a price or a rate
        self.type_of_methode = '' # type of method uses to extrapolate the Zero-coupon price

    def get_maturity(self):
        return self.maturity
    
    def get_zc(self):
        return self.zc
    
    def get_is_price(self):
        return self.is_price
    
    def get_zc_rates_extrapolate(self):
        return self.zc_rates_extrapolate
    
    def get_zc_prices_extrapolate(self):
        return self.zc_prices_extrapolate
    
    def get_maturity_extrapolate(self):
        return self.maturity_extrapolate
    
    def calculate_prices(self, is_extrapolate: bool = False
                         ) -> np.ndarray:
        """Calculate prices from zero-coupon rates
        Args:
            self :
                rates: zero-coupon rates vector of length n
                maturity: time to maturity vector (in years) of length n
        Returns:
            Prices as vector of length n
        """
        try:
            # Convert list into numpy array
            if is_extrapolate:
                rates = np.array(self.zc_rates_extrapolate)
                maturity = np.array(self.maturity_extrapolate)
            else:
                rates = np.array(self.zc)
                maturity = np.array(self.maturity)
            if len(rates) != len(maturity):
                raise ValueError('The courbe of rates does not have the same length as the maturities')
            prices_array = np.power(1 + rates, -maturity)
            self.price_zc = prices_array
        except Exception as e:
            message_error = f"fonction appelée calculate_prices"
            message_error += f"\n {e}"  
            print(message_error)
        return prices_array
    
    def calculate_rates(self, is_extrapolate: bool
                        ) -> np.ndarray:
        """Calculate rates from zero-coupon prices
        Args:
            self :
                rates: zero-coupon prices vector of length n
                maturity: time to maturity vector (in years) of length n
            is_extrapolate : indicate if the data to calcul is the data extrapolate (True) or observe (False)
        Returns:
            rates as vector of length n
        """
        try:
            # Convert list into numpy array
            if is_extrapolate:
                prices = np.array(self.zc_prices_extrapolate)
                maturity = np.array(self.maturity_extrapolate)
            else:
                prices = np.array(self.zc)
                maturity = np.array(self.maturity)

            if len(prices) != len(maturity):
                raise ValueError('The courbe of prices does not have the same length as the maturities')
            
            rates_array = np.power(prices, -1/maturity) - 1
            self.rates_zc = rates_array
        except Exception as e:
            message_error = f"fonction appelée calculate_rates"
            message_error += f"\n {e}"  
            print(message_error)
        return rates_array

    def spline_funct(value_to_extrapolate: np.ndarray | List[float],
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
        tck = splrep(x=maturity_to_extrapolate, y=value_to_extrapolate, s=0)
        new_value_after_extrapolation = splev(new_maturity_after_extrapolation, tck, der=0)
        return new_value_after_extrapolation
    
class SmithWilson(CourbeDesTaux):
    
    def __init__(self, maturity, zc, is_price, maturity_extrapolate, ufr, alpha,
                 zc_rates_extrapolate = [], zc_prices_extrapolate = []) -> None:
        super().__init__(maturity, zc, is_price)
        self.type_of_methode = 'Smith_Wilson' # type of method uses to extrapolate the Zero-coupon price
        self.zc_rates_extrapolate = zc_rates_extrapolate
        self.zc_prices_extrapolate = zc_prices_extrapolate
        self.maturity_extrapolate = maturity_extrapolate # array of maturity for Zero-coupon prices at extrapolate
        self.ufr = ufr # Ultimate forward rate
        self.alpha = alpha # speed of convergence
    
    def get_type_of_methode(self):
        return self.type_of_methode
    
    def get_ufr(self):
        return self.ufr
    
    def get_alpha(self):
        return self.alpha

    def ufr_discount_factor(self,
                            maturity_obs: Union[np.ndarray, List[float]]
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
        try:
            # Convert annualized ultimate forward rate to log-return
            log_ufr = np.log(1 + self.ufr)

            # Convert list into numpy array
            maturity_array = np.array(maturity_obs)
        except Exception as e:
            message_error = f"fonction appelée ufr_discount_factor"
            message_error += f"\n {e}"  
            print(message_error)
        return np.exp(-log_ufr * maturity_array)


    def wilson_function(self,
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
        try:
            length_t1 = len(t1)
            length_t2 = len(t2)
            t1_Mat = np.repeat(t1, length_t2, axis=1)
            t2_Mat = np.repeat(t2, length_t1, axis=1).T

            # Calculate the minimum and maximum of the two matrices
            min_t = np.minimum(t1_Mat, t2_Mat)
            max_t = np.maximum(t1_Mat, t2_Mat)

            # Calculate the UFR discount factor - p.16
            ufr_disc = self.ufr_discount_factor(maturity_obs=(t1_Mat + t2_Mat))
            W = ufr_disc * (self.alpha * min_t - 0.5 * np.exp(-self.alpha * max_t) * \
                (np.exp(self.alpha * min_t) - np.exp(-self.alpha * min_t)))
        except Exception as e:
            message_error = f"fonction appelée wilson_function"
            message_error += f"\n {e}"  
            logging.error(message_error)
            raise ValueError(message_error)
        return W
    
    def fit_parameters(self) -> np.ndarray:
        """Calculate Smith-Wilson parameter vector ζ
        Given the Wilson-matrix, vector of discount factors and prices,
        the parameter vector can be calculated as follows (p.17):
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
        try:
            W = self.wilson_function(t1=self.maturity, t2=self.maturity)
            mu = self.ufr_discount_factor(maturity_obs=self.maturity)
            if not(self.is_price):
                Pzc = self.calculate_prices(is_extrapolate=False)
            else:
                Pzc = self.zc
            # Calculate vector of parameters (p. 17)
            # To invert the Wilson-matrix, conversion to type matrix is required
            zeta = np.matrix(W).I * (mu - Pzc)
            zeta = np.array(zeta)     # Convert back to more general array type
        except Exception as e:
            message_error = f"fonction appelée fit_parameters"
            message_error += f"\n {e}"  
            print(message_error)
        return zeta

    def fit_smithwilson_rates(self) -> np.ndarray:
        """Calculate zero-coupon yields with Smith-Wilson method based on observed rates.
        This function expects the rates and initial maturity vector to be
        before the Last Liquid Point (LLP). The targeted maturity vector can
        contain both, more granular maturity structure (interpolation) or terms after
        the LLP (extrapolation).
        The Smith-Wilson method calculated first the Wilson-matrix (p. 16):
            W = e^(-UFR * (t1 + t2)) * (α * min(t1, t2) - 0.5 * e^(-α * max(t1, t2))
                * (e^(α * min(t1, t2)) - e^(-α * min(t1, t2))))
        Given the Wilson-matrix, vector of discount factors and prices,
        the parameter vector can be calculated as follows (p.17):
            ζ = W^-1 * (μ - P)
        With the Smith-Wilson parameter and Wilson-matrix, the zero-coupon bond
        prices can be represented as (p. 18) in matrix notation:
            P = e^(-t * UFR) - W * zeta
        In the last case, t can be any maturity vector
        Source: EIOPA QIS 5 Technical Paper; Risk-free interest rates – Extrapolation method
        Args:
            self
                zc : Initially observed zero-coupon rates vector before LLP of length n
                maturity : Initially observed time to maturity vector (in years) of length n
                ufr: Ultimate Forward Rate (annualized/annual compounding)
                alpha: Convergence speed parameter. If not provided estimated using
                maturity_extrapolate: Targeted maturity vector (in years) with interpolated/extrapolated terms
                the `fit_convergence_parameter()` function
        Returns:
            None but update : zc_rates_extrapolate and zc_price_extrapolate
        """

        # Convert list to numpy array and use reshape to convert from 1-d to 2-d array
        # E.g. reshape((-1, 1)) converts an input of shape (10,) with second dimension
        # being empty (1-d vector) to shape (10, 1) where second dimension is 1 (2-d vector)
        try:
            self.zc = np.array(self.zc).reshape((-1, 1))
            self.maturity = np.array(self.maturity).reshape((-1, 1))
            self.maturity_extrapolate = np.array(self.maturity_extrapolate).reshape((-1, 1))

            zeta = self.fit_parameters()
            ufr_disc = self.ufr_discount_factor(maturity_obs=self.maturity_extrapolate)
            W = self.wilson_function(t1=self.maturity_extrapolate, t2=self.maturity)

            # Price vector - equivalent to discounting with zero-coupon yields 1/(1 + r)^t
            # for prices where self.maturity = self.maturity_extrapolate. All other matuirites are interpolated or extrapolated
            result_prices_vector = ufr_disc - W @ zeta     # '@' in numpy is the dot product of two matrices
            result_rates_vector = np.power(1 / result_prices_vector, 1 / self.maturity_extrapolate) - 1 # Transform price vector to zero-coupon rate vector (1/P)^(1/t) - 1
            self.zc_rates_extrapolate = result_rates_vector
            self.zc_prices_extrapolate = result_prices_vector
            
        except Exception as e:
            message_error = f"fonction appelée fit_smithwilson_rates"
            message_error += f"\n {e}"  
            print(message_error)
        return None
    

if "__main__" == __name__:
    # exemple
    rates = [
        -0.00803, -0.00814, -0.00778, -0.00725, -0.00652,
        -0.00565, -0.0048, -0.00391, -0.00313, -0.00214,
        -0.0014, -0.00067, -0.00008, 0.00051, 0.00108,
        0.00157, 0.00197, 0.00228, 0.0025, 0.00264,
        0.00271, 0.00274, 0.0028, 0.00291, 0.00309
        ]
    terms = [float(y + 1) for y in range(len(rates))] # [1.0, 2.0, ..., 25.0]
    ufr = 0.029
    alpha = 0.128562
    # Extrapolate to 150 years
    terms_target = [float(y + 1) for y in range(150)]
    extrapolation = SmithWilson(
        maturity=terms,
        zc=rates,
        is_price=False, maturity_extrapolate=terms_target, ufr=ufr, alpha=alpha)
    resultat_fonct = extrapolation.fit_smithwilson_rates()
    print( extrapolation.zc_rates_extrapolate )
    print( len(extrapolation.zc) )
    print( extrapolation.calculate_prices(is_extrapolate=True) )

    len(terms)

