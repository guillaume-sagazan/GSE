import numpy as np

def calcul_inst_fwd(pzc, dt):
    pzc_tronc = np.insert(pzc, 0, 1)[:-1]
    pzc_shifed = np.array(pzc)
    fwd = -(np.log(pzc_shifed) - np.log(pzc_tronc)) / dt
    return fwd

def compute_structure_taux(param, fwd, t):
    a, b, sigma, eta, rho = param
    phi = fwd + \
        ((sigma / a)**2 * (1 - np.exp(-a * t))**2 +\
         (eta / b)**2 * (1 - np.exp(-b * t))**2)/2 +\
        sigma * eta * rho * (1 - np.exp(-a * t)) * (1 - np.exp(-b * t)) / (a * b)
    return phi

def generate_taux(param, aleas1, aleas2, dt, fwd):
    # Affectation des paramètres
    a, b, sigma, eta, rho = param
    
    # Nombre de simulations et nombre de périodes
    N, T = aleas1.shape
    
    # Transformation aleas1 et aleas2 en deux variables aléatoire suivant loi normale centrée
    # réduite avec un coefficient de corrélation rho.
    alea1 = aleas1
    alea2 = rho * aleas1 + np.sqrt(1 - rho**2) * aleas2
    
    # Initialisation x0=0, y0=0, r0
    x = np.zeros((N, T+1))
    y = np.zeros((N, T+1))
    r = np.zeros((N, T+1))
    
    r[:, 0] = fwd[0] * np.ones(N)
    
    for i in range(1, T+1):
        # Durée de temps entre 0 et t
        t = dt * i

        int_x_1 = np.sqrt((1 - np.exp(-2 * a * dt)) / (2 * a))
        int_y_1 = np.sqrt((1 - np.exp(-2 * b * dt)) / (2 * b)) 
        # Simulation du premier facteur
        x[:, i] = x[:, i-1] * np.exp(-a * dt) + sigma * int_x_1 * alea1[:, i-1]
        
        # Simulation du deuxième facteur
        y[:, i] = y[:, i-1] * np.exp(-b * dt) + eta * int_y_1 * alea2[:, i-1]
        
        # Facteur phi
        phi = compute_structure_taux(param, fwd[i-1], t)
        
        # Taux d'intérêt
        r[:, i] = x[:, i] + y[:, i] + phi
    
    return r, x, y

def calcul_B_i_t_T(a, t, T):
    B_i_t_T = (1 - np.exp(-a*(T-t)))/a
    return B_i_t_T

def calcul_B_i_j_t_T(a, b, t, T):
    B_i_j_t_T = (1 - np.exp(-(a+b)*(T-t)))/(a + b)
    return B_i_j_t_T

def calcul_intermediaire_V(param, t, T):
    a, b, sigma, eta, rho = param
    B_1_t_T = calcul_B_i_t_T(a, t, T)
    interm_1 = (sigma/a)**2*(T-t - B_1_t_T - a * B_1_t_T**2 / 2)

    B_2_t_T = calcul_B_i_t_T(b, t, T)
    interm_2 = (eta/b)**2*(T-t - B_2_t_T - a * B_2_t_T**2 / 2)

    B_12_t_T = calcul_B_i_j_t_T(a, b, t, T)
    interm_3 = 2*sigma*eta*rho/(a*b)*(T-t - B_1_t_T - B_2_t_T + B_12_t_T)

    return interm_1 + interm_2 + interm_3

def calcul_moyenne_process(param, t, T, xt, yt):
    a, b, sigma, eta, rho = param
    mtT = xt * calcul_B_i_t_T(a, t, T) + yt * calcul_B_i_t_T(b, t, T)
    return mtT

def pricing_pzc(param, t, T, PZC_t, PZC_T, xt, yt):
    # Affectation des paramètres
    a, b, sigma, eta, rho = param
    
    # Moyenne
    mtT = calcul_moyenne_process(param, t, T, xt, yt)

    v0t = calcul_intermediaire_V(param, 0, t)
    v0T = calcul_intermediaire_V(param, 0, T)
    vtT = calcul_intermediaire_V(param, t, T)

    # Prix du zero-coupon
    PZC = np.exp(-mtT + 0.5 * (vtT + v0t - v0T)) * PZC_T / PZC_t
    #print(f"mtT: {mtT}, v0t: {v0t}, v0T: {v0T}, vtT: {vtT}, PZC: {PZC}")
    
    return PZC

def calculate_prices_zc(param, pzc, xt, yt):
    step_of_time = 1/12
    number_of_simulation = 1000
    number_of_year_maturity = 40
    year_projected = 8

    mat_pzc = np.zeros(
        shape=(number_of_year_maturity, year_projected, number_of_simulation)
        )

    year_annual_projected = np.arange(
        int(1/step_of_time),
        int((year_projected + 1)/step_of_time),
        int(1/step_of_time))
    year_annual_maturity = np.arange(
        int(1/step_of_time),
        int((number_of_year_maturity + 1)/step_of_time),
        int(1/step_of_time))
    
    for indice_tenor, tenor in enumerate(year_annual_projected):
        PZC_T = pzc[tenor]
        for indice_maturity, maturity in enumerate(year_annual_maturity):
            PZC_t = pzc[maturity-1]
            mat_pzc[indice_maturity, indice_tenor, :] = \
                pricing_pzc(
                    param,
                    tenor,
                    maturity + tenor, 
                    PZC_T,
                    PZC_t,
                    xt[:, tenor-1], yt[:, tenor-1]
                    )
    return mat_pzc

    



if __name__ == '__main__':
    # Exemple d'utilisation
    param = [0.1, 0.2, 0.3, 0.4, 0.5]
    t = 0.1
    T = 1.0
    PZC_t = 0.95
    PZC_T = 1.0
    xt = 0.02
    yt = 0.03

    resultat_pzc = pricing_zc(param, t, T, PZC_t, PZC_T, xt, yt)
    print(resultat_pzc)