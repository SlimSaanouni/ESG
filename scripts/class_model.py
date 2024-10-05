import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

ASSET_MODELS    = { "Interest rates": ["Vasicek", "G2++"],
                    "Equity"        : ["Black-Scholes", "Dupire", "Heston"],
                    "Real Estate"   : ["Black-Scholes"],
                    "Inflation"     : ["Vasicek"]}

MODEL_TYPE      = { "Vasicek"       : "price",
                    "G2++"          : "price",
                    "Black-Scholes" : "index",
                    "Dupire"        : "index",
                    "Heston"        : "index"}

class Model:
    def __init__(self, name):
        self.name = name
        self.parameters = {}

    def vasicek_pricing(self, model_params, market_data):
        """
        Calcule les prix des obligations zéro-coupon sous le modèle de Vasicek pour une courbe de taux donnée.
        
        :param kappa: Vitesse de réversion
        :param theta: Niveau de réversion à long terme
        :param sigma: Volatilité
        :param spot_rates: DataFrame contenant les taux spot pour chaque maturité (index : maturités en années)
        :return: DataFrame avec les prix des obligations zéro-coupon pour chaque maturité
        """
        # Parameters of the model
        kappa = model_params[0]
        theta = model_params[1]
        sigma = model_params[2]
        r0    = model_params[3]

        # Market data
        maturities = market_data.index
        
        list_zc_prices = []
        # Vasicek features
        for T in maturities:            
            # Paramètres du modèle de Vasicek pour calculer le prix du zéro-coupon
            B = (1 - np.exp(-kappa * T)) / kappa
            A = (theta - (sigma**2) / (2 * kappa**2)) * (B - T) - (sigma**2) * (B**2) / (4 * kappa)
            
            # Calcul du prix du zéro-coupon sous le modèle de Vasicek
            zc_price = np.exp(A - B * r0)
            list_zc_prices.append(zc_price)

        #print(zero_coupon_prices)
        zc_df = pd.Series(data = list_zc_prices, index=maturities)

        return zc_df
    
    def blackscholes_pricing(self, model_params, market_data):
        """
        Calcule le prix d'une option call ou put avec le modèle Black-Scholes.
        
        :param model_params: Dictionnaire contenant les paramètres du modèle ('sigma')
        :param market_data: DataFrame avec les données de marché (doit contenir les colonnes 'S', 'K', 'r', 'T', 'option_type')
        :return: DataFrame contenant les prix modélisés
        """
        # Parameters of the model
        sigma = model_params[0]

        # Market data
        S = market_data['S']
        K = market_data['K']
        r = market_data['r']
        T = market_data['T']
        option_type = market_data['option_type']

        # BS features
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if all(option_type == "call"):
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        elif all(option_type == "put"):
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.cdf(-d1)
        else:
            raise ValueError("Le type d'option doit être 'call' ou 'put', et unique.")

        return price

    def vasicek_projection(self, model_params, Tmax, N, spot_rates, dt = 1 / 12):
        """
        Projette les taux d'intérêt sous le modèle de Vasicek pour des maturités allant de 1 à 40 ans, sur 1000 simulations.
        
        :param kappa: Vitesse de réversion
        :param theta: Niveau de réversion à long terme
        :param sigma: Volatilité
        :param spot_rates: DataFrame contenant les taux spot pour les maturités (les index sont les maturités)
        :param N_simulations: Nombre de simulations
        :return: DataFrame contenant les taux d'intérêt projetés pour chaque simulation et chaque maturité
        """

        # Parameters of the model
        kappa = model_params['kappa']
        theta = model_params['theta']
        sigma = model_params['sigma']
        
        nb_steps = int(Tmax / dt)

        time_idx = range(1, Tmax + 1, 1)
        rfr_spot = pd.DataFrame(data = spot_rates(time_idx), index = time_idx, columns = ["IR"])

        maturities = rfr_spot.index  # Récupère les maturités à partir des index du DataFrame spot_rates

        dict_paths = {}

        # Loop on maturities        
        for T in maturities:
            # Initialisation of spot rate
            r_0 = rfr_spot.loc[T]
            # Initialization of forcast matrix
            paths = np.zeros((N, nb_steps + 1))
            # Same initial value for all simulation
            paths[:, 0] = r_0

            # Moments adjustment
            Z = np.random.standard_normal(N)
            Z_list = Z.tolist()
            Z = (Z - np.mean(Z_list)) / np.std(Z_list)
            
            # Projeter l'évolution du taux pour cette maturité
            for t in range(1, nb_steps + 1):
                paths[:, t] = paths[:, t - 1] + kappa * (theta - paths[:, t - 1]) * dt + sigma * np.sqrt(dt) * Z
            
            # Création d'un DataFrame : chaque ligne est une simulation, chaque colonne est un pas de temps
            df_paths = pd.DataFrame(paths, columns=np.linspace(0, Tmax, nb_steps + 1), index = range(1, N + 1))
            df_paths = df_paths[range(0, Tmax + 1)]

            # Vasicek parameters
            B = (1 - np.exp(-kappa * T)) / kappa
            A = (theta - (sigma**2) / (2 * kappa**2)) * (B - T) - (sigma**2) * (B**2) / (4 * kappa)
            # From rates to ZCB prices
            df_paths = np.exp(A - B * df_paths)

            dict_paths[T] = df_paths.copy()

        return dict_paths

    def blackscholes_projection(self, model_params, T, N, rfr, dt=1/252, S0=100):
        """
        Projette l'indice sous-jacent selon le modèle Black-Scholes avec simulation de Monte Carlo.
        
        :param model_name: Nom du modèle ('Black-Scholes')
        :param model_params: Dictionnaire contenant les paramètres du modèle (e.g., {'sigma': 0.2})
        :param T: Horizon de temps de projection en années
        :param N: Nombre de simulations
        :param rfr: Courbe des taux / Drifts pour projection
        :param dt: Pas de temps (par défaut, 1/252 pour un jour de trading)
        :param S0: Valeur initiale de l'indice (par défaut, 100)
        :return: DataFrame contenant les trajectoires simulées (chaque ligne correspond à une simulation)
        """

        # Extraction of parameters
        sigma = model_params['sigma']

        # Time step calculation
        nb_steps = int(T / dt)
        
        # Initialization of forcast matrix
        paths = np.zeros((N, nb_steps + 1))
        
        # Same initial value for all simulation
        paths[:, 0] = S0

        # Preparation of RFR data
        adj_rfr_list = [0]
        adj_rfr_list.extend([rfr(i + 1) * (i + 1) - rfr(i) * i for i in range(0, T)])
        
        # Simulation de N trajectoires
        for t in range(1, nb_steps + 1):
            # Gaussian sampling
            Z = np.random.standard_normal(N)
            # Moments adjustment
            Z_list = Z.tolist()
            Z = (Z - np.mean(Z_list)) / np.std(Z_list)
            # Antithetic variables - A créer
            # Processus de diffusion géométrique pour chaque simulation à chaque pas de temps
            rfr_temp = (adj_rfr_list[int(np.ceil(t * dt))]) 
            paths[:, t] = paths[:, t-1] * np.exp((rfr_temp - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)            

        # Création d'un DataFrame : chaque ligne est une simulation, chaque colonne est un pas de temps
        df_paths = pd.DataFrame(paths, columns=np.linspace(0, T, nb_steps + 1), index = range(1, N + 1))
        df_paths = df_paths[range(0, T + 1)]
        
        return df_paths

    def derivative_pricing(self, model_params, market_data):
        """
        Calcul du prix d'une option en fonction du modèle sélectionné.
        
        :param model_name: Nom du modèle ('Black-Scholes', 'Heston', etc.)
        :param model_params: Dictionnaire contenant les paramètres du modèle
        :param market_data: DataFrame avec les données de marché (doit contenir les colonnes 'S', 'K', 'r', 'T', 'option_type')
        :return: DataFrame avec les prix modélisés
        """
        match self.name:
            case 'Black-Scholes':
                return self.blackscholes_pricing(model_params, market_data)
            case 'Vasicek':
                return self.vasicek_pricing(model_params, market_data)
            case _:
                raise NotImplementedError(f"Le pricing pour le modèle {self.name} n'est pas encore implémenté.")

    def calibration(self, market_data):
        """
        Calibre les paramètres du modèle en minimisant l'écart entre les prix de marché
        et les prix calculés par le modèle.
        
        :param model_name: Nom du modèle
        :param market_data: DataFrame avec les données de marché (doit contenir 'market_price', 'S', 'K', 'r', 'T', 'option_type')
        :return: Paramètre calibré (ici, la volatilité 'sigma' pour Black-Scholes)
        """
        match self.name:
            case 'Black-Scholes':
                initial_guess = [0.2]
                bounds_temp = [(0, None)]
            case 'Vasicek':
                initial_guess = [0.1, 0.03, 0.02, 0.02]
                bounds_temp = [(0.00001, 1), (-0.1, 0.1), (0.00001, 0.5), (0, 0.1)]
                market_data = market_data.loc[[1, 2, 5, 10, 20, 30]]
                # Temporaire A SUPPRIMER
                '''
                self.parameters['kappa'] = initial_guess[0]
                self.parameters['theta'] = initial_guess[1]
                self.parameters['sigma'] = initial_guess[2]
                # Fin du truc à supprimer!!!
                return self.parameters
                '''
            case _:    
                raise NotImplementedError(f"La calibration n'est implémentée pour le modèle {self.name}.")
        
        # Fonction objectif qui calcule l'écart quadratique entre les prix marché et les prix modèles
        def objective_function(params):        
            model_prices = self.derivative_pricing(params, market_data)
            return np.sum((market_data['market_price'] - model_prices) ** 2)
                
        # Minimisation de l'écart entre les prix observés et modélisés
        result = minimize(objective_function, initial_guess, method='Nelder-Mead', bounds = bounds_temp)
        
        # Mise à jour des paramètres
        match self.name:
            case 'Black-Scholes':
                self.parameters['sigma'] = result.x[0]
            case 'Vasicek':
                self.parameters['kappa'] = result.x[0]
                self.parameters['theta'] = result.x[1]
                self.parameters['sigma'] = result.x[2]
                self.parameters['r0']    = result.x[3]
            case _:
                raise NotImplementedError(f"La calibration n'est implémentée pour le modèle {self.name}.")

        return self.parameters
    

def deflator_calculation(df):
    df_adj = df.cumprod(axis = 1)
    return df_adj / df

def ir_to_ZCB(T, rfr):
    time_idx = range(1, T + 1, 1)
    rfr_list = rfr(time_idx).tolist()
    zc_prices = [np.exp(- rfr_list[i] * (i + 1)) for i in range(len(rfr_list))]
    df = pd.DataFrame(zc_prices, index = time_idx, columns = ['market_price'])
    return df
