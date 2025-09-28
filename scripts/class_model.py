import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from scripts.vasicek_class import VasicekModel

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
        
        # Initialisation du modèle Vasicek si nécessaire
        if self.name == 'Vasicek':
            self.vasicek_model = VasicekModel()

    def vasicek_spot_curve(self, T):
        """
        Délègue le calcul de la courbe spot au modèle Vasicek
        """
        if self.name != 'Vasicek':
            raise ValueError("Cette méthode n'est disponible que pour le modèle Vasicek")
        
        # Mise à jour des paramètres dans le modèle Vasicek
        self.vasicek_model.set_parameters(
            self.parameters['kappa'],
            self.parameters['theta'],
            self.parameters['sigma'],
            self.parameters['r0']
        )
        
        return self.vasicek_model.spot_curve(T)

    def vasicek_pricing(self, model_params, market_data):
        """
        Délègue le pricing au modèle Vasicek
        """
        if self.name != 'Vasicek':
            raise ValueError("Cette méthode n'est disponible que pour le modèle Vasicek")
        
        vasicek_temp = VasicekModel()
        return vasicek_temp.pricing(model_params, market_data)
    
    def vasicek_projection(self, model_params, Tmax, N, spot_rates, dt=1/12):
        """
        Délègue la projection au modèle Vasicek
        """
        if self.name != 'Vasicek':
            raise ValueError("Cette méthode n'est disponible que pour le modèle Vasicek")
        
        vasicek_temp = VasicekModel()
        return vasicek_temp.projection(model_params, Tmax, N, spot_rates, dt)
    
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
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        else:
            raise ValueError("Le type d'option doit être 'call' ou 'put', et unique.")

        return price

    def blackscholes_projection(self, model_params, T, N, rfr, dt=1/252, S0=100):
        """
        Projette l'indice sous-jacent selon le modèle Black-Scholes avec simulation de Monte Carlo.
        
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
        
        :param market_data: DataFrame avec les données de marché (doit contenir 'market_price', 'S', 'K', 'r', 'T', 'option_type')
        :return: Paramètres calibrés
        """
        match self.name:
            case 'Black-Scholes':
                initial_guess = [0.2]
                bounds_temp = [(0, None)]
            case 'Vasicek':
                initial_guess = [0.1, 0.03, 0.02, 0.02]
                bounds_temp = [(0.00001, 1), (-0.1, 0.1), (0.00001, 0.5), (0, 0.1)]
                market_data = market_data.loc[[1, 2, 3, 4, 5, 10, 40]]
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
    

def ir_to_ZCB(T, rfr):
    time_idx = range(1, T + 1, 1)
    rfr_list = rfr(time_idx).tolist()
    zcb_prices = [np.exp(- rfr_list[i] * (i + 1)) for i in range(len(rfr_list))]
    df = pd.DataFrame(zcb_prices, index = time_idx, columns = ['market_price'])
    return df

def ZCB_to_ir(zcb):
    time_idx = zcb.index
    ir_list = [- np.log(zcb.loc[t]) / t for t in time_idx]
    df = pd.Series(ir_list, index = time_idx, columns = ['ir'])
    return df

def deflator_calculation(dict_df):
    deflator_df = dict_df[1]
    adj_df = deflator_df.cumprod(axis = 1)
    return adj_df / deflator_df