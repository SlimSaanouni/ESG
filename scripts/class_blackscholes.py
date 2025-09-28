import numpy as np
import pandas as pd
from scipy.stats import norm

class BlackScholesModel:
    """
    Classe pour le modèle de Black-Scholes
    """
    def __init__(self):
        self.parameters = {}
    
    def set_parameters(self, sigma):
        """
        Définir les paramètres du modèle de Black-Scholes
        
        :param sigma: Volatilité
        """
        self.parameters = {
            'sigma': sigma
        }
    
    def pricing(self, model_params, market_data):
        """
        Calcule le prix d'une option call ou put avec le modèle Black-Scholes
        
        :param model_params: Liste contenant la volatilité [sigma]
        :param market_data: DataFrame avec les données de marché (colonnes 'S', 'K', 'r', 'T', 'option_type')
        :return: Prix de l'option
        """
        # Paramètre du modèle
        sigma = model_params[0]
        
        # Données de marché
        S = market_data['S']
        K = market_data['K']
        r = market_data['r']
        T = market_data['T']
        option_type = market_data['option_type']
        
        # Calcul des paramètres Black-Scholes
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Prix selon le type d'option
        if all(option_type == "call"):
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        elif all(option_type == "put"):
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        else:
            raise ValueError("Le type d'option doit être 'call' ou 'put', et unique.")
        
        return price
    
    def projection(self, model_params, T, N, rfr, dt=1/252, S0=100):
        """
        Projette l'indice sous-jacent selon le modèle Black-Scholes avec simulation de Monte Carlo
        
        :param model_params: Dictionnaire contenant les paramètres ({'sigma': volatilité})
        :param T: Horizon de temps de projection en années
        :param N: Nombre de simulations
        :param rfr: Fonction de la courbe des taux sans risque
        :param dt: Pas de temps (par défaut, 1/252 pour un jour de trading)
        :param S0: Valeur initiale de l'indice (par défaut, 100)
        :return: DataFrame contenant les trajectoires simulées
        """
        # Extraction des paramètres
        sigma = model_params['sigma']
        
        # Calcul du nombre d'étapes
        nb_steps = int(T / dt)
        
        # Initialisation de la matrice des trajectoires
        paths = np.zeros((N, nb_steps + 1))
        
        # Même valeur initiale pour toutes les simulations
        paths[:, 0] = S0
        
        # Préparation des données de taux sans risque
        adj_rfr_list = [0]
        adj_rfr_list.extend([rfr(i + 1) * (i + 1) - rfr(i) * i for i in range(0, T)])
        
        # Simulation de N trajectoires
        for t in range(1, nb_steps + 1):
            # Échantillonnage gaussien
            Z = np.random.standard_normal(N)
            # Ajustement des moments
            Z_list = Z.tolist()
            Z = (Z - np.mean(Z_list)) / np.std(Z_list)
            
            # Processus de diffusion géométrique
            rfr_temp = adj_rfr_list[int(np.ceil(t * dt))]
            paths[:, t] = paths[:, t-1] * np.exp(
                (rfr_temp - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
            )
        
        # Création du DataFrame final
        df_paths = pd.DataFrame(
            paths, 
            columns=np.linspace(0, T, nb_steps + 1), 
            index=range(1, N + 1)
        )
        df_paths = df_paths[range(0, T + 1)]
        
        return df_paths