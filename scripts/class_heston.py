import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.integrate import quad

class HestonModel:
    """
    Classe pour le modèle de Heston (volatilité stochastique)
    """
    def __init__(self):
        self.parameters = {}
    
    def set_parameters(self, v0, kappa, theta, sigma, rho):
        """
        Définir les paramètres du modèle de Heston
        
        :param v0: Variance initiale
        :param kappa: Vitesse de réversion de la variance
        :param theta: Variance à long terme
        :param sigma: Volatilité de la variance
        :param rho: Corrélation entre le prix et la variance
        """
        self.parameters = {
            'v0': v0,
            'kappa': kappa,
            'theta': theta,
            'sigma': sigma,
            'rho': rho
        }
    
    def pricing(self, model_params, market_data):
        """
        Calcule le prix d'une option avec le modèle de Heston
        Note: Pour simplifier, on utilise une approximation Black-Scholes avec volatilité moyenne
        
        :param model_params: Liste contenant les paramètres [v0, kappa, theta, sigma, rho]
        :param market_data: DataFrame avec les données de marché (colonnes 'S', 'K', 'r', 'T', 'option_type')
        :return: Prix de l'option
        """
        # Paramètres du modèle
        v0 = model_params[0]
        kappa = model_params[1] 
        theta = model_params[2]
        sigma_v = model_params[3]
        rho = model_params[4]
        
        # Données de marché
        S = market_data['S']
        K = market_data['K']
        r = market_data['r']
        T = market_data['T']
        option_type = market_data['option_type']
        
        # Approximation : volatilité moyenne pour le pricing
        # Dans Heston, la variance suit un processus de réversion vers la moyenne
        avg_variance = theta + (v0 - theta) * (1 - np.exp(-kappa * T)) / (kappa * T) if kappa > 0 else v0
        avg_volatility = np.sqrt(avg_variance)
        
        # Calcul des paramètres Black-Scholes avec volatilité moyenne
        d1 = (np.log(S / K) + (r + 0.5 * avg_variance) * T) / (avg_volatility * np.sqrt(T))
        d2 = d1 - avg_volatility * np.sqrt(T)
        
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
        Projette l'indice sous-jacent selon le modèle de Heston avec volatilité stochastique
        
        :param model_params: Dictionnaire contenant les paramètres du modèle
        :param T: Horizon de temps de projection en années
        :param N: Nombre de simulations
        :param rfr: Fonction de la courbe des taux sans risque
        :param dt: Pas de temps (par défaut, 1/252 pour un jour de trading)
        :param S0: Valeur initiale de l'indice (par défaut, 100)
        :return: DataFrame contenant les trajectoires simulées
        """
        # Extraction des paramètres
        v0 = model_params['v0']
        kappa = model_params['kappa']
        theta = model_params['theta']
        sigma_v = model_params['sigma']
        rho = model_params['rho']
        
        # Calcul du nombre d'étapes
        nb_steps = int(T / dt)
        
        # Initialisation des matrices des trajectoires
        price_paths = np.zeros((N, nb_steps + 1))
        variance_paths = np.zeros((N, nb_steps + 1))
        
        # Valeurs initiales
        price_paths[:, 0] = S0
        variance_paths[:, 0] = v0
        
        # Préparation des données de taux sans risque
        adj_rfr_list = [0]
        adj_rfr_list.extend([rfr(i + 1) * (i + 1) - rfr(i) * i for i in range(0, T)])
        
        # Simulation de N trajectoires avec le modèle de Heston
        for t in range(1, nb_steps + 1):
            # Génération de bruits corrélés
            Z1 = np.random.standard_normal(N)
            Z2 = np.random.standard_normal(N)
            
            # Ajustement des moments
            Z1 = (Z1 - np.mean(Z1)) / np.std(Z1)
            Z2 = (Z2 - np.mean(Z2)) / np.std(Z2)
            
            # Création de la corrélation
            W1 = Z1
            W2 = rho * Z1 + np.sqrt(1 - rho**2) * Z2
            
            # Récupération des valeurs précédentes
            S_prev = price_paths[:, t-1]
            v_prev = np.maximum(variance_paths[:, t-1], 0)  # Assurer que la variance reste positive
            
            # Taux sans risque pour ce pas de temps
            rfr_temp = adj_rfr_list[int(np.ceil(t * dt))]
            
            # Mise à jour de la variance (processus CIR)
            variance_paths[:, t] = np.maximum(
                v_prev + kappa * (theta - v_prev) * dt + sigma_v * np.sqrt(v_prev * dt) * W2,
                0.0001  # Plancher pour éviter variance négative
            )
            
            # Mise à jour du prix (processus géométrique brownien avec volatilité stochastique)
            price_paths[:, t] = S_prev * np.exp(
                (rfr_temp - 0.5 * v_prev) * dt + np.sqrt(v_prev * dt) * W1
            )
        
        # Création du DataFrame final
        df_paths = pd.DataFrame(
            price_paths, 
            columns=np.linspace(0, T, nb_steps + 1), 
            index=range(1, N + 1)
        )
        df_paths = df_paths[range(0, T + 1)]
        
        return df_paths