import numpy as np
import pandas as pd
from scipy.stats import norm


class DupireModel:
    """
    Classe pour le modèle de Dupire (volatilité locale)
    """
    def __init__(self):
        self.parameters = {}
    
    def set_parameters(self, sigma_function):
        """
        Définir les paramètres du modèle de Dupire
        
        :param sigma_function: Fonction de volatilité locale sigma(S, t)
        """
        self.parameters = {
            'sigma_function': sigma_function
        }
    
    def pricing(self, model_params, market_data):
        """
        Calcule le prix d'une option avec le modèle de Dupire
        Note: Pour simplifier, on utilise une volatilité constante pour le pricing
        
        :param model_params: Liste contenant la volatilité [sigma]
        :param market_data: DataFrame avec les données de marché (colonnes 'S', 'K', 'r', 'T', 'option_type')
        :return: Prix de l'option
        """
        # Paramètre du modèle (simplifié à une volatilité constante pour le pricing)
        sigma = model_params[0]
        
        # Données de marché
        S = market_data['S']
        K = market_data['K']
        r = market_data['r']
        T = market_data['T']
        option_type = market_data['option_type']
        
        # Calcul des paramètres Black-Scholes (approximation)
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
    
    def _local_volatility(self, S, t, base_sigma=0.2, vol_of_vol=0.1):
        """
        Fonction de volatilité locale simple pour le modèle de Dupire
        sigma(S,t) = base_sigma * (1 + vol_of_vol * (S/100 - 1))
        
        :param S: Prix du sous-jacent
        :param t: Temps
        :param base_sigma: Volatilité de base
        :param vol_of_vol: Volatilité de la volatilité
        :return: Volatilité locale
        """
        # Modèle simple : volatilité qui dépend du niveau de prix
        return base_sigma * (1 + vol_of_vol * (S / 100 - 1))
    
    def projection(self, model_params, T, N, rfr, dt=1/252, S0=100):
        """
        Projette l'indice sous-jacent selon le modèle de Dupire avec volatilité locale
        
        :param model_params: Dictionnaire contenant les paramètres ({'sigma': volatilité de base})
        :param T: Horizon de temps de projection en années
        :param N: Nombre de simulations
        :param rfr: Fonction de la courbe des taux sans risque
        :param dt: Pas de temps (par défaut, 1/252 pour un jour de trading)
        :param S0: Valeur initiale de l'indice (par défaut, 100)
        :return: DataFrame contenant les trajectoires simulées
        """
        # Extraction des paramètres
        base_sigma = model_params['sigma']
        vol_of_vol = model_params.get('vol_of_vol', 0.1)  # Paramètre optionnel
        
        # Calcul du nombre d'étapes
        nb_steps = int(T / dt)
        
        # Initialisation de la matrice des trajectoires
        paths = np.zeros((N, nb_steps + 1))
        
        # Même valeur initiale pour toutes les simulations
        paths[:, 0] = S0
        
        # Préparation des données de taux sans risque
        adj_rfr_list = [0]
        adj_rfr_list.extend([rfr(i + 1) * (i + 1) - rfr(i) * i for i in range(0, T)])
        
        # Simulation de N trajectoires avec volatilité locale
        for t in range(1, nb_steps + 1):
            # Échantillonnage gaussien
            Z = np.random.standard_normal(N)
            # Ajustement des moments
            Z_list = Z.tolist()
            Z = (Z - np.mean(Z_list)) / np.std(Z_list)
            
            # Temps actuel
            current_time = t * dt
            
            # Calcul de la volatilité locale pour chaque trajectoire
            current_prices = paths[:, t-1]
            local_vols = np.array([
                self._local_volatility(S, current_time, base_sigma, vol_of_vol) 
                for S in current_prices
            ])
            
            # Processus de diffusion géométrique avec volatilité locale
            rfr_temp = adj_rfr_list[int(np.ceil(t * dt))]
            paths[:, t] = paths[:, t-1] * np.exp(
                (rfr_temp - 0.5 * local_vols ** 2) * dt + local_vols * np.sqrt(dt) * Z
            )
        
        # Création du DataFrame final
        df_paths = pd.DataFrame(
            paths, 
            columns=np.linspace(0, T, nb_steps + 1), 
            index=range(1, N + 1)
        )
        df_paths = df_paths[range(0, T + 1)]
        
        return df_paths