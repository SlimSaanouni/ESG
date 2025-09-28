import numpy as np
import pandas as pd

class VasicekModel:
    """
    Classe pour le modèle de Vasicek
    """
    def __init__(self):
        self.parameters = {}
    
    def set_parameters(self, kappa, theta, sigma, r0):
        """
        Définir les paramètres du modèle de Vasicek
        
        :param kappa: Vitesse de réversion
        :param theta: Niveau de réversion à long terme
        :param sigma: Volatilité
        :param r0: Taux initial
        """
        self.parameters = {
            'kappa': kappa,
            'theta': theta,
            'sigma': sigma,
            'r0': r0
        }
    
    def spot_curve(self, T):
        """
        Calcule la courbe des taux spot selon le modèle de Vasicek
        
        :param T: Horizon maximum
        :return: Série des taux d'intérêt par maturité
        """
        if not self.parameters:
            raise ValueError("Les paramètres du modèle doivent être définis avant le calcul")
        
        kappa = self.parameters['kappa']
        theta = self.parameters['theta']
        sigma = self.parameters['sigma']
        r0 = self.parameters['r0']
        
        maturities = range(1, T + 1, 1)
        ir = []
        
        for maturity in maturities:
            B = (1 - np.exp(-kappa * maturity)) / kappa
            A = ((theta - (sigma ** 2) / (2 * kappa ** 2)) * (B - maturity) - 
                 (sigma ** 2 * B ** 2) / (4 * kappa))
            zero_temp = np.exp(A - B * r0)
            ir_temp = - np.log(zero_temp) / maturity
            ir.append(ir_temp)
        
        return pd.Series(ir, index=maturities)
    
    def pricing(self, model_params, market_data):
        """
        Calcule les prix des obligations zéro-coupon sous le modèle de Vasicek
        
        :param model_params: Liste des paramètres [kappa, theta, sigma, r0]
        :param market_data: DataFrame contenant les maturités en index
        :return: Série avec les prix des obligations zéro-coupon
        """
        kappa = model_params[0]
        theta = model_params[1]
        sigma = model_params[2]
        r0 = model_params[3]
        
        maturities = market_data.index
        list_zc_prices = []
        
        for T in maturities:
            B = (1 - np.exp(-kappa * T)) / kappa
            A = ((theta - (sigma**2) / (2 * kappa**2)) * (B - T) - 
                 (sigma**2) * (B**2) / (4 * kappa))
            zc_price = np.exp(A - B * r0)
            list_zc_prices.append(zc_price)
        
        return pd.Series(data=list_zc_prices, index=maturities)
    
    def projection(self, model_params, Tmax, N, spot_rates, dt=1/12):
        """
        Projette les taux d'intérêt sous le modèle de Vasicek
        
        :param model_params: Dictionnaire des paramètres
        :param Tmax: Horizon de projection
        :param N: Nombre de simulations
        :param spot_rates: Fonction des taux spot
        :param dt: Pas de temps
        :return: Dictionnaire contenant les trajectoires pour chaque maturité
        """
        kappa = model_params['kappa']
        theta = model_params['theta']
        sigma = model_params['sigma']
        r0 = model_params['r0']
        
        # Nombre d'étapes
        nb_steps = int(Tmax / dt)
        
        # Préparation des taux spot
        time_idx = range(1, Tmax + 1, 1)
        rfr_spot = pd.DataFrame(
            data=spot_rates(time_idx), 
            index=time_idx, 
            columns=["IR"]
        )
        maturities = rfr_spot.index
        
        # Initialisation des trajectoires
        paths = np.zeros((N, nb_steps + 1))
        paths[:, 0] = r0
        
        # Simulation des trajectoires
        for t in range(1, nb_steps + 1):
            # Ajustement des moments
            Z = np.random.standard_normal(N)
            Z_list = Z.tolist()
            Z = (Z - np.mean(Z_list)) / np.std(Z_list)
            
            # Projection des taux
            paths[:, t] = (paths[:, t - 1] + 
                          kappa * (theta - paths[:, t - 1]) * dt + 
                          sigma * np.sqrt(dt) * Z)
        
        # Création du DataFrame des trajectoires
        df_paths = pd.DataFrame(
            paths, 
            columns=np.linspace(0, Tmax, nb_steps + 1), 
            index=range(1, N + 1)
        )
        df_paths = df_paths[range(0, Tmax + 1)]
        
        # Dictionnaire des trajectoires par maturité
        dict_paths = {}
        
        # Boucle sur les maturités
        for T in maturities:
            # Paramètres de Vasicek
            B = (1 - np.exp(-kappa * T)) / kappa
            A = ((theta - (sigma**2) / (2 * kappa**2)) * (B - T) - 
                 (sigma**2) * (B**2) / (4 * kappa))
            
            # Conversion des taux en prix ZC
            df_path_temp = np.exp(A - B * df_paths)
            dict_paths[T] = df_path_temp.copy()
        
        # Calcul du déflateur
        dict_paths["Deflator"] = self._deflator_calculation(dict_paths)
        
        return dict_paths
    
    def _deflator_calculation(self, dict_df):
        """
        Calcule le déflateur à partir des prix ZC
        
        :param dict_df: Dictionnaire des DataFrames de prix ZC
        :return: DataFrame du déflateur
        """
        deflator_df = dict_df[1]
        adj_df = deflator_df.cumprod(axis=1)
        return adj_df / deflator_df