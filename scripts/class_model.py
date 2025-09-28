import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scripts.class_vasicek import VasicekModel
from scripts.class_blackscholes import BlackScholesModel
from scripts.class_dupire import DupireModel

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
        
        # Initialisation du modèle approprié selon le nom
        if self.name == 'Vasicek':
            self.vasicek_model = VasicekModel()
        elif self.name == 'Black-Scholes':
            self.blackscholes_model = BlackScholesModel()
        elif self.name == 'Dupire':
            self.dupire_model = DupireModel()

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
        Délègue le pricing au modèle Black-Scholes
        """
        if self.name != 'Black-Scholes':
            raise ValueError("Cette méthode n'est disponible que pour le modèle Black-Scholes")
        
        blackscholes_temp = BlackScholesModel()
        return blackscholes_temp.pricing(model_params, market_data)
    
    def blackscholes_projection(self, model_params, T, N, rfr, dt=1/252, S0=100):
        """
        Délègue la projection au modèle Black-Scholes
        """
        if self.name != 'Black-Scholes':
            raise ValueError("Cette méthode n'est disponible que pour le modèle Black-Scholes")
    
        blackscholes_temp = BlackScholesModel()
        return blackscholes_temp.projection(model_params, T, N, rfr, dt, S0)
        
    def dupire_pricing(self, model_params, market_data):
        """
        Délègue le pricing au modèle de Dupire
        """
        if self.name != 'Dupire':
            raise ValueError("Cette méthode n'est disponible que pour le modèle de Dupire")
        
        dupire_temp = DupireModel()
        return dupire_temp.pricing(model_params, market_data)
    
    def dupire_projection(self, model_params, T, N, rfr, dt=1/252, S0=100):
        """
        Délègue la projection au modèle de Dupire
        """
        if self.name != 'Dupire':
            raise ValueError("Cette méthode n'est disponible que pour le modèle de Dupire")
        
        dupire_temp = DupireModel()
        return dupire_temp.projection(model_params, T, N, rfr, dt, S0)

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
            case 'Dupire':
                return self.dupire_pricing(model_params, market_data)
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
            case 'Dupire':
                initial_guess = [0.2, 0.1]  # [sigma, vol_of_vol]
                bounds_temp = [(0, None), (0, 1)]
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
            case 'Dupire':
                self.parameters['sigma'] = result.x[0]
                self.parameters['vol_of_vol'] = result.x[1]
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