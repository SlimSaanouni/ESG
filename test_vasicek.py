import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class VasicekModel:
    def __init__(self):
        pass

    def zero_coupon_price_vasicek(self, kappa, theta, sigma, r0, maturities):
        """
        Calcule les prix des obligations zéro-coupon sous le modèle Vasicek.
        :param kappa: Vitesse de réversion
        :param theta: Niveau de réversion à long terme
        :param sigma: Volatilité
        :param r0: Taux court initial (au temps 0)
        :param maturities: Maturités (en années) pour lesquelles on veut calculer les prix
        :return: DataFrame avec les prix des obligations zéro-coupon pour chaque maturité
        """
        prices = []
        for T in maturities:
            B = (1 - np.exp(-kappa * T)) / kappa
            A = (theta - (sigma**2) / (2 * kappa**2)) * (B - T) - (sigma**2 * B**2) / (4 * kappa)
            zero_coupon_price = np.exp(A - B * r0)
            prices.append(zero_coupon_price)

        return pd.Series(prices, index=maturities)

    def spot_rate_from_price(self, zero_coupon_prices):
        """
        Calcule les taux spot à partir des prix des obligations zéro-coupon.
        :param zero_coupon_prices: Prix des obligations zéro-coupon
        :return: Taux spot pour chaque maturité
        """
        maturities = zero_coupon_prices.index
        spot_rates = -np.log(zero_coupon_prices) / maturities
        return pd.Series(spot_rates, index=maturities)

    def calibration_error(self, params, observed_prices, maturities):
        """
        Fonction d'erreur pour la calibration.
        :param params: [kappa, theta, sigma, r0]
        :param observed_prices: Prix observés des obligations zéro-coupon
        :param maturities: Maturités des obligations zéro-coupon
        :return: Erreur quadratique moyenne
        """
        kappa, theta, sigma, r0 = params
        model_prices = self.zero_coupon_price_vasicek(kappa, theta, sigma, r0, maturities)
        
        # Diagnostic: Vérifiez les prix observés et modélisés
        print("Paramètres:", params)
        print("Prix observés:", observed_prices.values)
        print("Prix modélisés:", model_prices.values)
        
        # Calcul de l'erreur
        error = np.sum((observed_prices - model_prices) ** 2)
        return error

    def calibrate(self, observed_prices, maturities):
        """
        Calibre les paramètres du modèle Vasicek en minimisant l'erreur entre les prix observés et modélisés.
        :param observed_prices: Prix des obligations zéro-coupon observés sur le marché
        :param maturities: Maturités des obligations zéro-coupon
        :return: Paramètres calibrés (kappa, theta, sigma, r0)
        """
        # Estimations initiales des paramètres (kappa, theta, sigma, r0)
        initial_params = [0.1, 0.05, 0.01, 0.02]  # r0 initial estimé
        bounds = [(0.00001, 5), (-0.1, 0.1), (0.00001, 0.1), (0, 0.1)]  # kappa, theta, sigma, r0: resserré pour guider la calibration
        
        # Minimisation de l'erreur de calibration
        result = minimize(self.calibration_error, initial_params, args=(observed_prices, maturities),
                          method='L-BFGS-B', bounds=bounds)
        
        # Diagnostic: Vérifier si la calibration a convergé
        if result.success:
            print("Calibration réussie !")
        else:
            print("Problème lors de la calibration:", result.message)
        
        return result.x  # Retourne les paramètres calibrés

# Fonction de visualisation des courbes
def plot_curves(observed_spot_rates, model_spot_rates, maturities):
    plt.figure(figsize=(10, 6))
    plt.plot(maturities, observed_spot_rates, label="Courbe des taux observée", marker='o')
    plt.plot(maturities, model_spot_rates, label="Courbe des taux modélisée (Vasicek)", linestyle='--', marker='x')
    plt.title("Courbes des taux observée et modélisée")
    plt.xlabel("Maturité (années)")
    plt.ylabel("Taux spot")
    plt.legend()
    plt.grid(True)
    plt.show()

# Exemple d'utilisation
if __name__ == "__main__":
    # Courbe des taux zéro-coupon observée (fictive)
    maturities = np.array([1, 2, 5, 10, 20, 30])  # en années
    spot_rates = pd.Series([0.033, 0.027, 0.025, 0.026, 0.0266, 0.027], index=maturities)
    
    # Prix zéro-coupon observés (fictifs)
    observed_prices = np.exp(-spot_rates * maturities)  # Transformer les taux spot en prix de zéro-coupon

    # Création du modèle et calibration
    vasicek_model = VasicekModel()
    calibrated_params = vasicek_model.calibrate(observed_prices, maturities)
    
    print("Paramètres calibrés (kappa, theta, sigma, r0) :", calibrated_params)
    
    # Calcul des prix modélisés avec les paramètres calibrés
    kappa, theta, sigma, r0 = calibrated_params
    model_prices = vasicek_model.zero_coupon_price_vasicek(kappa, theta, sigma, r0, maturities)
    
    # Convertir les prix des obligations modélisés en taux spot
    model_spot_rates = vasicek_model.spot_rate_from_price(model_prices)

    # Visualisation des courbes des taux observée et modélisée
    plot_curves(spot_rates, model_spot_rates, maturities)

# [0.033, 0.027, 0.025, 0.026, 0.0266, 0.027]