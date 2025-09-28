import pandas as pd
from scipy import stats
from numpy import sqrt, exp, log

class Martingality_test:
    def __init__(self, type):
        self.type = type
        self.validity = "TBD"

    def martingality_calcs(self, df, rfr, alpha=0.05):
        """
        Calcule les tests de martingalité selon le type de modèle
        
        :param df: DataFrame des projections ou dictionnaire pour les modèles de prix
        :param rfr: Fonction ou série des taux sans risque
        :param alpha: Niveau de confiance pour les tests
        :return: Résultats des tests de martingalité
        """
        match self.type:
            case 'index':
                return self._test_index_martingality(df, rfr, alpha)
            case 'price':
                return self._test_price_martingality(df, rfr, alpha)
            case _:
                raise NotImplementedError(f"Le testing pour le type {self.type} n'est pas encore implémenté.")

    def _test_index_martingality(self, df, rfr, alpha):
        """
        Test de martingalité pour les modèles d'indices (Black-Scholes, Dupire, Heston)
        
        :param df: DataFrame des trajectoires simulées
        :param rfr: Fonction des taux sans risque
        :param alpha: Niveau de confiance
        :return: DataFrame des résultats de test
        """
        N = df.shape[0]
        T = df.shape[1]
        z_score = stats.norm.ppf(1 - alpha / 2)
        
        # Gestion des taux sans risque
        rfr_temp = rfr(range(0, T)).tolist()
        rfr_price_temp = [100 * exp(rfr_temp[t] * t) for t in range(0, T)]
        rfr_price_temp = pd.Series(data=rfr_price_temp, index=df.columns)
        
        # Initialisation du DataFrame des résultats
        martingality_data = pd.DataFrame()
        
        # Calculs des statistiques
        martingality_data["Expected"] = rfr_price_temp
        martingality_data["Results"] = df.mean() / martingality_data["Expected"]
        martingality_data["Std"] = df.std() / martingality_data["Expected"]
        
        # Normalisation pour le test (espérance = 1)
        martingality_data["Expected"] = 1
        martingality_data["Lower Confidence Interval"] = (
            martingality_data["Results"] - z_score * martingality_data["Std"] / sqrt(N)
        )
        martingality_data["Upper Confidence Interval"] = (
            martingality_data["Results"] + z_score * martingality_data["Std"] / sqrt(N)
        )
        
        # Test de validation
        martingality_data["Test"] = (
            (martingality_data["Expected"] <= martingality_data["Upper Confidence Interval"]) &
            (martingality_data["Expected"] >= martingality_data["Lower Confidence Interval"])
        )
        
        return martingality_data

    def _test_price_martingality(self, df, rfr, alpha):
        """
        Test de martingalité pour les modèles de prix (Vasicek, G2++)
        
        :param df: Dictionnaire contenant les trajectoires par maturité et le déflateur
        :param rfr: Série des taux sans risque par maturité
        :param alpha: Niveau de confiance
        :return: Dictionnaire des résultats de test
        """
        # Test du déflateur
        df_def = df["Deflator"]
        N = df_def.shape[0]
        T = df_def.shape[1]
        z_score = stats.norm.ppf(1 - alpha / 2)

        # Gestion des taux sans risque
        rfr_temp = rfr.copy()
        rfr_temp[0] = 0
        rfr_price_temp = [exp(rfr_temp[t] * t) for t in range(0, T)]
        rfr_price_temp = pd.Series(data=rfr_price_temp, index=df_def.columns)
        rfr_temp = pd.Series(data=rfr_temp, index=df_def.columns)
        
        # Dictionnaire des résultats
        martingality_data = {}

        # Test du déflateur
        martingality_data["Deflator"] = self._calculate_deflator_test(
            df_def, rfr_price_temp, z_score, N
        )

        # Test des prix zéro-coupon
        martingality_data["ZC_Price"] = self._calculate_zc_price_test(
            df, df_def, rfr_temp, z_score, N, T
        )
        
        return martingality_data

    def _calculate_deflator_test(self, df_def, rfr_price_temp, z_score, N):
        """
        Calcule le test de martingalité pour le déflateur
        """
        martingality_data_temp = pd.DataFrame()
        
        # Calculs des statistiques
        martingality_data_temp["Expected"] = rfr_price_temp
        martingality_data_temp["Results"] = df_def.mean() * martingality_data_temp["Expected"]
        martingality_data_temp["Std"] = df_def.std() * martingality_data_temp["Expected"]
        
        # Normalisation pour le test (espérance = 1)
        martingality_data_temp["Expected"] = 1
        martingality_data_temp["Lower Confidence Interval"] = (
            martingality_data_temp["Results"] - z_score * martingality_data_temp["Std"] / sqrt(N)
        )
        martingality_data_temp["Upper Confidence Interval"] = (
            martingality_data_temp["Results"] + z_score * martingality_data_temp["Std"] / sqrt(N)
        )
        
        # Test de validation
        martingality_data_temp["Test"] = (
            (martingality_data_temp["Expected"] <= martingality_data_temp["Upper Confidence Interval"]) &
            (martingality_data_temp["Expected"] >= martingality_data_temp["Lower Confidence Interval"])
        )
        
        return martingality_data_temp

    def _calculate_zc_price_test(self, df, df_def, rfr_temp, z_score, N, T):
        """
        Calcule le test de martingalité pour les prix zéro-coupon
        """
        # Initialisation
        martingality_data_temp = pd.DataFrame(
            columns=["Expected", "Results", "Std"],
            index=range(1, T)
        )
        
        # Première maturité (pas de test, valeur fixe)
        martingality_data_temp.loc[1, "Expected"] = rfr_temp.loc[1]
        martingality_data_temp.loc[1, "Results"] = rfr_temp.loc[1]
        martingality_data_temp.loc[1, "Std"] = 0

        # Autres maturités
        for maturity in range(2, T):
            zc_data_temp = zero_coupon_calculation(df, df_def, maturity)
            ir_data_temp = -log(zc_data_temp) / maturity
            
            martingality_data_temp.loc[maturity, "Expected"] = rfr_temp.loc[maturity]
            martingality_data_temp.loc[maturity, "Results"] = ir_data_temp.values.mean()
            martingality_data_temp.loc[maturity, "Std"] = ir_data_temp.values.std()
        
        # Intervalles de confiance
        martingality_data_temp["Lower Confidence Interval"] = (
            martingality_data_temp["Results"] - z_score * martingality_data_temp["Std"] / sqrt(N)
        )
        martingality_data_temp["Upper Confidence Interval"] = (
            martingality_data_temp["Results"] + z_score * martingality_data_temp["Std"] / sqrt(N)
        )
        
        # Test de validation
        martingality_data_temp["Test"] = (
            (martingality_data_temp["Expected"] <= martingality_data_temp["Upper Confidence Interval"]) &
            (martingality_data_temp["Expected"] >= martingality_data_temp["Lower Confidence Interval"])
        )
        
        return martingality_data_temp


def deflator_calculation(dict_df):
    """
    Calcule le déflateur à partir des prix zéro-coupon de maturité 1 an
    
    :param dict_df: Dictionnaire contenant les DataFrames de prix ZC par maturité
    :return: DataFrame du déflateur
    """
    deflator_df = dict_df[1]
    adj_df = deflator_df.cumprod(axis=1)
    return adj_df / deflator_df


def zero_coupon_calculation(dict_df, deflator_df, T: int):
    """
    Calcule les prix zéro-coupon pour une maturité T donnée
    
    :param dict_df: Dictionnaire des prix ZC par maturité
    :param deflator_df: DataFrame du déflateur
    :param T: Maturité cible
    :return: DataFrame des prix ZC pour la maturité T
    """
    if T < 2:
        raise ValueError("La maturité T doit être supérieure ou égale à 2")
    
    zc_df = pd.DataFrame()
    
    for idx_t in range(1, T):
        idx_T = T - idx_t
        
        if idx_T not in dict_df:
            continue
            
        deflator_temp = deflator_df[idx_t]
        fwd_zc_temp = dict_df[idx_T][idx_t]
        zc_temp = deflator_temp * fwd_zc_temp
        zc_df[idx_t] = zc_temp
    
    return zc_df