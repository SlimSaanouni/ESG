import pandas as pd
from scipy import stats
from numpy import sqrt, exp

class Martingality_test:
    def __init__(self, type):
        self.type = type
        self.validity = "TBD"

    def martingality_calcs(self, df, rfr, alpha = 0.05):

        match self.type:
            case 'index':
                N = df.shape[0]
                T = df.shape[1]
                z_score = stats.norm.ppf(1 - alpha / 2)
                # RFR management
                rfr_temp = rfr(range(0, T)).tolist()
                rfr_temp = [100 * exp(rfr_temp[t] * t) for t in range(0, T)]
                rfr_temp = pd.DataFrame(data = rfr_temp,
                                        index = df.columns)
                # DF init
                martingality_data = pd.DataFrame()
                # First definition
                martingality_data["Expected"] = rfr_temp
                martingality_data["Results"]  = df.mean() / martingality_data["Expected"]
                martingality_data["Std"]      = df.std() / martingality_data["Expected"]
                # Second definition
                martingality_data["Expected"] = 1
                martingality_data["Lower Confidence Interval"] = (martingality_data["Results"]
                                                                - z_score * martingality_data["Std"] / sqrt(N))
                martingality_data["Upper Confidence Interval"] = (martingality_data["Results"]
                                                                + z_score * martingality_data["Std"] / sqrt(N))
                martingality_data["Test"]  = ((martingality_data["Expected"] <= martingality_data["Upper Confidence Interval"]) &
                                              (martingality_data["Expected"] >= martingality_data["Lower Confidence Interval"]))
            case _:
                # Faire un test de déflateur sur l'année 1

            
                # Faire un test sur la valeur des Zero Coupons from 1 to T

                raise NotImplementedError(f"Le testing pour le type {self.type} n'est pas encore implémenté.")
            
        return martingality_data

