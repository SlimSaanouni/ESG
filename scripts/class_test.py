import pandas as pd
from scipy import stats
from numpy import sqrt, exp, log

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
                rfr_temp        = rfr(range(0, T)).tolist()
                rfr_price_temp  = [100 * exp(rfr_temp[t] * t) for t in range(0, T)]
                rfr_price_temp  = pd.Series(data = rfr_price_temp,
                                            index = df.columns)
                # DF init
                martingality_data = pd.DataFrame()
                # First definition
                martingality_data["Expected"] = rfr_price_temp
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
                return martingality_data
            
            case 'price':
                # Deflator test
                df_def  = df["Deflator"]
                N       = df_def.shape[0]
                T       = df_def.shape[1]
                z_score = stats.norm.ppf(1 - alpha / 2)

                # RFR management
                rfr_temp        = rfr
                rfr_temp[0]     = 0
                rfr_price_temp  = [exp(rfr_temp[t] * t) for t in range(0, T)]
                rfr_price_temp  = pd.Series(data = rfr_price_temp,
                                            index = df_def.columns)
                rfr_temp        = pd.Series(data = rfr_temp,
                                            index = df_def.columns)
                
                # Dictionnary init
                martingality_data = {}

                '''Deflator test'''
                # DF init
                martingality_data_temp = pd.DataFrame()
                # First definition
                martingality_data_temp["Expected"] = rfr_price_temp
                martingality_data_temp["Results"]  = df_def.mean() * martingality_data_temp["Expected"]
                martingality_data_temp["Std"]      = df_def.std() * martingality_data_temp["Expected"]
                # Second definition
                martingality_data_temp["Expected"] = 1
                martingality_data_temp["Lower Confidence Interval"] = (martingality_data_temp["Results"]
                                                                      - z_score * martingality_data_temp["Std"] / sqrt(N))
                martingality_data_temp["Upper Confidence Interval"] = (martingality_data_temp["Results"]
                                                                      + z_score * martingality_data_temp["Std"] / sqrt(N))
                martingality_data_temp["Test"]  = ((martingality_data_temp["Expected"] <= martingality_data_temp["Upper Confidence Interval"]) &
                                                   (martingality_data_temp["Expected"] >= martingality_data_temp["Lower Confidence Interval"]))            

                # Faire un test sur la valeur des Zero Coupons from 1 to T
                martingality_data["Deflator"] = martingality_data_temp

                '''Zero-coupon price test'''
                #Mettre le taux année 1
                # Initialization
                martingality_data_temp  = pd.DataFrame(columns = ["Expected", "Results", "Std"],
                                                        index = range(1, T))
                
                # First maturity
                martingality_data_temp.loc[1, "Expected"]  = rfr_temp.loc[1]
                martingality_data_temp.loc[1, "Results"]   = rfr_temp.loc[1]
                martingality_data_temp.loc[1, "Std"]       = 0

                # Other maturities
                for maturity in range(2, T):
                    zc_data_temp = zero_coupon_calculation(df, df_def, maturity)
                    ir_data_temp = - log(zc_data_temp) / maturity
                    martingality_data_temp.loc[maturity, "Expected"] = rfr_temp.loc[maturity]
                    martingality_data_temp.loc[maturity, "Results"] = ir_data_temp.values.mean()
                    martingality_data_temp.loc[maturity, "Std"]     = ir_data_temp.values.std()
                
                martingality_data_temp["Lower Confidence Interval"] = (martingality_data_temp["Results"]
                                                                      - z_score * martingality_data_temp["Std"] / sqrt(N))
                martingality_data_temp["Upper Confidence Interval"] = (martingality_data_temp["Results"]
                                                                      + z_score * martingality_data_temp["Std"] / sqrt(N))
                martingality_data_temp["Test"]  = ((martingality_data_temp["Expected"] <= martingality_data_temp["Upper Confidence Interval"]) &
                                                   (martingality_data_temp["Expected"] >= martingality_data_temp["Lower Confidence Interval"]))            
                
                # Feeding of the output
                martingality_data["ZC_Price"]   = martingality_data_temp
                
                return martingality_data

            case _:

                raise NotImplementedError(f"Le testing pour le type {self.type} n'est pas encore implémenté.")
            

def deflator_calculation(dict_df):
    deflator_df = dict_df[1]
    adj_df = deflator_df.cumprod(axis = 1)
    return adj_df / deflator_df

def zero_coupon_calculation(dict_df, deflator_df, T : int):
    # mettre une exception sur la valeur de T
    zc_df = pd.DataFrame()
    for idx_t in range(1, T):
        idx_T           = T - idx_t
        deflator_temp   = deflator_df[idx_t]
        fwd_zc_temp     = dict_df[idx_T][idx_t]
        zc_temp         = deflator_temp * fwd_zc_temp
        zc_df[idx_t]    = zc_temp
    return zc_df

