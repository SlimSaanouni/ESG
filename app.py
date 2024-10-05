import streamlit as st

from scripts.class_model import Model, deflator_calculation, MODEL_TYPE
from scripts.class_test import Martingality_test
from scripts.class_template import RiskFreeRates, InputsTemplate, TestsResultsTemplates, class_list

N_COL_MAX   = 3
nb_class    = len(class_list)

T = 50 # Projectionhorizon
N = 1000 # Number of simulations

'''
# Financial models calibration

## Risk free rates spot
'''
# Structuration of RFR
rfr = RiskFreeRates("2024.09")
rfr_dict = rfr.render()

if not rfr_dict == {}:
    '''
    ## Choice of models
    '''
    # Implémenter plus tard les autres modes (Base, IR_Up, IR_Down)
    rfr_temp = rfr_dict["Base"]

    # Inputs
    dict_simulations = {}
    inputs_templates = [InputsTemplate(class_list[i]) for i in range(nb_class)]
    for i in range(0, nb_class, N_COL_MAX):
        cols = st.columns(N_COL_MAX)
        for j, template in enumerate(inputs_templates[i:i+N_COL_MAX]):
            with cols[j]:
                output_temp = template.render(T, rfr_temp)
                dict_simulations[template.asset_class] = output_temp
    '''
    ## Testing

    ### Martingality testing
    '''
    # Tests
    tests_results_templates = [TestsResultsTemplates(class_list[i], dict_simulations[class_list[i]]['model_name']) for i in range(nb_class)]
    tabs_result = st.tabs(class_list)

    for i, template in enumerate(tests_results_templates):
        with tabs_result[i]:
            # Reading of the market data that will be used to calibrate the chosen model
            df_temp = dict_simulations[template.asset_class]['calibration_df']
            if not df_temp.empty:
                # Name of the asset model for the asset class
                model_name_temp = dict_simulations[template.asset_class]['model_name']
                type_temp = MODEL_TYPE[model_name_temp]
                # Calibration of the model
                model_temp = Model(name = model_name_temp)
                calibrated_params = model_temp.calibration(df_temp)
                st.write(calibrated_params)
                # Projection using the model
                match model_name_temp:
                    case 'Black-Scholes':
                        proj_df_temp = model_temp.blackscholes_projection(calibrated_params, T, N, rfr_temp)
                        # Martingality test
                        mart_test_temp = Martingality_test(type = type_temp)
                        martingality_df_temp = mart_test_temp.martingality_calcs(proj_df_temp, rfr_temp, 0.05)
                        # Creation of the streamlit features
                        template.render_equity(proj_df_temp, martingality_df_temp)
                    case 'Vasicek':
                        proj_dict_temp = model_temp.vasicek_projection(calibrated_params, T, N, rfr_temp)
                        # Extraction of maturity = 1 for deflator calculation
                        test_def_temp = proj_dict_temp[1] # To be removed
                        st.write(test_def_temp)
                        deflator = deflator_calculation(test_def_temp)
                        st.write(deflator)
                        # Martingality test
                        mart_test_temp = Martingality_test(type = type_temp)
                        martingality_df_temp = mart_test_temp.martingality_calcs(proj_dict_temp, rfr_temp, 0.05)
                        # Creation of the streamlit features
                        template.render_interest_rates(proj_df_temp, martingality_df_temp)
            else:
                st.write("No data uploaded for " + class_list[i] + ".")