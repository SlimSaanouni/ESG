import streamlit as st

from scripts.class_model import Model, MODEL_TYPE
from scripts.class_test import Martingality_test
from scripts.class_template import RiskFreeRates, InputsTemplate, TestsResultsTemplates, class_list

N_COL_MAX   = 3
nb_class    = len(class_list)

T = 50 # Projection horizon
N = 1000 # Number of simulations

# Configuration de la page
st.set_page_config(
    page_title="Financial Models Calibration",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"  # Sidebar cachée initialement
)

'''
# Financial models calibration

## Risk free rates spot
'''
# Structuration of RFR
rfr = RiskFreeRates("2024.09")
rfr_dict = rfr.render()

# Vérification si le fichier RFR est uploadé
if rfr_dict == {}:
    # Sidebar vide et cachée
    st.info("📁 Please upload the Risk Free Rates file to continue.")
    st.stop()

# Si RFR est chargé, on configure la sidebar en mode "expanded"
st.markdown(
    """
    <script>
        var sidebar = window.parent.document.querySelector('[data-testid="stSidebar"]');
        if (sidebar) {
            sidebar.style.display = 'block';
        }
    </script>
    """,
    unsafe_allow_html=True
)

'''
## Choice of models
'''
# Implémenter plus tard les autres modes (Base, IR_Up, IR_Down)
rfr_temp = rfr_dict["Base"]

# Inputs dans la sidebar
with st.sidebar:
    st.header("📋 General parameters")
    st.markdown("---")
    T = st.number_input("Projection horizon", min_value=1, max_value=100, value=50, step=1, key="T")
    N = st.number_input("Number of simulations", min_value=100, max_value=10000, value=1000, step=100, key="N")
    st.header("📋 Model Selection")
    st.markdown("---")
    
    dict_simulations = {}
    inputs_templates = [InputsTemplate(class_list[i]) for i in range(nb_class)]
    
    for i, template in enumerate(inputs_templates):
        with st.expander(f"🔧 {template.asset_class}", expanded=True):
            output_temp = template.render(T, rfr_temp)
            dict_simulations[template.asset_class] = output_temp

'''
## Testing
'''
# Tests dans le contenu principal
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
            '''
            ### Calibrated parameters
            '''
            st.write(calibrated_params)
            # Projection using the model
            match model_name_temp:
                case 'Black-Scholes':
                    proj_df_temp = model_temp.blackscholes_projection(calibrated_params, T, N, rfr_temp)
                    # Martingality test
                    mart_test_temp = Martingality_test(type = type_temp)
                    martingality_df_temp = mart_test_temp.martingality_calcs(proj_df_temp, rfr_temp, 0.05)
                    # Display of results
                    template.render_index(proj_df_temp, martingality_df_temp)
                case 'Dupire':
                    proj_df_temp = model_temp.dupire_projection(calibrated_params, T, N, rfr_temp)
                    mart_test_temp = Martingality_test(type = type_temp)
                    martingality_df_temp = mart_test_temp.martingality_calcs(proj_df_temp, rfr_temp, 0.05)
                    template.render_index(proj_df_temp, martingality_df_temp)
                case 'Heston':
                    proj_df_temp = model_temp.heston_projection(calibrated_params, T, N, rfr_temp)
                    # Martingality test
                    mart_test_temp = Martingality_test(type = type_temp)
                    martingality_df_temp = mart_test_temp.martingality_calcs(proj_df_temp, rfr_temp, 0.05)
                    # Display of results
                    template.render_index(proj_df_temp, martingality_df_temp)
                case 'Vasicek':
                    model_spot_curve = model_temp.vasicek_spot_curve(T)
                    template.display_calibrated_ir(rfr_temp, model_spot_curve)
                    proj_dict_temp = model_temp.vasicek_projection(calibrated_params, T, N, rfr_temp)
                    # Martingality test
                    mart_test_temp = Martingality_test(type = type_temp)
                    martingality_dict_temp = mart_test_temp.martingality_calcs(proj_dict_temp, model_spot_curve, 0.05)
                    # Display of results
                    template.render_interest_rates(martingality_dict_temp)
        else:
            st.write("No data uploaded for " + class_list[i] + ".")