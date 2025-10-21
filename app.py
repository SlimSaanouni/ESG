import streamlit as st
import pandas as pd

from scripts.class_model import Model, MODEL_TYPE, NB_WEINER
from scripts.class_test import Martingality_test
from scripts.class_template import RiskFreeRates, InputsTemplate, TestsResultsTemplates, class_list
from scripts.dependency_tab import render_dependency_tab
from scripts.simulation_tab import render_simulation_tab
from scripts.test_tab import render_tests_tab
from scripts.tab_export import render_export_tab  # NOUVEAU

N_COL_MAX   = 3
nb_class    = len(class_list)

# Configuration de la page
st.set_page_config(
    page_title="Financial Models Calibration",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
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
    T = st.number_input("Projection horizon (years)", min_value=1, max_value=100, value=50, step=1, key="T")
    N = st.number_input("Number of simulations", min_value=100, max_value=10000, value=1000, step=100, key="N")
    
    st.header("📋 Model Selection")
    st.markdown("---")
    
    dict_simulations = {}
    inputs_templates = [InputsTemplate(class_list[i]) for i in range(nb_class)]
    
    for i, template in enumerate(inputs_templates):
        with st.expander(f"🔧 {template.asset_class}", expanded=True):
            output_temp = template.render(T, rfr_temp)
            dict_simulations[template.asset_class] = output_temp

    st.header("📋 Financial parameters")
    st.markdown("---")
    re_vol_ratio    = st.number_input("Real Estate volatility ratio (%. of Equity volatility)",
                                      min_value=0, max_value=100, value=50, key="re_vol_ratio") / 100
    dividend_rate   = st.number_input("Dividend yield (%)", min_value=0, max_value=100, value=2, key="div_rate") / 100
    rental_rate     = st.number_input("Rental yield (%)", min_value=0, max_value=100, value=2, key="rent_rate") / 100
    


# Vérification de l'état des modèles calibrés
# Pour la dependency structure, on a besoin des 3 classes principales
required_classes = ["Interest rates", "Equity", "Real Estate"]
models_ready = all(
    not dict_simulations[asset_class]['calibration_df'].empty 
    for asset_class in required_classes
)

# Extraction des modèles sélectionnés
selected_models = {
    asset_class: dict_simulations[asset_class]['model_name']
    for asset_class in required_classes
}

# Dictionnaire pour stocker les paramètres calibrés
calibrated_parameters = {}

'''
## Results
'''
# Création des onglets (AJOUT de l'onglet Export)
tab_names = class_list + ["Dependency Structure", "Correlated Simulations", "Martingality Tests", "Export"]
tabs = st.tabs(tab_names)

# Tests dans les onglets de résultats
tests_results_templates = [
    TestsResultsTemplates(class_list[i], dict_simulations[class_list[i]]['model_name']) 
    for i in range(nb_class)
]

# Onglets pour chaque classe d'actifs
for i, template in enumerate(tests_results_templates):
    with tabs[i]:
        # Reading of the market data that will be used to calibrate the chosen model
        df_temp = dict_simulations[template.asset_class]['calibration_df']
        
        if not df_temp.empty:
            # Name of the asset model for the asset class
            model_name_temp = dict_simulations[template.asset_class]['model_name']
            type_temp = MODEL_TYPE[model_name_temp]
            
            # Créer un préfixe de clé unique
            key_prefix = f"calib_{template.asset_class.replace(' ', '_').lower()}"

            # Calibration of the model
            with st.spinner(f"Calibrating {model_name_temp} model..."):
                model_temp = Model(name=model_name_temp)
                calibrated_params = model_temp.calibration(df_temp)
                if template.asset_class == "Real Estate":
                    # Ajuster la volatilité selon le ratio spécifié
                    calibrated_params['sigma'] *= re_vol_ratio
                
                # Stocker les paramètres calibrés
                calibrated_parameters[template.asset_class] = calibrated_params.copy()
            
            '''
            ### Calibrated parameters
            '''
            # Affichage des paramètres calibrés dans un expander
            with st.expander("📊 View Calibrated Parameters", expanded=True):
                params_df = pd.DataFrame([calibrated_params]).T
                params_df.columns = ['Value']
                st.dataframe(params_df, use_container_width=True)
            
            # Projection using the model
            with st.spinner(f"Running {N} simulations..."):
                match model_name_temp:
                    case 'Black-Scholes':
                        proj_df_temp = model_temp.blackscholes_projection(calibrated_params, T, N, rfr_temp)
                        # Martingality test
                        mart_test_temp = Martingality_test(type=type_temp)
                        martingality_df_temp = mart_test_temp.martingality_calcs(proj_df_temp, rfr_temp, 0.05)
                        # Display of results
                        template.render_index(proj_df_temp, martingality_df_temp, key_prefix)
                    
                    case 'Dupire':
                        proj_df_temp = model_temp.dupire_projection(calibrated_params, T, N, rfr_temp)
                        mart_test_temp = Martingality_test(type=type_temp)
                        martingality_df_temp = mart_test_temp.martingality_calcs(proj_df_temp, rfr_temp, 0.05)
                        template.render_index(proj_df_temp, martingality_df_temp, key_prefix)
                    
                    case 'Heston':
                        proj_df_temp = model_temp.heston_projection(calibrated_params, T, N, rfr_temp)
                        # Martingality test
                        mart_test_temp = Martingality_test(type=type_temp)
                        martingality_df_temp = mart_test_temp.martingality_calcs(proj_df_temp, rfr_temp, 0.05)
                        # Display of results
                        template.render_index(proj_df_temp, martingality_df_temp, key_prefix)
                    
                    case 'Vasicek':
                        model_spot_curve = model_temp.vasicek_spot_curve(T)
                        template.display_calibrated_ir(rfr_temp, model_spot_curve, key_prefix)
                        proj_dict_temp = model_temp.vasicek_projection(calibrated_params, T, N, rfr_temp)
                        # Martingality test
                        mart_test_temp = Martingality_test(type=type_temp)
                        martingality_dict_temp = mart_test_temp.martingality_calcs(proj_dict_temp, model_spot_curve, 0.05)
                        # Display of results
                        template.render_interest_rates(martingality_dict_temp, key_prefix)
        else:
            st.warning(f"⚠️ No data uploaded for {class_list[i]}.")
            st.info("Please upload calibration data in the sidebar to see results.")

# Onglet Dependency Structure (4ème avant la fin)
with tabs[-4]:
    # Render et stocker la matrice de corrélation dans session_state
    final_correlation_matrix = render_dependency_tab(
        models_ready=models_ready,
        selected_models=selected_models,
        nb_weiner_dict=NB_WEINER,
        calibrated_parameters=calibrated_parameters,
        empirical_corr_df=None
    )
    
    # Stocker la matrice finale dans session_state pour l'onglet Simulation
    if final_correlation_matrix is not None:
        st.session_state['final_correlation_matrix'] = final_correlation_matrix

# Onglet Correlated Simulations (3ème avant la fin)
with tabs[-3]:
    # Préparer le dictionnaire des modèles avec leurs paramètres
    models_dict_for_simulation = {}
    for asset_class in required_classes:
        if asset_class in calibrated_parameters:
            models_dict_for_simulation[asset_class] = {
                'model_name': selected_models[asset_class],
                'params': calibrated_parameters[asset_class]
            }
    
    # Récupérer la matrice de corrélation depuis session_state
    correlation_matrix = st.session_state.get('final_correlation_matrix', None)
    
    # Importer le CorrelationManager pour le passer à la simulation
    from scripts.correlation_manager import CorrelationManager
    
    if correlation_matrix is not None and models_dict_for_simulation:
        # Créer une instance de CorrelationManager
        corr_manager = CorrelationManager(
            model_types=selected_models,
            nb_weiner_dict=NB_WEINER,
            calibrated_parameters=calibrated_parameters
        )
        
        # Render l'onglet de simulation
        render_simulation_tab(
            models_dict=models_dict_for_simulation,
            correlation_matrix=correlation_matrix,
            corr_manager=corr_manager,
            T=T,
            N=N,
            rfr=rfr_temp
        )
    else:
        st.info("⏳ Please complete the Dependency Structure configuration first to enable simulations.")

# Onglet Martingality Tests (2ème avant la fin)
with tabs[-2]:
    # Préparer le dictionnaire des modèles avec leurs paramètres
    models_dict_for_tests = {}
    for asset_class in required_classes:
        if asset_class in calibrated_parameters:
            models_dict_for_tests[asset_class] = {
                'model_name': selected_models[asset_class],
                'params': calibrated_parameters[asset_class]
            }
    
    # Récupérer la matrice de corrélation depuis session_state
    correlation_matrix = st.session_state.get('final_correlation_matrix', None)
    
    if models_dict_for_tests:
        # Render l'onglet de tests
        render_tests_tab(
            models_dict=models_dict_for_tests,
            rfr=rfr_temp,
            correlation_matrix=correlation_matrix
        )
    else:
        st.info("⏳ Please calibrate all models and run simulations first to enable tests.")

# Onglet Export (dernier onglet - NOUVEAU)
with tabs[-1]:
    render_export_tab()