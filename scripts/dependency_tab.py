"""
Module pour l'onglet de structure de dépendance dans Streamlit
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scripts.correlation_manager import CorrelationManager


def render_dependency_tab(models_ready, selected_models, nb_weiner_dict, calibrated_parameters=None, empirical_corr_df=None):
    """
    Affiche l'onglet de structure de dépendance
    
    :param models_ready: Boolean indiquant si tous les modèles sont calibrés
    :param selected_models: Dictionnaire {asset_class: model_name}
    :param nb_weiner_dict: Dictionnaire {model_name: nb_browniens}
    :param calibrated_parameters: Dictionnaire {asset_class: params_dict} avec les paramètres calibrés
    :param empirical_corr_df: DataFrame de corrélation empirique (optionnel)
    :return: Matrice de corrélation finale (PSD) ou None
    """
    
    st.header("🔗 Dependency Structure")
    
    if not models_ready:
        st.info("⏳ Please calibrate all required models (Interest rates, Equity, Real Estate) to configure the dependency structure.")
        return None
    
    st.success("✅ All required models are calibrated!")
    
    # Valeur par défaut pour les paramètres calibrés
    if calibrated_parameters is None:
        calibrated_parameters = {}
    
    # Création du gestionnaire de corrélations
    corr_manager = CorrelationManager(selected_models, nb_weiner_dict, calibrated_parameters)
    
    # Section 1: Information sur les browniens
    st.subheader("📊 Brownian Motion Structure")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Brownian Motions", corr_manager.get_total_brownians())
    
    with col2:
        st.metric("Asset Classes", len(selected_models))
    
    # Affichage du mapping des browniens
    with st.expander("🔍 View Brownian Mapping Details"):
        mapping_data = []
        for asset_class, info in corr_manager.get_brownian_mapping().items():
            brownians = [f"W{i+1}" for i in range(info['start_idx'], info['end_idx'])]
            
            # Récupérer la corrélation intra-modèle si applicable
            intra_corr = "N/A"
            if info['nb_brownians'] == 2:
                model_name = info['model']
                if asset_class in calibrated_parameters:
                    params = calibrated_parameters[asset_class]
                    if model_name == 'Heston':
                        intra_corr = f"{params.get('rho', -0.5):.4f}"
                    elif model_name == 'G2++':
                        intra_corr = f"{params.get('rho', 0.3):.4f}"
                else:
                    if model_name == 'Heston':
                        intra_corr = "-0.5000 (default)"
                    elif model_name == 'G2++':
                        intra_corr = "0.3000 (default)"
            
            mapping_data.append({
                'Asset Class': asset_class,
                'Model': info['model'],
                'Nb Brownians': info['nb_brownians'],
                'Brownian Indices': ', '.join(brownians),
                'Intra-Model Correlation': intra_corr
            })
        
        mapping_df = pd.DataFrame(mapping_data)
        st.dataframe(mapping_df, use_container_width=True)
        
        # Affichage détaillé des corrélations intra-modèles
        if any(info['nb_brownians'] == 2 for info in corr_manager.get_brownian_mapping().values()):
            st.markdown("#### Intra-Model Correlations Details")
            
            for asset_class, info in corr_manager.get_brownian_mapping().items():
                if info['nb_brownians'] == 2:
                    model_name = info['model']
                    
                    if asset_class in calibrated_parameters:
                        params = calibrated_parameters[asset_class]
                        if model_name == 'Heston':
                            rho_value = params.get('rho', -0.5)
                            st.success(f"✅ **{asset_class} ({model_name})**: ρ = {rho_value:.4f} (calibrated)")
                            st.caption(f"   Correlation between W{info['start_idx']+1} (price) and W{info['start_idx']+2} (variance)")
                        elif model_name == 'G2++':
                            rho_value = params.get('rho', 0.3)
                            st.success(f"✅ **{asset_class} ({model_name})**: ρ = {rho_value:.4f} (calibrated)")
                            st.caption(f"   Correlation between W{info['start_idx']+1} (factor 1) and W{info['start_idx']+2} (factor 2)")
                    else:
                        if model_name == 'Heston':
                            st.warning(f"⚠️ **{asset_class} ({model_name})**: ρ = -0.5000 (default - not calibrated yet)")
                        elif model_name == 'G2++':
                            st.warning(f"⚠️ **{asset_class} ({model_name})**: ρ = 0.3000 (default - not calibrated yet)")
    
    # Section 2: Corrélation empirique
    st.subheader("📈 Empirical Correlation Matrix")
    
    if empirical_corr_df is None:
        # Matrice de corrélation par défaut si non fournie
        asset_classes = list(selected_models.keys())
        empirical_corr_df = pd.DataFrame(
            [[1.0, 0.6, 0.5],
             [0.6, 1.0, 0.7],
             [0.5, 0.7, 1.0]],
            index=asset_classes,
            columns=asset_classes
        )
        st.warning("⚠️ Using default empirical correlations. You can upload your own data.")
    
    # Option pour uploader une matrice de corrélation personnalisée
    with st.expander("⬆️ Upload Custom Empirical Correlation"):
        uploaded_corr = st.file_uploader(
            "Upload correlation matrix (CSV):",
            type=["csv"],
            key="corr_upload",
            help="Upload a CSV file with asset class correlations"
        )
        
        if uploaded_corr is not None:
            try:
                empirical_corr_df = pd.read_csv(uploaded_corr, index_col=0)
                st.success("✅ Custom correlation matrix loaded!")
            except Exception as e:
                st.error(f"Error loading correlation matrix: {e}")
    
    # Affichage de la matrice empirique
    st.dataframe(
        empirical_corr_df.style.background_gradient(cmap='RdYlGn', vmin=-1, vmax=1),
        use_container_width=True
    )
    
    # Section 3: Construction de la matrice théorique
    st.subheader("🔧 Theoretical Correlation Matrix")
    
    theoretical_corr = corr_manager.build_theoretical_correlation_matrix(empirical_corr_df)
    
    # Vérification PSD
    is_psd, eigenvalues = corr_manager.is_psd(theoretical_corr)
    
    col1, col2 = st.columns(2)
    with col1:
        if is_psd:
            st.success("✅ Matrix is Positive Semi-Definite")
        else:
            st.error("❌ Matrix is NOT Positive Semi-Definite")
    
    with col2:
        min_eigenvalue = np.min(eigenvalues)
        st.metric("Minimum Eigenvalue", f"{min_eigenvalue:.6f}")
    
    # Affichage de la matrice théorique
    with st.expander("📊 View Theoretical Correlation Matrix"):
        st.dataframe(
            theoretical_corr.style.background_gradient(cmap='RdYlGn', vmin=-1, vmax=1).format("{:.3f}"),
            use_container_width=True
        )
    
    # Affichage des valeurs propres
    with st.expander("🔢 View Eigenvalues"):
        eigen_df = pd.DataFrame({
            'Index': range(1, len(eigenvalues) + 1),
            'Eigenvalue': eigenvalues,
            'Status': ['✅ Positive' if ev >= 0 else '❌ Negative' for ev in eigenvalues]
        })
        st.dataframe(eigen_df, use_container_width=True)
        
        # Graphique des valeurs propres
        fig_eigen = go.Figure()
        fig_eigen.add_trace(go.Bar(
            x=eigen_df['Index'],
            y=eigen_df['Eigenvalue'],
            marker_color=['green' if ev >= 0 else 'red' for ev in eigenvalues]
        ))
        fig_eigen.update_layout(
            title='Eigenvalues of Theoretical Correlation Matrix',
            xaxis_title='Eigenvalue Index',
            yaxis_title='Value',
            showlegend=False
        )
        st.plotly_chart(fig_eigen, use_container_width=True)
    
    # Variable pour stocker la matrice finale à retourner
    final_matrix = theoretical_corr
    
    # Section 4: Correction PSD si nécessaire
    if not is_psd:
        st.subheader("🔧 PSD Correction (Higham Algorithm)")
        
        with st.spinner("Applying Higham correction..."):
            corrected_corr = corr_manager.make_psd_higham(theoretical_corr)
        
        # Vérification de la matrice corrigée
        is_psd_corrected, eigenvalues_corrected = corr_manager.is_psd(corrected_corr)
        
        if is_psd_corrected:
            st.success("✅ Matrix successfully corrected to PSD!")
            final_matrix = corrected_corr
        else:
            st.warning("⚠️ Matrix correction may need additional iterations")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Min Eigenvalue (Original)", f"{np.min(eigenvalues):.6f}")
        with col2:
            st.metric("Min Eigenvalue (Corrected)", f"{np.min(eigenvalues_corrected):.6f}")
        
        # Affichage de la matrice corrigée
        st.subheader("📊 Corrected Correlation Matrix")
        st.dataframe(
            corrected_corr.style.background_gradient(cmap='RdYlGn', vmin=-1, vmax=1).format("{:.3f}"),
            use_container_width=True
        )
        
        # Section 5: Analyse des différences
        st.subheader("📉 Correction Analysis")
        
        # Statistiques de correction
        stats_df = corr_manager.calculate_correction_stats(theoretical_corr, corrected_corr)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.dataframe(stats_df, use_container_width=True)
        
        with col2:
            # Matrice des différences
            diff_matrix = corr_manager.get_difference_matrix(theoretical_corr, corrected_corr)
            
            st.write("**Element-wise Differences:**")
            st.dataframe(
                diff_matrix.style.background_gradient(cmap='RdBu', vmin=-0.1, vmax=0.1).format("{:.4f}"),
                use_container_width=True
            )
        
        # Heatmap des différences
        fig_diff = go.Figure(data=go.Heatmap(
            z=diff_matrix.values,
            x=diff_matrix.columns,
            y=diff_matrix.index,
            colorscale='RdBu',
            zmid=0,
            text=diff_matrix.values,
            texttemplate='%{text:.4f}',
            textfont={"size": 8},
            colorbar=dict(title="Difference")
        ))
        
        fig_diff.update_layout(
            title='Heatmap of Differences (Corrected - Original)',
            xaxis_title='',
            yaxis_title='',
            height=500
        )
        
        st.plotly_chart(fig_diff, use_container_width=True)
        
        # Bouton pour exporter les matrices
        st.subheader("💾 Export Matrices")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv_theoretical = theoretical_corr.to_csv()
            st.download_button(
                label="📥 Download Theoretical Matrix",
                data=csv_theoretical,
                file_name="theoretical_correlation.csv",
                mime="text/csv"
            )
        
        with col2:
            csv_corrected = corrected_corr.to_csv()
            st.download_button(
                label="📥 Download Corrected Matrix",
                data=csv_corrected,
                file_name="corrected_correlation.csv",
                mime="text/csv"
            )
        
        with col3:
            csv_diff = diff_matrix.to_csv()
            st.download_button(
                label="📥 Download Difference Matrix",
                data=csv_diff,
                file_name="difference_matrix.csv",
                mime="text/csv"
            )
    
    else:
        st.info("ℹ️ The theoretical correlation matrix is already PSD. No correction needed!")
        
        # Bouton pour exporter
        csv_theoretical = theoretical_corr.to_csv()
        st.download_button(
            label="📥 Download Correlation Matrix",
            data=csv_theoretical,
            file_name="correlation_matrix.csv",
            mime="text/csv"
        )
    
    # Retourner la matrice finale (corrigée si nécessaire, sinon théorique)
    return final_matrix