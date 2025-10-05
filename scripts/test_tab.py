"""
Module pour l'onglet de tests de martingalité sur les simulations corrélées
"""

import streamlit as st
import pandas as pd
from scripts.class_test import Martingality_test
from scripts.class_template import TestsResultsTemplates
from scripts.class_model import MODEL_TYPE


def render_tests_tab(models_dict, rfr, correlation_matrix=None):
    """
    Affiche l'onglet de tests de martingalité et market consistency
    
    :param models_dict: Dict {asset_class: {'model_name': str, 'params': dict}}
    :param rfr: Fonction des taux sans risque
    :param correlation_matrix: Matrice de corrélation utilisée (optionnel)
    """
    
    st.header("🧪 Martingality & Market Consistency Tests")
    
    # Vérifier si des simulations existent
    if 'simulation_results' not in st.session_state:
        st.info("⏳ Please run the correlated simulations first to perform tests.")
        st.stop()
    
    # Récupérer les résultats de simulation
    results = st.session_state['simulation_results']
    trajectories = results['trajectories']
    metadata = results['metadata']
    
    # Section 1: Informations sur la simulation testée
    st.subheader("📋 Simulation Information")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Seed", metadata['seed'])
    
    with col2:
        st.metric("Simulations", metadata['N'])
    
    with col3:
        st.metric("Horizon", f"{metadata['T']} years")
    
    with col4:
        timestamp = metadata['timestamp']
        st.metric("Generated", timestamp.strftime("%H:%M"))
    
    # Afficher la matrice de corrélation utilisée si disponible
    if correlation_matrix is not None:
        with st.expander("🔗 View Correlation Matrix Used"):
            st.dataframe(
                correlation_matrix.style.background_gradient(cmap='RdYlGn', vmin=-1, vmax=1).format("{:.3f}"),
                use_container_width=True
            )
    
    st.markdown("---")
    
    # Section 2: Tests par classe d'actifs
    st.subheader("📊 Tests by Asset Class")
    
    # Créer des tabs pour chaque classe d'actifs
    asset_classes = list(trajectories.keys())
    tabs = st.tabs(asset_classes)
    
    # Dictionnaire pour stocker les résultats des tests
    test_results = {}
    
    for i, asset_class in enumerate(asset_classes):
        with tabs[i]:
            model_name = models_dict[asset_class]['model_name']
            traj_data = trajectories[asset_class]
            
            st.markdown(f"### {asset_class} - {model_name}")
            
            # Créer le template de résultats
            template = TestsResultsTemplates(asset_class, model_name)
            
            # Récupérer le type de modèle
            type_temp = MODEL_TYPE[model_name]
            
            # Créer un préfixe de clé unique basé sur l'asset class
            key_prefix = f"test_{asset_class.replace(' ', '_').lower()}"

            # Lancer les tests selon le type
            with st.spinner(f"Running tests for {asset_class}..."):
                try:
                    mart_test = Martingality_test(type=type_temp)
                    
                    if type_temp == 'index':
                        # Tests pour modèles d'indices
                        martingality_df = mart_test.martingality_calcs(traj_data, rfr, 0.05)
                        
                        # Affichage avec le template
                        template.render_index(traj_data, martingality_df, key_prefix)
                        
                        # Stocker les résultats
                        test_results[asset_class] = {
                            'type': 'index',
                            'martingality_df': martingality_df,
                            'pass_rate': _calculate_pass_rate(martingality_df)
                        }
                    
                    elif type_temp == 'price':
                        # Tests pour modèles de prix (taux)
                        # Calculer la courbe spot du modèle
                        from scripts.class_model import Model
                        model_temp = Model(name=model_name)
                        
                        if model_name == 'Vasicek':
                            model_temp.parameters = models_dict[asset_class]['params']
                            model_spot_curve = model_temp.vasicek_spot_curve(metadata['T'])
                            
                            # Afficher la courbe calibrée
                            template.display_calibrated_ir(rfr, model_spot_curve, key_prefix)
                            
                            # Tests de martingalité
                            martingality_dict = mart_test.martingality_calcs(
                                traj_data, 
                                model_spot_curve, 
                                0.05
                            )
                            
                            # Affichage avec le template
                            template.render_interest_rates(martingality_dict, key_prefix)
                            
                            # Stocker les résultats
                            test_results[asset_class] = {
                                'type': 'price',
                                'deflator_df': martingality_dict["Deflator"],
                                'zc_price_df': martingality_dict["ZC_Price"],
                                'deflator_pass_rate': _calculate_pass_rate(martingality_dict["Deflator"]),
                                'zc_pass_rate': _calculate_pass_rate(martingality_dict["ZC_Price"])
                            }
                        
                        elif model_name == 'G2++':
                            st.warning("⚠️ Tests for G2++ model not yet implemented")
                            test_results[asset_class] = None
                    
                    else:
                        st.error(f"❌ Unknown model type: {type_temp}")
                        test_results[asset_class] = None
                
                except Exception as e:
                    st.error(f"❌ Error running tests: {e}")
                    test_results[asset_class] = None
    
    # Section 3: Résumé des tests
    st.markdown("---")
    st.subheader("📈 Tests Summary")
    
    _render_test_summary(test_results, asset_classes)
    
    # Section 4: Analyse de la corrélation empirique des trajectoires
    st.markdown("---")
    st.subheader("🔍 Empirical Correlation Analysis")
    
    _render_empirical_correlation_analysis(trajectories, models_dict, correlation_matrix)
    
    # Section 5: Export des résultats de tests
    st.markdown("---")
    st.subheader("💾 Export Test Results")
    
    _render_export_section(test_results, asset_classes, metadata)


def _calculate_pass_rate(martingality_df):
    """
    Calcule le taux de réussite des tests de martingalité
    
    :param martingality_df: DataFrame avec colonne 'Test' (booléens)
    :return: Taux de réussite (0-1)
    """
    if 'Test' not in martingality_df.columns:
        return None
    
    total_tests = len(martingality_df)
    passed_tests = martingality_df['Test'].sum()
    
    return passed_tests / total_tests if total_tests > 0 else 0


def _render_test_summary(test_results, asset_classes):
    """
    Affiche un résumé des résultats de tests
    
    :param test_results: Dict des résultats de tests par asset class
    :param asset_classes: Liste des classes d'actifs
    """
    summary_data = []
    
    for asset_class in asset_classes:
        result = test_results.get(asset_class)
        
        if result is None:
            summary_data.append({
                'Asset Class': asset_class,
                'Type': 'N/A',
                'Test Coverage': 'N/A',
                'Pass Rate': 'N/A',
                'Status': '❌ Error'
            })
        
        elif result['type'] == 'index':
            pass_rate = result['pass_rate']
            summary_data.append({
                'Asset Class': asset_class,
                'Type': 'Index',
                'Test Coverage': 'Martingality',
                'Pass Rate': f"{pass_rate:.1%}",
                'Status': '✅ Pass' if pass_rate >= 0.95 else '⚠️ Partial'
            })
        
        elif result['type'] == 'price':
            deflator_rate = result['deflator_pass_rate']
            zc_rate = result['zc_pass_rate']
            avg_rate = (deflator_rate + zc_rate) / 2
            
            summary_data.append({
                'Asset Class': asset_class,
                'Type': 'Price',
                'Test Coverage': 'Deflator + ZC',
                'Pass Rate': f"{avg_rate:.1%}",
                'Status': '✅ Pass' if avg_rate >= 0.95 else '⚠️ Partial'
            })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Affichage avec style
    st.dataframe(
        summary_df,
        use_container_width=True,
        hide_index=True
    )
    
    # Statistiques globales
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_tests = len([r for r in test_results.values() if r is not None])
        st.metric("Total Tests", total_tests)
    
    with col2:
        passed = len([r for r in test_results.values() 
                     if r is not None and (
                         (r['type'] == 'index' and r['pass_rate'] >= 0.95) or
                         (r['type'] == 'price' and (r['deflator_pass_rate'] + r['zc_pass_rate'])/2 >= 0.95)
                     )])
        st.metric("Passed Tests", passed)
    
    with col3:
        pass_percentage = (passed / total_tests * 100) if total_tests > 0 else 0
        st.metric("Overall Pass Rate", f"{pass_percentage:.1f}%")


def _render_empirical_correlation_analysis(trajectories, models_dict, theoretical_corr):
    """
    Analyse la corrélation empirique des trajectoires générées
    
    :param trajectories: Dict des trajectoires par asset class
    :param models_dict: Dict des modèles
    :param theoretical_corr: Matrice de corrélation théorique utilisée
    """
    st.markdown("Compare theoretical correlations with empirical correlations from generated trajectories.")
    
    # Extraire les séries de returns pour chaque asset class
    returns_dict = {}
    
    for asset_class, traj_data in trajectories.items():
        if isinstance(traj_data, dict):
            # Pour les modèles de taux, utiliser le déflateur
            if "Deflator" in traj_data:
                df = traj_data["Deflator"]
            else:
                df = traj_data[1]  # Premier ZC
        else:
            df = traj_data
        
        # Calculer les returns (log returns)
        import numpy as np
        returns = df.apply(lambda x: np.log(x / x.shift(1)), axis=1).dropna(axis=1)
        
        # Moyenne des returns sur toutes les simulations
        avg_returns = returns.mean(axis=0)
        returns_dict[asset_class] = avg_returns
    
    # Créer un DataFrame des returns
    returns_df = pd.DataFrame(returns_dict)
    
    # Calculer la corrélation empirique
    empirical_corr = returns_df.corr()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Empirical Correlation (from trajectories)**")
        st.dataframe(
            empirical_corr.style.background_gradient(cmap='RdYlGn', vmin=-1, vmax=1).format("{:.3f}"),
            use_container_width=True
        )
    
    with col2:
        if theoretical_corr is not None:
            st.markdown("**Theoretical Correlation (asset level)**")
            
            # Extraire la sous-matrice pour les asset classes
            asset_classes = list(trajectories.keys())
            
            # Créer une matrice de corrélation théorique au niveau des assets
            # En moyennant les corrélations des browniens de chaque asset
            from scripts.correlation_manager import CorrelationManager
            
            # Recréer le mapping pour extraire les corrélations
            model_types = {ac: models_dict[ac]['model_name'] for ac in asset_classes}
            from scripts.class_model import NB_WEINER
            
            corr_manager = CorrelationManager(model_types, NB_WEINER)
            mapping = corr_manager.get_brownian_mapping()
            
            # Construire la matrice asset-level
            asset_level_corr = pd.DataFrame(
                index=asset_classes,
                columns=asset_classes,
                dtype=float
            )
            
            for i, asset_i in enumerate(asset_classes):
                for j, asset_j in enumerate(asset_classes):
                    if i == j:
                        asset_level_corr.loc[asset_i, asset_j] = 1.0
                    else:
                        # Prendre la corrélation du premier brownien de chaque asset
                        idx_i = mapping[asset_i]['start_idx']
                        idx_j = mapping[asset_j]['start_idx']
                        
                        brownian_i = f"W{idx_i + 1}"
                        brownian_j = f"W{idx_j + 1}"
                        
                        asset_level_corr.loc[asset_i, asset_j] = theoretical_corr.loc[brownian_i, brownian_j]
            
            st.dataframe(
                asset_level_corr.style.background_gradient(cmap='RdYlGn', vmin=-1, vmax=1).format("{:.3f}"),
                use_container_width=True
            )
    
    # Calcul des différences
    if theoretical_corr is not None:
        st.markdown("**Difference (Empirical - Theoretical)**")
        
        diff_corr = empirical_corr - asset_level_corr.astype(float)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.dataframe(
                diff_corr.style.background_gradient(cmap='RdBu', vmin=-0.2, vmax=0.2).format("{:.3f}"),
                use_container_width=True
            )
        
        with col2:
            # Statistiques sur les différences
            import numpy as np
            
            # Extraire uniquement le triangle supérieur (hors diagonale)
            mask = np.triu(np.ones_like(diff_corr, dtype=bool), k=1)
            diffs = diff_corr.values[mask]
            
            stats_data = {
                'Max Abs Diff': np.max(np.abs(diffs)),
                'Mean Abs Diff': np.mean(np.abs(diffs)),
                'Std Diff': np.std(diffs),
                'RMSE': np.sqrt(np.mean(diffs**2))
            }
            
            stats_df = pd.DataFrame([stats_data]).T.rename(columns={0: 'Value'})
            st.dataframe(stats_df.style.format("{:.4f}"), use_container_width=True)
        
        # Interprétation
        max_diff = np.max(np.abs(diffs))
        if max_diff < 0.05:
            st.success("✅ Excellent match between empirical and theoretical correlations!")
        elif max_diff < 0.10:
            st.info("ℹ️ Good match between empirical and theoretical correlations.")
        else:
            st.warning("⚠️ Noticeable difference between empirical and theoretical correlations. Consider increasing the number of simulations.")


def _render_export_section(test_results, asset_classes, metadata):
    """
    Section d'export des résultats de tests
    
    :param test_results: Dict des résultats
    :param asset_classes: Liste des asset classes
    :param metadata: Métadonnées de la simulation
    """
    cols = st.columns(len(asset_classes))
    
    for i, asset_class in enumerate(asset_classes):
        with cols[i]:
            result = test_results.get(asset_class)
            
            if result is None:
                st.info(f"No data for {asset_class}")
                continue
            
            if result['type'] == 'index':
                csv_data = result['martingality_df'].to_csv()
                filename = f"{asset_class.replace(' ', '_')}_martingality_test.csv"
            
            elif result['type'] == 'price':
                # Combiner deflator et ZC dans un seul export
                combined_df = pd.concat([
                    result['deflator_df'].add_prefix('Deflator_'),
                    result['zc_price_df'].add_prefix('ZC_')
                ], axis=1)
                csv_data = combined_df.to_csv()
                filename = f"{asset_class.replace(' ', '_')}_martingality_tests.csv"
            
            st.download_button(
                label=f"📥 {asset_class}",
                data=csv_data,
                file_name=filename,
                mime="text/csv",
                use_container_width=True
            )
    
    # Export du résumé complet
    st.markdown("---")
    
    # Créer un rapport complet
    report_lines = []
    report_lines.append(f"MARTINGALITY TEST REPORT")
    report_lines.append(f"=" * 50)
    report_lines.append(f"Seed: {metadata['seed']}")
    report_lines.append(f"Simulations: {metadata['N']}")
    report_lines.append(f"Horizon: {metadata['T']} years")
    report_lines.append(f"Generated: {metadata['timestamp']}")
    report_lines.append("")
    report_lines.append("RESULTS BY ASSET CLASS")
    report_lines.append("-" * 50)
    
    for asset_class in asset_classes:
        result = test_results.get(asset_class)
        report_lines.append(f"\n{asset_class}:")
        
        if result is None:
            report_lines.append("  Status: Error or not tested")
        elif result['type'] == 'index':
            report_lines.append(f"  Type: Index model")
            report_lines.append(f"  Pass Rate: {result['pass_rate']:.2%}")
        elif result['type'] == 'price':
            report_lines.append(f"  Type: Price model")
            report_lines.append(f"  Deflator Pass Rate: {result['deflator_pass_rate']:.2%}")
            report_lines.append(f"  ZC Pass Rate: {result['zc_pass_rate']:.2%}")
    
    report_text = "\n".join(report_lines)
    
    st.download_button(
        label="📥 Download Full Test Report",
        data=report_text,
        file_name=f"martingality_report_seed_{metadata['seed']}.txt",
        mime="text/plain"
    )