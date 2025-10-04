"""
Module pour l'onglet de simulation de trajectoires corrélées dans Streamlit
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
from scripts.simulation_engine import CorrelatedSimulationEngine


def render_simulation_tab(models_dict, correlation_matrix, corr_manager, T, N, rfr):
    """
    Affiche l'onglet de simulation de trajectoires corrélées
    
    :param models_dict: Dict {asset_class: {'model_name': str, 'params': dict}}
    :param correlation_matrix: Matrice de corrélation finale (PSD)
    :param corr_manager: Instance de CorrelationManager
    :param T: Horizon de projection
    :param N: Nombre de simulations
    :param rfr: Fonction des taux sans risque
    """
    
    st.header("🎲 Correlated Simulations")
    
    # Vérifications préalables
    if not models_dict or correlation_matrix is None:
        st.info("⏳ Please calibrate all models and configure the correlation matrix first.")
        st.stop()
    
    # Vérifier que tous les modèles ont des paramètres
    missing_params = [ac for ac, info in models_dict.items() if not info.get('params')]
    if missing_params:
        st.warning(f"⚠️ Missing calibrated parameters for: {', '.join(missing_params)}")
        st.stop()
    
    # Section 1: Configuration de la simulation
    st.subheader("⚙️ Simulation Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Projection Horizon (T)", f"{T} years")
    
    with col2:
        st.metric("Number of Simulations (N)", N)
    
    with col3:
        st.metric("Total Brownian Motions", corr_manager.get_total_brownians())
    
    # Configuration de la graine
    st.markdown("---")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        seed_input = st.number_input(
            "Random Seed",
            min_value=0,
            max_value=2**31 - 1,
            value=st.session_state.get('last_seed', 42),
            step=1,
            help="Seed for reproducibility. Use the same seed to reproduce exact results."
        )
    
    with col2:
        if st.button("🎲 Random Seed"):
            seed_input = np.random.randint(0, 2**31 - 1)
            st.session_state['last_seed'] = seed_input
            st.rerun()
    
    # Bouton de lancement
    st.markdown("---")
    run_simulation = st.button(
        "▶️ Run Correlated Simulation",
        type="primary",
        use_container_width=True
    )
    
    # Lancement de la simulation
    if run_simulation:
        with st.spinner("🔄 Generating correlated trajectories..."):
            # Créer le moteur de simulation
            engine = CorrelatedSimulationEngine(
                models_dict=models_dict,
                correlation_matrix=correlation_matrix,
                corr_manager=corr_manager,
                T=T,
                N=N,
                rfr=rfr,
                seed=seed_input
            )
            
            # Lancer les simulations
            try:
                trajectories = engine.run_simulations()
                
                # Stocker dans session_state
                st.session_state['simulation_results'] = engine.export_to_dict()
                st.session_state['last_seed'] = seed_input
                
                st.success(f"✅ Simulation completed successfully! (Seed: {seed_input})")
            
            except Exception as e:
                st.error(f"❌ Error during simulation: {e}")
                st.stop()
    
    # Section 2: Affichage des résultats
    if 'simulation_results' in st.session_state:
        results = st.session_state['simulation_results']
        trajectories = results['trajectories']
        metadata = results['metadata']
        
        st.markdown("---")
        st.subheader("📊 Simulation Results")
        
        # Métadonnées
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Seed Used", metadata['seed'])
        
        with col2:
            st.metric("Simulations", metadata['N'])
        
        with col3:
            st.metric("Time Steps", f"{metadata['T']} years")
        
        with col4:
            timestamp = metadata['timestamp']
            st.metric("Generated", timestamp.strftime("%H:%M:%S"))
        
        # Visualisation des trajectoires par asset class
        st.markdown("---")
        st.subheader("📈 Trajectories by Asset Class")
        
        # Créer des tabs pour chaque classe d'actifs
        asset_classes = list(trajectories.keys())
        tabs = st.tabs(asset_classes)
        
        for i, asset_class in enumerate(asset_classes):
            with tabs[i]:
                model_name = models_dict[asset_class]['model_name']
                
                # Récupérer les trajectoires
                traj_data = trajectories[asset_class]
                
                # Vérifier le type de données (DataFrame vs Dict pour les taux)
                if isinstance(traj_data, dict):
                    # Modèles de taux (Vasicek, G2++)
                    _render_rate_trajectories(asset_class, model_name, traj_data, N)
                else:
                    # Modèles d'indices (Black-Scholes, Dupire, Heston)
                    _render_index_trajectories(asset_class, model_name, traj_data, N)
        
        # Section 3: Export des résultats
        st.markdown("---")
        st.subheader("💾 Export Results")
        
        cols = st.columns(len(asset_classes))
        for i, asset_class in enumerate(asset_classes):
            with cols[i]:
                traj_data = trajectories[asset_class]
                
                if isinstance(traj_data, dict):
                    # Pour les modèles de taux, exporter le déflateur
                    if "Deflator" in traj_data:
                        csv_data = traj_data["Deflator"].to_csv()
                        filename = f"{asset_class.replace(' ', '_')}_deflator.csv"
                    else:
                        csv_data = traj_data[1].to_csv()
                        filename = f"{asset_class.replace(' ', '_')}_rates.csv"
                else:
                    csv_data = traj_data.to_csv()
                    filename = f"{asset_class.replace(' ', '_')}_trajectories.csv"
                
                st.download_button(
                    label=f"📥 {asset_class}",
                    data=csv_data,
                    file_name=filename,
                    mime="text/csv",
                    use_container_width=True
                )
        
        # Export de la matrice de corrélation utilisée
        st.markdown("---")
        csv_corr = results['correlation_matrix'].to_csv()
        st.download_button(
            label="📥 Download Correlation Matrix Used",
            data=csv_corr,
            file_name=f"correlation_matrix_seed_{metadata['seed']}.csv",
            mime="text/csv"
        )
        
        # Section 4: Historique des simulations
        _render_simulation_history()
    
    else:
        st.info("👆 Configure the seed and click 'Run Correlated Simulation' to start.")


def _render_index_trajectories(asset_class, model_name, df_trajectories, N):
    """
    Affiche les trajectoires pour les modèles d'indices
    
    :param asset_class: Nom de la classe d'actifs
    :param model_name: Nom du modèle
    :param df_trajectories: DataFrame des trajectoires
    :param N: Nombre de simulations
    """
    st.markdown(f"**Model:** {model_name}")
    
    # Statistiques descriptives
    col1, col2, col3, col4 = st.columns(4)
    
    final_values = df_trajectories.iloc[:, -1]
    
    with col1:
        st.metric("Final Mean", f"{final_values.mean():.2f}")
    
    with col2:
        st.metric("Final Std", f"{final_values.std():.2f}")
    
    with col3:
        st.metric("Final Min", f"{final_values.min():.2f}")
    
    with col4:
        st.metric("Final Max", f"{final_values.max():.2f}")
    
    # Graphique des trajectoires
    fig = go.Figure()
    
    # Limiter l'affichage à 50 trajectoires pour la lisibilité
    num_to_plot = min(50, N)
    
    for idx in df_trajectories.index[:num_to_plot]:
        fig.add_trace(go.Scatter(
            x=df_trajectories.columns,
            y=df_trajectories.loc[idx],
            mode='lines',
            name=f"Sim {idx}",
            line=dict(width=0.5),
            opacity=0.6,
            showlegend=False
        ))
    
    # Ajouter la moyenne
    fig.add_trace(go.Scatter(
        x=df_trajectories.columns,
        y=df_trajectories.mean(axis=0),
        mode='lines',
        name='Mean',
        line=dict(color='red', width=3),
        showlegend=True
    ))
    
    fig.update_layout(
        title=f'{asset_class} - {model_name} Trajectories (showing {num_to_plot}/{N})',
        xaxis_title='Time (years)',
        yaxis_title='Index Value',
        hovermode='x unified',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Tableau des statistiques par année
    with st.expander("📊 View Statistics by Year"):
        stats_df = pd.DataFrame({
            'Mean': df_trajectories.mean(axis=0),
            'Std': df_trajectories.std(axis=0),
            'Min': df_trajectories.min(axis=0),
            'Q25': df_trajectories.quantile(0.25, axis=0),
            'Median': df_trajectories.median(axis=0),
            'Q75': df_trajectories.quantile(0.75, axis=0),
            'Max': df_trajectories.max(axis=0)
        })
        st.dataframe(stats_df.T, use_container_width=True)


def _render_rate_trajectories(asset_class, model_name, dict_trajectories, N):
    """
    Affiche les trajectoires pour les modèles de taux
    
    :param asset_class: Nom de la classe d'actifs
    :param model_name: Nom du modèle
    :param dict_trajectories: Dict des trajectoires
    :param N: Nombre de simulations
    """
    st.markdown(f"**Model:** {model_name}")
    
    # Sélectionner le déflateur pour visualisation
    if "Deflator" in dict_trajectories:
        df_deflator = dict_trajectories["Deflator"]
        
        st.markdown("**Deflator Trajectories**")
        
        # Statistiques
        col1, col2, col3, col4 = st.columns(4)
        
        final_values = df_deflator.iloc[:, -1]
        
        with col1:
            st.metric("Final Mean", f"{final_values.mean():.4f}")
        
        with col2:
            st.metric("Final Std", f"{final_values.std():.4f}")
        
        with col3:
            st.metric("Final Min", f"{final_values.min():.4f}")
        
        with col4:
            st.metric("Final Max", f"{final_values.max():.4f}")
        
        # Graphique
        fig = go.Figure()
        
        num_to_plot = min(50, N)
        
        for idx in df_deflator.index[:num_to_plot]:
            fig.add_trace(go.Scatter(
                x=df_deflator.columns,
                y=df_deflator.loc[idx],
                mode='lines',
                name=f"Sim {idx}",
                line=dict(width=0.5),
                opacity=0.6,
                showlegend=False
            ))
        
        # Moyenne
        fig.add_trace(go.Scatter(
            x=df_deflator.columns,
            y=df_deflator.mean(axis=0),
            mode='lines',
            name='Mean',
            line=dict(color='red', width=3),
            showlegend=True
        ))
        
        fig.update_layout(
            title=f'{asset_class} - Deflator Trajectories (showing {num_to_plot}/{N})',
            xaxis_title='Time (years)',
            yaxis_title='Deflator Value',
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Afficher aussi les prix ZC pour quelques maturités
    with st.expander("📊 View Zero-Coupon Prices by Maturity"):
        # Sélectionner quelques maturités
        maturities = [m for m in dict_trajectories.keys() if isinstance(m, int)]
        selected_maturities = st.multiselect(
            "Select maturities to display:",
            maturities,
            default=maturities[:min(3, len(maturities))]
        )
        
        if selected_maturities:
            fig_zc = go.Figure()
            
            for maturity in selected_maturities:
                df_zc = dict_trajectories[maturity]
                mean_zc = df_zc.mean(axis=0)
                
                fig_zc.add_trace(go.Scatter(
                    x=df_zc.columns,
                    y=mean_zc,
                    mode='lines',
                    name=f'Maturity {maturity}Y',
                    line=dict(width=2)
                ))
            
            fig_zc.update_layout(
                title='Zero-Coupon Prices by Maturity (Mean)',
                xaxis_title='Time (years)',
                yaxis_title='ZC Price',
                hovermode='x unified',
                height=400
            )
            
            st.plotly_chart(fig_zc, use_container_width=True)


def _render_simulation_history():
    """
    Affiche l'historique des simulations dans session_state
    """
    st.markdown("---")
    st.subheader("📜 Simulation History")
    
    # Initialiser l'historique s'il n'existe pas
    if 'simulation_history' not in st.session_state:
        st.session_state['simulation_history'] = []
    
    # Ajouter la simulation actuelle à l'historique si pas déjà présente
    if 'simulation_results' in st.session_state:
        current_seed = st.session_state['simulation_results']['metadata']['seed']
        
        # Vérifier si cette simulation est déjà dans l'historique
        existing_seeds = [s['seed'] for s in st.session_state['simulation_history']]
        
        if current_seed not in existing_seeds:
            history_entry = {
                'seed': current_seed,
                'timestamp': st.session_state['simulation_results']['metadata']['timestamp'],
                'T': st.session_state['simulation_results']['metadata']['T'],
                'N': st.session_state['simulation_results']['metadata']['N']
            }
            st.session_state['simulation_history'].append(history_entry)
    
    # Afficher l'historique
    if st.session_state['simulation_history']:
        history_df = pd.DataFrame(st.session_state['simulation_history'])
        history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
        history_df = history_df.sort_values('timestamp', ascending=False)
        
        st.dataframe(
            history_df.style.format({
                'timestamp': lambda x: x.strftime('%Y-%m-%d %H:%M:%S'),
                'seed': '{:d}',
                'T': '{:d}',
                'N': '{:d}'
            }),
            use_container_width=True,
            hide_index=True
        )
        
        # Bouton pour effacer l'historique
        if st.button("🗑️ Clear History"):
            st.session_state['simulation_history'] = []
            st.rerun()
    else:
        st.info("No simulations in history yet.")