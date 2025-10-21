"""
Module pour l'onglet Export dans Streamlit - Projet ESG
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from scripts.class_export import OutputGenerator


def render_export_tab():
    """
    Affiche l'onglet Export pour générer et exporter les scénarios ESG.
    S'intègre avec la structure de données du projet ESG.
    """
    st.header("💾 Export ESG Scenarios")
    
    # Vérifier que les simulations corrélées ont été lancées
    if 'simulation_results' not in st.session_state:
        st.info("⏳ Please run the correlated simulations first to enable exports.")
        st.info("👉 Go to the 'Correlated Simulations' tab and click 'Run Correlated Simulation'.")
        st.stop()
    
    # Récupérer les données de simulation
    results = st.session_state['simulation_results']
    trajectories = results['trajectories']
    metadata = results['metadata']
    
    # Vérifier que toutes les classes d'actifs nécessaires sont présentes
    required_classes = ["Interest rates", "Equity", "Real Estate"]
    missing_classes = [cls for cls in required_classes if cls not in trajectories]
    
    if missing_classes:
        st.error(f"❌ Missing required asset classes: {', '.join(missing_classes)}")
        st.info("Please ensure all required models are calibrated and simulated.")
        st.stop()
    
    # Extraire les paramètres depuis session_state
    T = metadata['T']
    N = metadata['N']
    seed = metadata['seed']
    
    # Récupérer les taux de dividende et de loyer depuis session_state
    dividend_rate = st.session_state.get('div_rate', 0.02) / 100  # Valeur par défaut
    rental_rate = st.session_state.get('rent_rate', 0.02) / 100
    
    # Extraire les données des trajectoires
    equity_index = trajectories["Equity"]
    real_estate_index = trajectories["Real Estate"]
    
    # Pour les taux, extraire le dict des prix ZC et le déflateur
    ir_data = trajectories["Interest rates"]
    zc_prices_dict = {k: v for k, v in ir_data.items() if isinstance(k, int)}
    deflator = ir_data["Deflator"]
    
    # Section informations
    st.subheader("📋 Export Information")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Simulations", N)
    
    with col2:
        st.metric("Horizon", f"{T} years")
    
    with col3:
        st.metric("Seed Used", seed)
    
    with col4:
        timestamp = metadata['timestamp']
        st.metric("Generated", timestamp.strftime("%H:%M:%S"))
    
    # Afficher les taux
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Dividend Yield", f"{dividend_rate:.2%}")
    with col2:
        st.metric("Rental Yield", f"{rental_rate:.2%}")
    
    st.markdown("---")
    
    # Configuration du template
    st.subheader("⚙️ Template Configuration")
    
    template_path = st.text_input(
        "Template Path",
        value="inputs/Templates/ESG_template.csv",
        help="Path to the ESG template CSV file (with columns: CLASS;MEASURE;OS_TERM;0)"
    )
    
    template_exists = Path(template_path).exists()
    
    if template_exists:
        st.success(f"✅ Template found: {template_path}")
        
        # Afficher un aperçu du template
        with st.expander("📄 View Template Structure"):
            template_df = pd.read_csv(template_path, sep=';')
            st.dataframe(template_df, use_container_width=True)
            st.caption(f"Template contains {len(template_df)} rows")
    else:
        st.error(f"❌ Template not found: {template_path}")
        st.info("Please ensure the template file exists at the specified path.")
        st.stop()
    
    st.markdown("---")
    
    # Section 1: Génération du rapport
    st.subheader("1️⃣ Generate Export")
    
    st.info("""
    This will create an OutputGenerator instance that will prepare all simulations 
    for export according to the ESG template format.
    """)
    
    if st.button("🔄 Generate Export Structure", type="primary", use_container_width=True):
        with st.spinner("Generating export structure..."):
            try:
                # Créer le générateur
                generator = OutputGenerator(
                    T=T,
                    N=N,
                    dividend_rate=dividend_rate,
                    rental_rate=rental_rate,
                    zc_prices_dict=zc_prices_dict,
                    equity_index=equity_index,
                    real_estate_index=real_estate_index,
                    deflator=deflator,
                    template_path=template_path
                )
                
                # Stocker le générateur dans session_state
                st.session_state['output_generator'] = generator
                st.session_state['export_ready'] = True
                
                # Afficher les infos
                export_info = generator.get_export_info()
                
                st.success("✅ Export structure generated successfully!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Template Rows", export_info['template_rows'])
                with col2:
                    st.metric("Simulations", export_info['N'])
                with col3:
                    st.metric("Total Rows", f"{export_info['total_rows']:,}")
                
            except Exception as e:
                st.error(f"❌ Error generating export: {str(e)}")
                st.exception(e)
                st.session_state['export_ready'] = False
    
    st.markdown("---")
    
    # Section 2: Aperçu d'une simulation
    st.subheader("2️⃣ Preview a Simulation")
    
    if not st.session_state.get('export_ready', False):
        st.info("👆 Generate the export structure first to preview simulations.")
    else:
        col1, col2 = st.columns([1, 3])
        
        with col1:
            sim_number = st.number_input(
                "Simulation Number",
                min_value=1,
                max_value=N,
                value=1,
                step=1,
                help="Select which simulation to preview"
            )
        
        try:
            generator = st.session_state['output_generator']
            preview_df = generator.get_simulation_preview(sim_number)
            
            st.markdown(f"**Preview of Simulation #{sim_number}**")
            
            # Afficher le tableau avec des options de formatage
            st.dataframe(
                preview_df.style.format({
                    col: "{:.6f}" for col in preview_df.columns 
                    if col not in ['Simulation', 'CLASS', 'MEASURE', 'OS_TERM']
                }),
                use_container_width=True,
                height=400
            )
            
            # Statistiques rapides sur cette simulation
            st.markdown("**Statistics for this simulation:**")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Rows", preview_df.shape[0])
            
            with col2:
                st.metric("Time Columns", T + 1)
            
            with col3:
                numeric_cols = preview_df.select_dtypes(include=['number']).columns
                non_zero = (preview_df[numeric_cols] != 0).sum().sum()
                st.metric("Non-Zero Values", f"{non_zero:,}")
            
            with col4:
                # Compte des lignes par CLASS
                unique_classes = preview_df['CLASS'].nunique()
                st.metric("Asset Classes", unique_classes)
            
            # Détails par classe d'actifs
            with st.expander("📊 View by Asset Class"):
                for asset_class in preview_df['CLASS'].unique():
                    class_df = preview_df[preview_df['CLASS'] == asset_class]
                    st.markdown(f"**{asset_class}** ({len(class_df)} rows)")
                    st.dataframe(class_df, use_container_width=True)
        
        except Exception as e:
            st.error(f"❌ Error generating preview: {str(e)}")
            st.exception(e)
    
    st.markdown("---")
    
    # Section 3: Export final
    st.subheader("3️⃣ Export to CSV")
    
    if not st.session_state.get('export_ready', False):
        st.info("👆 Generate the export structure first to enable CSV export.")
    else:
        st.info("""
        This will generate a consolidated CSV file containing all simulations in ESG format.
        The file can be quite large depending on the number of simulations.
        """)
        
        col1, col2 = st.columns([2, 2])
        
        with col1:
            output_filename = st.text_input(
                "Output Filename",
                value=f"esg_scenarios_N{N}_T{T}_seed{seed}.csv",
                help="Name of the CSV file to export"
            )
        
        with col2:
            st.write("")  # Espacement
            st.write("")  # Espacement
            
            if st.button("💾 Generate & Download CSV", type="primary", use_container_width=True):
                with st.spinner("Generating consolidated CSV... This may take a moment..."):
                    try:
                        generator = st.session_state['output_generator']
                        
                        # Générer le DataFrame consolidé
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        status_text.text("Generating all simulations...")
                        consolidated_df = generator.generate_all_simulations()
                        progress_bar.progress(50)
                        
                        status_text.text("Converting to CSV...")
                        csv_data = consolidated_df.to_csv(index=False).encode('utf-8')
                        progress_bar.progress(100)
                        
                        status_text.empty()
                        progress_bar.empty()
                        
                        # Bouton de téléchargement
                        st.download_button(
                            label="📥 Download CSV File",
                            data=csv_data,
                            file_name=output_filename,
                            mime='text/csv',
                            type="primary",
                            use_container_width=True
                        )
                        
                        st.success(f"✅ Export ready! Click the button above to download.")
                        
                        # Statistiques de l'export
                        file_size_mb = len(csv_data) / (1024 * 1024)
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total Rows", f"{len(consolidated_df):,}")
                        
                        with col2:
                            st.metric("Total Columns", consolidated_df.shape[1])
                        
                        with col3:
                            st.metric("File Size", f"{file_size_mb:.2f} MB")
                        
                        with col4:
                            st.metric("Format", "CSV")
                        
                        # Aperçu du fichier consolidé
                        with st.expander("👁️ Preview Consolidated File (first 100 rows)"):
                            st.dataframe(
                                consolidated_df.head(100).style.format({
                                    col: "{:.6f}" for col in consolidated_df.columns 
                                    if col not in ['Simulation', 'CLASS', 'MEASURE', 'OS_TERM']
                                }),
                                use_container_width=True
                            )
                        
                        # Distribution par simulation
                        with st.expander("📊 View Distribution by Simulation"):
                            sim_counts = consolidated_df['Simulation'].value_counts().sort_index()
                            st.bar_chart(sim_counts)
                            st.caption("Each simulation should have the same number of rows")
                        
                    except Exception as e:
                        st.error(f"❌ Error generating CSV export: {str(e)}")
                        st.exception(e)
        
        # Note importante
        st.markdown("---")
        st.info("""
        **Note:** The exported CSV file follows the ESG format with columns:
        - `Simulation`: Simulation number (1 to N)
        - `CLASS`: Asset class (EQUITIES, REAL_ESTATE, ZCB, VALN, etc.)
        - `MEASURE`: Measure type (RET_IDX, PRICE, DEF, RNY_PC, etc.)
        - `OS_TERM`: Outstanding term (maturity for ZCB, 0 otherwise)
        - `0` to `T`: Time steps (values for each year)
        """)


if __name__ == "__main__":
    # Pour tester le module indépendamment
    st.set_page_config(page_title="Export ESG", layout="wide")
    render_export_tab()