# templates.py

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from scipy.interpolate import CubicSpline
from scripts.class_model import ir_to_ZCB, ASSET_MODELS

class_list = list(ASSET_MODELS.keys())

class InputsTemplate:
    def __init__(self, asset_class):
        self.asset_class = asset_class

    def render(self, T, rfr):
        """ Method to display the input template (optimisé pour sidebar) """
        
        # Selection widget
        model_choice = st.selectbox(
            "Model:",
            ASSET_MODELS[self.asset_class],
            key=f"model_choice_{self.asset_class}"
        )

        # Initialisation of ZCB_flag
        ZCB_flag = False
        if model_choice == 'Vasicek':
            ZCB_flag = st.checkbox(
                "Use Risk Free Rates for ZC valuation", 
                value=False, 
                key=self.asset_class
            )

        if ZCB_flag:
            df = ir_to_ZCB(T, rfr)
            st.success("✅ Using RFR for calibration")
        else:
            # Data upload
            uploaded_file = st.file_uploader(
                "Upload calibration data:", 
                type=["csv", "xlsx"], 
                key=f"file_uploader_{self.asset_class}"
            )

            # Data reading
            df = pd.DataFrame()
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file, sep=',', header='infer', encoding='utf-8')
                st.success(f"✅ Loaded {len(df)} rows")
            else:
                st.info("⏳ Waiting for data upload...")
            
        return {'model_name': model_choice, 'calibration_df': df}


class TestsResultsTemplates:
    def __init__(self, asset_class, model_name):
        self.asset_class = asset_class
        self.model_name = model_name

    def display_calibrated_ir(self, rfr, spot):
        """Affiche la courbe des taux calibrée"""
        time_idx = spot.index
        observed_rfr = pd.Series(rfr(time_idx), index=time_idx)

        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            y=observed_rfr, 
            x=time_idx,
            mode='lines', 
            name="Market rates",
            line=dict(color='blue', width=2)
        ))
        fig1.add_trace(go.Scatter(
            y=spot, 
            x=time_idx,
            mode='lines', 
            name="Modeled rates",
            line=dict(color='red', width=2, dash='dash')
        ))
        fig1.update_layout(
            title=f'Replication of spot risk free rates with {self.model_name}',
            xaxis_title='Maturity (years)',
            yaxis_title='Interest rates',
            showlegend=True,
            hovermode='x unified'
        )
        st.plotly_chart(fig1, use_container_width=True)

    def render_interest_rates(self, martingality_dict):
        """Affiche les résultats des tests pour les modèles de taux"""
        deflator_df = martingality_dict["Deflator"]
        zc_price_df = martingality_dict["ZC_Price"]

        # Graphique du déflateur
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=deflator_df.index,
            y=deflator_df["Expected"],
            mode='lines',
            name="Target",
            line=dict(color='green', width=2)
        ))
        fig1.add_trace(go.Scatter(
            x=deflator_df.index,
            y=deflator_df["Results"],
            mode='lines',
            name="Results",
            line=dict(color='blue', width=2)
        ))
        fig1.add_trace(go.Scatter(
            x=deflator_df.index.tolist() + deflator_df.index[::-1].tolist(),
            y=deflator_df["Lower Confidence Interval"].tolist() + 
              deflator_df["Upper Confidence Interval"][::-1].tolist(),
            fill='toself',
            fillcolor='rgba(0, 100, 250, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Confidence Interval'
        ))
        fig1.update_layout(
            title='Deflator Martingality Testing',
            xaxis_title='Time (years)',
            yaxis_title='Value',
            showlegend=True,
            hovermode='x unified'
        )
        
        # Graphique des prix ZC
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=zc_price_df.index,
            y=zc_price_df["Expected"],
            mode='lines',
            name="Target",
            line=dict(color='green', width=2)
        ))
        fig2.add_trace(go.Scatter(
            x=zc_price_df.index,
            y=zc_price_df["Results"],
            mode='lines',
            name="Results",
            line=dict(color='blue', width=2)
        ))
        fig2.add_trace(go.Scatter(
            x=zc_price_df.index.tolist() + zc_price_df.index[::-1].tolist(),
            y=zc_price_df["Lower Confidence Interval"].tolist() + 
              zc_price_df["Upper Confidence Interval"][::-1].tolist(),
            fill='toself',
            fillcolor='rgba(0, 100, 250, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Confidence Interval'
        ))
        fig2.update_layout(
            title='Zero-Coupon Rates Martingality Testing',
            xaxis_title='Time (years)',
            yaxis_title='Interest Rate',
            showlegend=True,
            hovermode='x unified'
        )

        # Affichage
        st.plotly_chart(fig1, use_container_width=True)
        with st.expander("📊 View Deflator Data"):
            st.dataframe(deflator_df, use_container_width=True)

        st.plotly_chart(fig2, use_container_width=True)
        with st.expander("📊 View Zero-Coupon Data"):
            st.dataframe(zc_price_df, use_container_width=True)

    def render_index(self, df, martingality_df):
        """Affiche les résultats des simulations pour les modèles d'indices"""
        
        # Graphique des simulations (seulement les 50 premières pour la lisibilité)
        fig1 = go.Figure()
        num_to_plot = min(50, len(df.index))
        for idx in df.index[:num_to_plot]:
            fig1.add_trace(go.Scatter(
                x=df.columns, 
                y=df.loc[idx], 
                mode='lines', 
                name=str(idx),
                line=dict(width=0.5),
                opacity=0.6,
                showlegend=False
            ))
        
        fig1.update_layout(
            title=f'Simulations of {self.model_name} model (showing {num_to_plot}/{len(df)} paths)',
            xaxis_title='Time (years)',
            yaxis_title='Index Value',
            showlegend=False,
            hovermode='x unified'
        )
        
        # Graphique de martingalité
        fig2 = go.Figure()
        col_names_martingality = [
            "Expected",
            "Results",
            "Lower Confidence Interval",
            "Upper Confidence Interval"
        ]
        
        colors = {
            "Expected": "green",
            "Results": "blue",
            "Lower Confidence Interval": "lightblue",
            "Upper Confidence Interval": "lightblue"
        }
        
        for name in col_names_martingality:
            fig2.add_trace(go.Scatter(
                x=martingality_df.index, 
                y=martingality_df[name], 
                mode='lines', 
                name=name,
                line=dict(
                    color=colors.get(name, 'gray'),
                    width=2 if name in ["Expected", "Results"] else 1,
                    dash='dash' if 'Confidence' in name else 'solid'
                )
            ))
        
        fig2.update_layout(
            title='Martingality Testing',
            xaxis_title='Time (years)',
            yaxis_title='Normalized Value',
            showlegend=True,
            hovermode='x unified'
        )
        
        # Affichage
        st.plotly_chart(fig1, use_container_width=True)
        st.plotly_chart(fig2, use_container_width=True)
        
        with st.expander("📊 View Martingality Test Data"):
            st.dataframe(martingality_df, use_container_width=True)


class RiskFreeRates:
    def __init__(self, period):
        self.period = period

    def render(self):
        """Upload et affichage des taux sans risque"""
        # Data upload
        uploaded_file = st.file_uploader(
            "📁 Upload Risk Free Rates data:", 
            type=["csv", "xlsx"], 
            key="RFR",
            help="Upload a CSV file with interest rates by maturity"
        )

        # Data reading
        rfr_dict = {}
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file, sep=',', header='infer', encoding='utf-8', index_col=0)
            
            # Création du graphique
            fig = go.Figure()
            for name in df.columns:
                rfr_cs = CubicSpline(df.index.to_list(), df[name].tolist())
                rfr_dict[name] = rfr_cs
                fig.add_trace(go.Scatter(
                    x=df.index, 
                    y=rfr_cs(df.index), 
                    mode='lines+markers', 
                    name=name,
                    line=dict(width=2)
                ))

            fig.update_layout(
                title=f"Interest Rates Curves at {self.period}",
                xaxis_title='Time (years)',
                yaxis_title='Rates',
                showlegend=True,
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
            st.success(f"✅ Loaded risk-free rates with {len(df.columns)} scenario(s)")
            
        return rfr_dict