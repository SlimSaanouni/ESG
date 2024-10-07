# templates.py

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from scipy.interpolate import CubicSpline
from scripts.class_model import ir_to_ZCB, ZCB_to_ir, ASSET_MODELS

class_list      = list(ASSET_MODELS.keys())

class InputsTemplate:
    def __init__(self, asset_class):
        self.asset_class = asset_class

    def render(self, T, rfr):
        """ Method to display the input template (with unique keys) """
        st.subheader(self.asset_class)

        # Selection widget
        model_choice = st.selectbox(
            "Selection of the model",
            ASSET_MODELS[self.asset_class],
            key=f"model_choice_{self.asset_class}"
        )

        # Initialisation of ZCB_flag
        ZCB_flag = False
        if model_choice == 'Vasicek':
            ZCB_flag = st.checkbox("Use of the Risk Free Rates for Zero-Coupon valuation", value = False, key = self.asset_class)

        if ZCB_flag:
            df = ir_to_ZCB(T, rfr)
        else:
            # Data upload
            uploaded_file = st.file_uploader("Uploading of data:", type=["csv", "xlsx"], key=f"file_uploader_{self.asset_class}")

            # Data reading
            df = pd.DataFrame()
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file, sep = ',', header = 'infer', encoding='utf-8')
            
        return {'model_name': model_choice, 'calibration_df': df}

class TestsResultsTemplates:
    def __init__(self, asset_class, model_name):
        self.asset_class = asset_class
        self.model_name  = model_name

    def display_calibrated_ir(self, rfr, spot):

        time_idx = spot.index
        observed_rfr = pd.Series(rfr(time_idx), index = time_idx)

        #modeled_rfr_df  = ZCB_to_ir(observed_rfr_df)

        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(y = observed_rfr, x = time_idx,
                                  mode = 'lines', name = "Market rates"))
        fig1.add_trace(go.Scatter(y = spot, x = time_idx,
                                  mode = 'lines', name = "Modeled rates"))
        fig1.update_layout(title = f'Replication of the spot risk free rates with the {self.model_name} model',
                          xaxis_title = 'Maturity (years)',
                          yaxis_title = 'Interest rates',
                          showlegend = True)
        st.plotly_chart(fig1)

    def render_interest_rates(self, df, martingality_df):
        """ Method to display the input template (with unique keys) """        
        
        fig1 = go.Figure()
        for idx in df.index:
            fig1.add_trace(go.Scatter(x = df.columns, y = df.loc[idx], mode = 'lines', name = idx))
        fig1.update_layout(title = f'Simulations of {self.model_name} model',
                          xaxis_title = 'Time (years)',
                          yaxis_title = 'Index',
                          showlegend = False)
        
        fig2 = go.Figure()
        col_names_martingality = ["Results",
                                  "Lower Confidence Interval",
                                  "Upper Confidence Interval",
                                  "Expected"]
        for name in col_names_martingality:
            fig2.add_trace(go.Scatter(x = martingality_df.index, y = martingality_df[name], mode = 'lines', name = name))
        fig2.update_layout(title = 'Martingality tests',
                          xaxis_title = 'Time (years)',
                          yaxis_title = 'Index',
                          showlegend = True)
        st.plotly_chart(fig1)
        st.plotly_chart(fig2)

        st.write(martingality_df)
        # On affiche un plot de la moyenne, de l'espérance, et des upper / lower

    def render_equity(self, df, martingality_df):
        """ Method to display the input template (with unique keys) """
        fig1 = go.Figure()
        for idx in df.index:
            fig1.add_trace(go.Scatter(x = df.columns, y = df.loc[idx], mode = 'lines', name = idx))
        fig1.update_layout(title = f'Simulations of {self.model_name} model',
                          xaxis_title = 'Time (years)',
                          yaxis_title = 'Index',
                          showlegend = False)
        
        fig2 = go.Figure()
        col_names_martingality = ["Results",
                                  "Lower Confidence Interval",
                                  "Upper Confidence Interval",
                                  "Expected"]
        for name in col_names_martingality:
            fig2.add_trace(go.Scatter(x = martingality_df.index, y = martingality_df[name], mode = 'lines', name = name))
        fig2.update_layout(title = 'Martingality tests',
                          xaxis_title = 'Time (years)',
                          yaxis_title = 'Index',
                          showlegend = True)
        st.plotly_chart(fig1)
        st.plotly_chart(fig2)

        st.write(martingality_df)
        # On affiche un plot de la moyenne, de l'espérance, et des upper / lower

class RiskFreeRates:
    def __init__(self, period):
        self.period = period

    def render(self):
        # Data upload
        uploaded_file = st.file_uploader("Uploading of data:", type=["csv", "xlsx"], key= "RFR")

        # Data reading
        rfr_dict = {}
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file, sep = ',', header = 'infer', encoding='utf-8', index_col = 0)
            fig = go.Figure()
            for name in df.columns:
                rfr_cs = CubicSpline(df.index.to_list(), df[name].tolist())
                rfr_dict[name] = rfr_cs
                fig.add_trace(go.Scatter(x = df.index, y = rfr_cs(df.index), mode = 'lines', name = name))

            fig.update_layout(title = "Interest rates curves at " + self.period,
                              xaxis_title = 'Time (years)',
                              yaxis_title = 'Rates',
                              showlegend = True)
            st.plotly_chart(fig)
        return rfr_dict

    
    
