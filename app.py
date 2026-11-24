"""
app.py
Interactive Streamlit Web Application for Wire EDM Performance Prediction
and Parametric Uncertainty Modeling.
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import scipy.stats as stats

st.set_page_config(
    page_title="Wire EDM ML Framework",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')
FIG_DIR = os.path.join(BASE_DIR, 'results', 'figures')
TABLES_DIR = os.path.join(BASE_DIR, 'results', 'tables')

FEATURE_COLS = ['Ton', 'Toff', 'IP', 'SV', 'WF', 'WT']


@st.cache_resource
def load_models_and_scalers():
    """Load serialized models and preprocessing scaler."""
    scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.joblib'))
    models = {
        'Linear Regression (LR)': {
            'cr': joblib.load(os.path.join(MODELS_DIR, 'lr_cr.joblib')),
            'sr': joblib.load(os.path.join(MODELS_DIR, 'lr_sr.joblib')),
            'scaled': False
        },
        'Decision Tree (DT)': {
            'cr': joblib.load(os.path.join(MODELS_DIR, 'dt_cr.joblib')),
            'sr': joblib.load(os.path.join(MODELS_DIR, 'dt_sr.joblib')),
            'scaled': False
        },
        'Support Vector Machine (SVM)': {
            'cr': joblib.load(os.path.join(MODELS_DIR, 'svr_cr.joblib')),
            'sr': joblib.load(os.path.join(MODELS_DIR, 'svr_sr.joblib')),
            'scaled': True
        }
    }
    return scaler, models


def main():
    st.title("⚡ Wire EDM Performance Modeling Under Parametric Uncertainty")
    st.caption("Machine Learning-Based Predictive Framework | National Institute of Technology Jamshedpur")

    # Check if models exist
    if not os.path.exists(os.path.join(MODELS_DIR, 'svr_cr.joblib')):
        st.warning("Models not found. Please execute `python run_pipeline.py` in the project root first!")
        return

    scaler, models = load_models_and_scalers()

    # Sidebar: Parameter Inputs
    st.sidebar.header("🛠️ Input Process Parameters")
    
    ton = st.sidebar.slider("Pulse-On Time (Ton) [µs]", 0.40, 1.40, 0.90, step=0.02)
    toff = st.sidebar.slider("Pulse-Off Time (Toff) [µs]", 14.0, 46.0, 30.0, step=0.5)
    ip = st.sidebar.slider("Peak Current (IP) [A]", 70.0, 210.0, 140.0, step=1.0)
    sv = st.sidebar.slider("Servo Voltage (SV) [V]", 16.0, 80.0, 48.0, step=1.0)
    wf = st.sidebar.slider("Wire Feed Rate (WF) [m/min]", 2.0, 12.0, 7.0, step=0.2)
    wt = st.sidebar.slider("Wire Tension (WT) [g]", 450.0, 1600.0, 1025.0, step=25.0)

    st.sidebar.markdown("---")
    selected_model_name = st.sidebar.selectbox(
        "Select Machine Learning Model",
        list(models.keys()),
        index=2 # SVR default (best performer)
    )

    uncertainty_pct = st.sidebar.select_slider(
        "Parametric Uncertainty (Stochastic Fluctuation)",
        options=[1, 3, 5],
        value=3,
        format_func=lambda x: f"±{x}%"
    )

    # Prepare input array
    input_vector = np.array([[ton, toff, ip, sv, wf, wt]])
    model_obj = models[selected_model_name]

    if model_obj['scaled']:
        input_proc = scaler.transform(input_vector)
    else:
        input_proc = input_vector

    pred_cr = float(model_obj['cr'].predict(input_proc)[0])
    pred_sr = float(model_obj['sr'].predict(input_proc)[0])

    # Monte Carlo simulation for selected operating point
    rng = np.random.RandomState(42)
    n_sims = 3000
    std_vector = (uncertainty_pct / 100.0) * input_vector[0]
    sim_inputs = rng.normal(loc=input_vector[0], scale=std_vector, size=(n_sims, 6))

    if model_obj['scaled']:
        sim_proc = scaler.transform(sim_inputs)
    else:
        sim_proc = sim_inputs

    cr_dist = model_obj['cr'].predict(sim_proc)
    sr_dist = model_obj['sr'].predict(sim_proc)

    cr_p2_5, cr_p97_5 = np.percentile(cr_dist, [2.5, 97.5])
    sr_p2_5, sr_p97_5 = np.percentile(sr_dist, [2.5, 97.5])

    # Layout: Top Metrics Cards
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Cutting Rate (CR)")
        st.metric(
            label="Estimated CR",
            value=f"{pred_cr:.3f} mm/min"
        )
        st.info(f"**95% Stochastic Bound (±{uncertainty_pct}% Uncertainty):** [{cr_p2_5:.3f} to {cr_p97_5:.3f}] mm/min")

    with col2:
        st.subheader("🎯 Surface Roughness (SR)")
        st.metric(
            label="Estimated SR (Ra)",
            value=f"{pred_sr:.3f} µm"
        )
        st.info(f"**95% Stochastic Bound (±{uncertainty_pct}% Uncertainty):** [{sr_p2_5:.3f} to {sr_p97_5:.3f}] µm")

    st.markdown("---")

    # Tabs for in-depth insights
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Probabilistic Behavior",
        "⚖️ Model Performance & Benchmarks",
        "🌪️ Sensitivity Analysis",
        "🎯 Regression Diagnostics",
        "📋 Dataset & Parameter Distribution"
    ])

    with tab1:
        st.markdown(f"### Output Distributions at Operating Point under ±{uncertainty_pct}% Parametric Uncertainty")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        kde_cr = stats.gaussian_kde(cr_dist)
        x_cr = np.linspace(cr_dist.min(), cr_dist.max(), 300)
        y_cr = kde_cr(x_cr)
        axes[0].plot(x_cr, y_cr, color='#2b5c8f', linewidth=2.2)
        axes[0].fill_between(x_cr, 0, y_cr, color='#2b5c8f', alpha=0.35)
        axes[0].axvline(pred_cr, color='red', linestyle='--', label=f'Nominal: {pred_cr:.3f}')
        axes[0].axvspan(cr_p2_5, cr_p97_5, color='#2b5c8f', alpha=0.15, label='95% Interval')
        axes[0].set_title('Cutting Rate Distribution (CR)', fontweight='bold')
        axes[0].set_xlabel('CR (mm/min)')
        axes[0].grid(True, linestyle=':', alpha=0.6)
        axes[0].legend()

        kde_sr = stats.gaussian_kde(sr_dist)
        x_sr = np.linspace(sr_dist.min(), sr_dist.max(), 300)
        y_sr = kde_sr(x_sr)
        axes[1].plot(x_sr, y_sr, color='#2ca02c', linewidth=2.2)
        axes[1].fill_between(x_sr, 0, y_sr, color='#2ca02c', alpha=0.35)
        axes[1].axvline(pred_sr, color='red', linestyle='--', label=f'Nominal: {pred_sr:.3f}')
        axes[1].axvspan(sr_p2_5, sr_p97_5, color='#2ca02c', alpha=0.15, label='95% Interval')
        axes[1].set_title('Surface Roughness Distribution (SR)', fontweight='bold')
        axes[1].set_xlabel('SR (µm Ra)')
        axes[1].grid(True, linestyle=':', alpha=0.6)
        axes[1].legend()

        st.pyplot(fig)

        # Precomputed stochastic bounds table
        bounds_file = os.path.join(TABLES_DIR, 'stochastic_bounds_summary.csv')
        if os.path.exists(bounds_file):
            st.markdown("#### Overall Baseline Stochastic Bounds (Nominal Mean Conditions)")
            b_df = pd.read_csv(bounds_file)
            st.dataframe(b_df, use_container_width=True)

    with tab2:
        st.markdown("### Comparative Performance of ML Regressors")
        metrics_file = os.path.join(TABLES_DIR, 'model_performance_metrics.csv')
        if os.path.exists(metrics_file):
            m_df = pd.read_csv(metrics_file)
            st.dataframe(m_df, use_container_width=True)

        fig2_img = os.path.join(FIG_DIR, 'fig2_model_performance_comparison.png')
        if os.path.exists(fig2_img):
            st.image(fig2_img, caption="Figure 2: Comparison of Model Performance Metrics for WEDM", use_container_width=True)

    with tab3:
        st.markdown("### Monte Carlo Sensitivity Ranking (10,000 MCS Iterations)")
        sens_file = os.path.join(TABLES_DIR, 'sensitivity_analysis_summary.csv')
        if os.path.exists(sens_file):
            s_df = pd.read_csv(sens_file)
            st.dataframe(s_df, use_container_width=True)

        fig4_img = os.path.join(FIG_DIR, 'fig4_sensitivity_analysis.png')
        if os.path.exists(fig4_img):
            st.image(fig4_img, caption="Figure 4: Sensitivity Analysis of WEDM Parameters for CR and SR", use_container_width=True)

    with tab4:
        st.markdown("### Support Vector Regression (SVR) Actual vs. Predicted Diagnostics")
        fig3_img = os.path.join(FIG_DIR, 'fig3_svr_actual_vs_predicted.png')
        if os.path.exists(fig3_img):
            st.image(fig3_img, caption="Figure 3: Predicted vs. Actual Values for CR and SR Using SVR", use_container_width=True)

    with tab5:
        st.markdown("### Training Dataset & Parameter Ranges")
        fig1_img = os.path.join(FIG_DIR, 'fig1_parameter_distributions.png')
        if os.path.exists(fig1_img):
            st.image(fig1_img, caption="Figure 1: Distribution of Process Parameters in the Training Dataset", use_container_width=True)

        train_file = os.path.join(DATA_DIR, 'train_dataset.csv')
        if os.path.exists(train_file):
            train_df = pd.read_csv(train_file)
            st.write(f"Sample Records (Total Training Samples: {len(train_df)}):")
            st.dataframe(train_df.head(20), use_container_width=True)


if __name__ == '__main__':
    main()
