"""
sensitivity.py
Performs Monte Carlo Sensitivity Analysis on Wire EDM parameters:
- Evaluates individual parameter influence by introducing ±5% stochastic variation
  around nominal mean while keeping all other parameters constant.
- Runs 10,000 Monte Carlo samples per scenario using the trained SVR surrogate model.
- Determines sensitivity ranking for Cutting Rate (CR) and Surface Roughness (SR).
- Generates Figure 4 (Sensitivity Analysis bar chart) and saves summary tables.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


FEATURE_COLS = ['Ton', 'Toff', 'IP', 'SV', 'WF', 'WT']
TARGET_COLS = ['CR', 'SR']


def perform_sensitivity_analysis(base_dir: str, n_mc_samples: int = 10000, variation_pct: float = 0.05):
    """
    Monte Carlo Sensitivity Analysis:
    For each parameter, injects ±5% variation around mean, holds others constant,
    evaluates using SVR model, and computes output dispersion (Std Dev).
    """
    models_dir = os.path.join(base_dir, 'models')
    data_dir = os.path.join(base_dir, 'data')
    fig_dir = os.path.join(base_dir, 'results', 'figures')
    table_dir = os.path.join(base_dir, 'results', 'tables')
    
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(table_dir, exist_ok=True)

    train_df = pd.read_csv(os.path.join(data_dir, 'train_dataset.csv'))
    scaler = joblib.load(os.path.join(models_dir, 'scaler.joblib'))
    svr_cr = joblib.load(os.path.join(models_dir, 'svr_cr.joblib'))
    svr_sr = joblib.load(os.path.join(models_dir, 'svr_sr.joblib'))

    mean_vals = train_df[FEATURE_COLS].mean().to_dict()
    rng = np.random.RandomState(42)

    sensitivity_records = []

    for param in FEATURE_COLS:
        # Base matrix where all features are set to their mean
        mc_data = np.tile([mean_vals[p] for p in FEATURE_COLS], (n_mc_samples, 1))
        param_idx = FEATURE_COLS.index(param)

        # Inject ±5% stochastic variation (Uniform across [mean*(1-0.05), mean*(1+0.05)])
        p_mean = mean_vals[param]
        low = p_mean * (1.0 - variation_pct)
        high = p_mean * (1.0 + variation_pct)
        mc_data[:, param_idx] = rng.uniform(low, high, size=n_mc_samples)

        # Scale and predict
        mc_scaled = scaler.transform(mc_data)
        cr_preds = svr_cr.predict(mc_scaled)
        sr_preds = svr_sr.predict(mc_scaled)

        cr_std = np.std(cr_preds)
        cr_range = np.ptp(cr_preds)
        sr_std = np.std(sr_preds)
        sr_range = np.ptp(sr_preds)

        sensitivity_records.append({
            'Parameter': param,
            'CR_StdDev': cr_std,
            'CR_Range': cr_range,
            'SR_StdDev': sr_std,
            'SR_Range': sr_range
        })

    sens_df = pd.DataFrame(sensitivity_records)
    
    # Calculate relative percentage sensitivity contribution
    sens_df['CR_Sensitivity_Pct'] = (sens_df['CR_StdDev'] / sens_df['CR_StdDev'].sum()) * 100
    sens_df['SR_Sensitivity_Pct'] = (sens_df['SR_StdDev'] / sens_df['SR_StdDev'].sum()) * 100

    table_path = os.path.join(table_dir, 'sensitivity_analysis_summary.csv')
    sens_df.to_csv(table_path, index=False)
    print(f"[Sensitivity Analysis] Summary saved: {table_path}")

    # Generate Figure 4: Sensitivity Bar Chart
    plot_sensitivity_tornado(sens_df, os.path.join(fig_dir, 'fig4_sensitivity_analysis.png'))

    return sens_df


def plot_sensitivity_tornado(sens_df: pd.DataFrame, output_path: str):
    """
    Plots horizontal bar charts showing parameter sensitivity ranking for CR and SR.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    # Sort for Cutting Rate
    cr_sorted = sens_df.sort_values('CR_Sensitivity_Pct', ascending=True)
    axes[0].barh(cr_sorted['Parameter'], cr_sorted['CR_Sensitivity_Pct'], color='#1f77b4', alpha=0.85, edgecolor='black')
    axes[0].set_title('Parameter Sensitivity on Cutting Rate (CR)', fontsize=12, fontweight='bold', pad=10)
    axes[0].set_xlabel('Relative Sensitivity Contribution (%)', fontsize=10, fontweight='bold')
    axes[0].grid(axis='x', linestyle='--', alpha=0.6)
    for i, v in enumerate(cr_sorted['CR_Sensitivity_Pct']):
        axes[0].text(v + 0.5, i, f'{v:.1f}%', va='center', fontsize=9, fontweight='bold')

    # Sort for Surface Roughness
    sr_sorted = sens_df.sort_values('SR_Sensitivity_Pct', ascending=True)
    axes[1].barh(sr_sorted['Parameter'], sr_sorted['SR_Sensitivity_Pct'], color='#2ca02c', alpha=0.85, edgecolor='black')
    axes[1].set_title('Parameter Sensitivity on Surface Roughness (SR)', fontsize=12, fontweight='bold', pad=10)
    axes[1].set_xlabel('Relative Sensitivity Contribution (%)', fontsize=10, fontweight='bold')
    axes[1].grid(axis='x', linestyle='--', alpha=0.6)
    for i, v in enumerate(sr_sorted['SR_Sensitivity_Pct']):
        axes[1].text(v + 0.5, i, f'{v:.1f}%', va='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Sensitivity Analysis] Figure 4 saved: {output_path}")


if __name__ == '__main__':
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    perform_sensitivity_analysis(project_root)
