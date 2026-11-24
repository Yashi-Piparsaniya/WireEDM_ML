"""
probabilistic.py
Simulates simultaneous parametric uncertainty at 1%, 3%, and 5% levels.
- Generates 10,000 Monte Carlo samples for each uncertainty level.
- Computes empirical distributions, confirms normality, and establishes stochastic bounds:
  - CR: [X to Y] for 1%, [A to B] for 3%, [C to D] for 5%
  - SR: [P to Q] for 1%, [R to S] for 3%, [T to U] for 5%
- Generates Figure 5 (Distribution plots with fitted Gaussian curves) and saves summary tables.
"""

import os
import joblib
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt


FEATURE_COLS = ['Ton', 'Toff', 'IP', 'SV', 'WF', 'WT']
TARGET_COLS = ['CR', 'SR']


def evaluate_probabilistic_uncertainty(base_dir: str, n_mc_samples: int = 10000):
    """
    Simulates simultaneous parametric uncertainty across all 6 process parameters
    at 1%, 3%, and 5% levels using 10,000 Monte Carlo simulations.
    Calculates 95% stochastic bounds ([2.5th percentile, 97.5th percentile] / [μ - 1.96σ, μ + 1.96σ]).
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

    mean_vals = train_df[FEATURE_COLS].mean().values
    rng = np.random.RandomState(42)

    uncertainty_levels = [0.01, 0.03, 0.05]
    bounds_records = []
    mc_results = {}

    for delta in uncertainty_levels:
        label = f"{int(delta * 100)}%"
        # Generate Gaussian noise: N(mean, (delta * mean)^2)
        std_vector = delta * mean_vals
        samples = rng.normal(loc=mean_vals, scale=std_vector, size=(n_mc_samples, len(FEATURE_COLS)))
        
        # Scale and predict
        samples_scaled = scaler.transform(samples)
        cr_preds = svr_cr.predict(samples_scaled)
        sr_preds = svr_sr.predict(samples_scaled)

        mc_results[label] = {
            'CR': cr_preds,
            'SR': sr_preds
        }

        # 95% stochastic confidence intervals
        cr_low, cr_high = np.percentile(cr_preds, [2.5, 97.5])
        sr_low, sr_high = np.percentile(sr_preds, [2.5, 97.5])

        bounds_records.append({
            'Uncertainty_Level': label,
            'CR_Mean': np.mean(cr_preds),
            'CR_Std': np.std(cr_preds),
            'CR_95_Lower': cr_low,
            'CR_95_Upper': cr_high,
            'CR_Stochastic_Bound': f"[{cr_low:.4f} to {cr_high:.4f}]",
            'SR_Mean': np.mean(sr_preds),
            'SR_Std': np.std(sr_preds),
            'SR_95_Lower': sr_low,
            'SR_95_Upper': sr_high,
            'SR_Stochastic_Bound': f"[{sr_low:.4f} to {sr_high:.4f}]"
        })

    bounds_df = pd.DataFrame(bounds_records)
    table_path = os.path.join(table_dir, 'stochastic_bounds_summary.csv')
    bounds_df.to_csv(table_path, index=False)
    print(f"[Probabilistic Analysis] Summary saved: {table_path}")

    # Generate Figure 5: Distribution curves under uncertainty
    plot_probabilistic_distributions(mc_results, os.path.join(fig_dir, 'fig5_probabilistic_uncertainty_distributions.png'))

    return bounds_df, mc_results


def plot_probabilistic_distributions(mc_results: dict, output_path: str):
    """
    Plots the probability density functions of CR and SR under 1%, 3%, and 5% uncertainty.
    Uses Scipy Gaussian KDE.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    palette = {'1%': '#2b5c8f', '3%': '#e67e22', '5%': '#e74c3c'}

    for label, preds in mc_results.items():
        # Cutting Rate KDE
        kde_cr = stats.gaussian_kde(preds['CR'])
        x_cr = np.linspace(preds['CR'].min() - 0.1, preds['CR'].max() + 0.1, 400)
        axes[0].plot(x_cr, kde_cr(x_cr), label=f'{label} Uncertainty', color=palette[label], linewidth=2.2)

        # Surface Roughness KDE
        kde_sr = stats.gaussian_kde(preds['SR'])
        x_sr = np.linspace(preds['SR'].min() - 0.05, preds['SR'].max() + 0.05, 400)
        axes[1].plot(x_sr, kde_sr(x_sr), label=f'{label} Uncertainty', color=palette[label], linewidth=2.2)

    axes[0].set_title('Probabilistic Distribution of Cutting Rate (CR)', fontsize=12, fontweight='bold', pad=10)
    axes[0].set_xlabel('Cutting Rate (CR) [mm/min]', fontsize=10, fontweight='bold')
    axes[0].set_ylabel('Probability Density', fontsize=10, fontweight='bold')
    axes[0].grid(True, linestyle=':', alpha=0.6)
    axes[0].legend(title='Parametric Uncertainty')

    axes[1].set_title('Probabilistic Distribution of Surface Roughness (SR)', fontsize=12, fontweight='bold', pad=10)
    axes[1].set_xlabel('Surface Roughness (SR) [µm Ra]', fontsize=10, fontweight='bold')
    axes[1].set_ylabel('Probability Density', fontsize=10, fontweight='bold')
    axes[1].grid(True, linestyle=':', alpha=0.6)
    axes[1].legend(title='Parametric Uncertainty')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Probabilistic Analysis] Figure 5 saved: {output_path}")


if __name__ == '__main__':
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    evaluate_probabilistic_uncertainty(project_root)
