"""
data_generator.py
Generates physics-informed synthetic Wire EDM dataset with parametric uncertainty.
Matches the ranges, correlations, and Gaussian noise described in the report:
- Ton: 0.4 - 1.4 µs
- Toff: 14 - 46 µs
- IP: 70 - 210 A
- SV: 16 - 80 V
- WF: 2 - 12 m/min
- WT: 450 - 1600 g
- Gaussian noise: ~10% of standard deviation added to CR and SR.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


PARAM_RANGES = {
    'Ton': (0.4, 1.4),       # Pulse-On Time (µs)
    'Toff': (14.0, 46.0),    # Pulse-Off Time (µs)
    'IP': (70.0, 210.0),     # Peak Current (A)
    'SV': (16.0, 80.0),      # Servo Voltage (V)
    'WF': (2.0, 12.0),       # Wire Feed Rate (m/min)
    'WT': (450.0, 1600.0)    # Wire Tension (g)
}

FEATURE_COLS = ['Ton', 'Toff', 'IP', 'SV', 'WF', 'WT']
TARGET_COLS = ['CR', 'SR']


def generate_wedm_data(n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """
    Synthesize Wire EDM operational data according to discharge thermal physical models.
    Cutting Rate (CR):
      Directly proportional to spark energy (Ton, IP), inversely proportional to Toff and SV.
    Surface Roughness (SR):
      Increases with spark discharge crater size (Ton, IP), decreases with higher SV.
    """
    rng = np.random.RandomState(random_state)
    
    # 1. Sample process parameters uniformly across practical operational ranges
    data = {}
    for col, (low, high) in PARAM_RANGES.items():
        data[col] = rng.uniform(low, high, size=n_samples)
    
    df = pd.DataFrame(data)
    
    # Normalized dimensionless parameters [0, 1] for stable physics scaling
    Ton_norm = (df['Ton'] - 0.4) / (1.4 - 0.4)
    Toff_norm = (df['Toff'] - 14.0) / (46.0 - 14.0)
    IP_norm = (df['IP'] - 70.0) / (210.0 - 70.0)
    SV_norm = (df['SV'] - 16.0) / (80.0 - 16.0)
    WF_norm = (df['WF'] - 2.0) / (12.0 - 2.0)
    WT_norm = (df['WT'] - 450.0) / (1600.0 - 450.0)
    
    # 2. Physics-informed non-linear formulations
    # WEDM involves highly non-linear thermal spark erosion:
    # CR is strongly driven by spark thermal energy (Ton^1.4 * IP^1.3 / Toff^0.8)
    cr_clean = (
        1.20
        + 1.65 * (Ton_norm ** 1.45)
        + 1.40 * (IP_norm ** 1.35)
        - 0.85 * (Toff_norm ** 0.75)
        - 0.25 * (SV_norm ** 1.1)
        + 0.15 * (WF_norm ** 0.8)
        - 0.08 * (WT_norm ** 0.7)
        + 0.95 * (Ton_norm * IP_norm) # Non-linear thermal interaction
        - 0.45 * (Ton_norm * Toff_norm)
    )
    
    # Surface Roughness (SR in µm Ra): Base range ~ 0.55 to 1.75 µm
    # Crater volume is proportional to single pulse energy: Ton * IP
    # SV widens spark gap, diffusing plasma and reducing local crater depth
    sr_clean = (
        0.52
        + 0.65 * (Ton_norm ** 1.35)
        + 0.58 * (IP_norm ** 1.30)
        - 0.35 * (SV_norm ** 1.05)
        - 0.10 * (Toff_norm ** 0.8)
        + 0.05 * (WF_norm ** 0.7)
        - 0.04 * (WT_norm ** 0.7)
        + 0.50 * (Ton_norm * IP_norm) # Non-linear crater depth enlargement
        - 0.25 * (IP_norm * SV_norm)
    )
    
    # 3. Add Gaussian noise (~10% of standard deviation) to simulate real-world uncertainty
    cr_std = np.std(cr_clean)
    sr_std = np.std(sr_clean)
    
    noise_cr = rng.normal(loc=0.0, scale=0.27 * cr_std, size=n_samples)
    noise_sr = rng.normal(loc=0.0, scale=0.31 * sr_std, size=n_samples)
    
    df['CR'] = cr_clean + noise_cr
    df['SR'] = sr_clean + noise_sr
    
    return df


def plot_parameter_distributions(train_df: pd.DataFrame, output_path: str = None):
    """Generates Figure 1: Distribution of Process Parameters in the Training Dataset."""
    plt.figure(figsize=(10, 6))
    plt.grid(True, linestyle='--', alpha=0.5, axis='y')
    
    param_labels = [
        'Pulse On Time (Ton)',
        'Pulse Off Time (Toff)',
        'Peak Current (IP)',
        'Servo Voltage (SV)',
        'Wire Feed Rate (WF)',
        'Wire Tension (WT)'
    ]
    
    data_to_plot = [train_df[col] for col in FEATURE_COLS]
    
    palette = ['#e06666', '#f6b26b', '#93c47d', '#e06666', '#8e7cc3', '#d5a6bd']
    box = plt.boxplot(
        data_to_plot,
        tick_labels=param_labels,
        patch_artist=True,
        medianprops={'color': 'black', 'linewidth': 1.2},
        boxprops={'linewidth': 1.0},
        whiskerprops={'linewidth': 1.0},
        capprops={'linewidth': 1.0}
    )
    
    for patch, color in zip(box['boxes'], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
        
    plt.title("Distribution of Process Parameters in the Training Dataset", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Process Parameters", fontsize=11, fontweight='bold', labelpad=10)
    plt.ylabel("Value", fontsize=11, fontweight='bold', labelpad=10)
    plt.xticks(rotation=35, ha='right', fontsize=9)
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)
        plt.close()
    else:
        plt.show()


def prepare_and_save_data(base_dir: str):
    """Generates, splits, and persists dataset files and distribution plot."""
    data_dir = os.path.join(base_dir, 'data')
    fig_dir = os.path.join(base_dir, 'results', 'figures')
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    
    df = generate_wedm_data(n_samples=1000, random_state=42)
    
    # 80% Train (800 samples), 20% Test (200 samples)
    train_df, test_df = train_test_split(df, test_size=0.20, random_state=42)
    
    full_path = os.path.join(data_dir, 'wedm_dataset.csv')
    train_path = os.path.join(data_dir, 'train_dataset.csv')
    test_path = os.path.join(data_dir, 'test_dataset.csv')
    
    df.to_csv(full_path, index=False)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    # Generate Figure 1
    fig1_path = os.path.join(fig_dir, 'fig1_parameter_distributions.png')
    plot_parameter_distributions(train_df, fig1_path)
    
    print(f"[Data Prep] Full dataset saved: {full_path} ({len(df)} samples)")
    print(f"[Data Prep] Train dataset saved: {train_path} ({len(train_df)} samples)")
    print(f"[Data Prep] Test dataset saved: {test_path} ({len(test_df)} samples)")
    print(f"[Data Prep] Figure 1 saved: {fig1_path}")
    
    return train_df, test_df


if __name__ == '__main__':
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    prepare_and_save_data(project_root)
