"""
models.py
Defines, trains, evaluates, and visualizes machine learning models:
1. Linear Regression (LR)
2. Decision Tree Regressor (DT)
3. Support Vector Regression (SVR with RBF kernel)

Generates:
- Figure 2: Comparison of Model Performance Metrics for WEDM
- Figure 3: Predicted vs Actual Values for CR and SR using SVR
- Serialized model artifacts in models/
- Performance metrics CSV in results/tables/
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


FEATURE_COLS = ['Ton', 'Toff', 'IP', 'SV', 'WF', 'WT']
TARGET_COLS = ['CR', 'SR']


def train_and_evaluate_models(train_df: pd.DataFrame, test_df: pd.DataFrame, base_dir: str):
    """
    Trains LR, DT, and SVR models on both targets (CR and SR), evaluates performance,
    generates Figures 2 & 3, and saves models and metric tables.
    """
    models_dir = os.path.join(base_dir, 'models')
    fig_dir = os.path.join(base_dir, 'results', 'figures')
    table_dir = os.path.join(base_dir, 'results', 'tables')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(table_dir, exist_ok=True)

    X_train = train_df[FEATURE_COLS].values
    X_test = test_df[FEATURE_COLS].values
    
    # Scaler for Support Vector Regression and distance-based operations
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.joblib'))

    model_definitions = {
        'Linear Regression (LR)': {
            'CR': LinearRegression(),
            'SR': LinearRegression(),
            'use_scaled': False
        },
        'Decision Tree (DT)': {
            'CR': DecisionTreeRegressor(max_depth=5, min_samples_split=12, min_samples_leaf=6, random_state=42),
            'SR': DecisionTreeRegressor(max_depth=5, min_samples_split=12, min_samples_leaf=6, random_state=42),
            'use_scaled': False
        },
        'Support Vector Machine (SVM)': {
            'CR': SVR(kernel='rbf', C=15.0, epsilon=0.15, gamma=0.15),
            'SR': SVR(kernel='rbf', C=12.0, epsilon=0.10, gamma=0.15),
            'use_scaled': True
        }
    }

    results = []
    predictions = {}

    for model_name, info in model_definitions.items():
        use_scaled = info['use_scaled']
        X_tr = X_train_scaled if use_scaled else X_train
        X_te = X_test_scaled if use_scaled else X_test

        predictions[model_name] = {}

        for target in TARGET_COLS:
            y_train = train_df[target].values
            y_test = test_df[target].values
            
            reg = info[target]
            reg.fit(X_tr, y_train)
            
            # Predict
            y_pred = reg.predict(X_te)
            predictions[model_name][target] = y_pred

            # Metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)

            results.append({
                'Model': model_name,
                'Target': target,
                'RMSE': round(rmse, 4),
                'R2': round(r2, 4),
                'MAE': round(mae, 4)
            })

            # Save model artifact
            prefix = 'lr' if 'Linear' in model_name else ('dt' if 'Decision' in model_name else 'svr')
            model_path = os.path.join(models_dir, f'{prefix}_{target.lower()}.joblib')
            joblib.dump(reg, model_path)

    results_df = pd.DataFrame(results)
    metrics_path = os.path.join(table_dir, 'model_performance_metrics.csv')
    results_df.to_csv(metrics_path, index=False)
    print(f"[Model Training] Evaluation Metrics saved to: {metrics_path}")

    # Generate Figure 2: Model Performance Comparison Bar Chart
    plot_model_comparison(results_df, os.path.join(fig_dir, 'fig2_model_performance_comparison.png'))
    
    # Generate Figure 3: SVR Actual vs Predicted Plot
    plot_svr_actual_vs_predicted(
        y_test_cr=test_df['CR'].values,
        y_pred_cr=predictions['Support Vector Machine (SVM)']['CR'],
        y_test_sr=test_df['SR'].values,
        y_pred_sr=predictions['Support Vector Machine (SVM)']['SR'],
        output_path=os.path.join(fig_dir, 'fig3_svr_actual_vs_predicted.png')
    )

    return results_df, predictions


def plot_model_comparison(results_df: pd.DataFrame, output_path: str):
    """
    Generates Figure 2: Grouped bar chart comparing RMSE, R², MAE across LR, DT, and SVM.
    Averages metrics across both targets (CR and SR) to showcase overall model efficacy.
    """
    avg_df = results_df.groupby('Model')[['RMSE', 'R2', 'MAE']].mean().reset_index()
    
    # Model order matching paper
    model_order = ['Linear Regression (LR)', 'Decision Tree (DT)', 'Support Vector Machine (SVM)']
    avg_df['Model'] = pd.Categorical(avg_df['Model'], categories=model_order, ordered=True)
    avg_df = avg_df.sort_values('Model')

    metrics = ['RMSE', 'R2', 'MAE']
    x = np.arange(len(metrics))
    width = 0.22

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#8ecae6', '#ffb703', '#90be6d']

    for i, (_, row) in enumerate(avg_df.iterrows()):
        vals = [row['RMSE'], row['R2'], row['MAE']]
        label = row['Model'].replace(' (LR)', '').replace(' (DT)', '').replace(' (SVM)', 'SVM')
        rects = ax.bar(x + (i - 1) * width, vals, width, label=label, color=colors[i], edgecolor='gray', alpha=0.9)
        # Add value labels on top of bars
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    ax.set_ylabel('Value', fontsize=11, fontweight='bold')
    ax.set_xlabel('Metric', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_title('Comparison of Model Performance Metrics for WEDM', fontsize=12, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(['RMSE', 'R²', 'MAE'], fontsize=10, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.legend(title='Models', frameon=True, fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Model Training] Figure 2 saved: {output_path}")


def plot_svr_actual_vs_predicted(y_test_cr, y_pred_cr, y_test_sr, y_pred_sr, output_path: str):
    """
    Generates Figure 3: SVR Actual vs Predicted for CR and SR side-by-side with 45-degree reference line.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Cutting Rate Plot (Blue points)
    axes[0].scatter(y_test_cr, y_pred_cr, color='blue', alpha=0.65, edgecolors='b', s=35)
    min_cr = min(y_test_cr.min(), y_pred_cr.min()) - 0.1
    max_cr = max(y_test_cr.max(), y_pred_cr.max()) + 0.1
    axes[0].plot([min_cr, max_cr], [min_cr, max_cr], 'r--', linewidth=2, label='Ideal 1:1')
    axes[0].set_title('SVR: Actual vs Predicted for Cutting Rate (CR)', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Actual CR (mm/min)', fontsize=10, fontweight='bold')
    axes[0].set_ylabel('Predicted CR (mm/min)', fontsize=10, fontweight='bold')
    axes[0].set_xlim(min_cr, max_cr)
    axes[0].set_ylim(min_cr, max_cr)
    axes[0].grid(True, linestyle=':', alpha=0.6)
    axes[0].legend(loc='upper left')

    # Surface Roughness Plot (Green points)
    axes[1].scatter(y_test_sr, y_pred_sr, color='forestgreen', alpha=0.65, edgecolors='g', s=35)
    min_sr = min(y_test_sr.min(), y_pred_sr.min()) - 0.05
    max_sr = max(y_test_sr.max(), y_pred_sr.max()) + 0.05
    axes[1].plot([min_sr, max_sr], [min_sr, max_sr], 'r--', linewidth=2, label='Ideal 1:1')
    axes[1].set_title('SVR: Actual vs Predicted for Surface Roughness (SR)', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Actual SR (µm)', fontsize=10, fontweight='bold')
    axes[1].set_ylabel('Predicted SR (µm)', fontsize=10, fontweight='bold')
    axes[1].set_xlim(min_sr, max_sr)
    axes[1].set_ylim(min_sr, max_sr)
    axes[1].grid(True, linestyle=':', alpha=0.6)
    axes[1].legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Model Training] Figure 3 saved: {output_path}")


if __name__ == '__main__':
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(project_root, 'data')
    train_df = pd.read_csv(os.path.join(data_dir, 'train_dataset.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'test_dataset.csv'))
    train_and_evaluate_models(train_df, test_df, project_root)
