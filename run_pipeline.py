"""
run_pipeline.py
Master execution pipeline for the Wire EDM Machine Learning Project.
Executes data generation, model training & evaluation, sensitivity analysis,
and probabilistic uncertainty modeling sequentially.
"""

import os
import sys
import time

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_generator import prepare_and_save_data
from src.models import train_and_evaluate_models
from src.sensitivity import perform_sensitivity_analysis
from src.probabilistic import evaluate_probabilistic_uncertainty


def main():
    print("=" * 78)
    print(" WIRE ELECTRIC DISCHARGE MACHINING (WEDM) MACHINE LEARNING FRAMEWORK ")
    print(" Predictive Modeling & Probabilistic Uncertainty Characterization ")
    print(" NIT Jamshedpur Minor Project Implementation ")
    print("=" * 78)
    
    start_time = time.time()

    # Step 1: Data Preparation
    print("\n[Step 1/4] Generating Physics-Informed Synthetic Dataset & Figure 1...")
    train_df, test_df = prepare_and_save_data(PROJECT_ROOT)

    # Step 2: Model Training & Evaluation
    print("\n[Step 2/4] Training Models (Linear Regression, Decision Tree, SVR)...")
    results_df, predictions = train_and_evaluate_models(train_df, test_df, PROJECT_ROOT)

    # Step 3: Sensitivity Analysis
    print("\n[Step 3/4] Running Monte Carlo Sensitivity Analysis (10,000 MCS Iterations)...")
    sens_df = perform_sensitivity_analysis(PROJECT_ROOT, n_mc_samples=10000)

    # Step 4: Probabilistic Uncertainty & Stochastic Bounds
    print("\n[Step 4/4] Evaluating Probabilistic Behavior (1%, 3%, 5% Uncertainty)...")
    bounds_df, mc_results = evaluate_probabilistic_uncertainty(PROJECT_ROOT, n_mc_samples=10000)

    elapsed = time.time() - start_time
    print("\n" + "=" * 78)
    print(f" PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.2f} SECONDS ")
    print("=" * 78)

    # Display key report tables
    print("\n>>> MODEL PERFORMANCE EVALUATION (TEST SET):")
    print(results_df.to_string(index=False))

    print("\n>>> PARAMETER SENSITIVITY RANKING:")
    cr_rank = sens_df.sort_values('CR_Sensitivity_Pct', ascending=False)[['Parameter', 'CR_Sensitivity_Pct']]
    sr_rank = sens_df.sort_values('SR_Sensitivity_Pct', ascending=False)[['Parameter', 'SR_Sensitivity_Pct']]
    print("\nCutting Rate (CR) Influence:")
    print(cr_rank.to_string(index=False))
    print("\nSurface Roughness (SR) Influence:")
    print(sr_rank.to_string(index=False))

    print("\n>>> STOCHASTIC BOUNDS UNDER PARAMETRIC UNCERTAINTY:")
    print(bounds_df[['Uncertainty_Level', 'CR_Mean', 'CR_Stochastic_Bound', 'SR_Mean', 'SR_Stochastic_Bound']].to_string(index=False))

    print("\n>>> GENERATED FIGURES (Saved in results/figures/):")
    print("  1. fig1_parameter_distributions.png")
    print("  2. fig2_model_performance_comparison.png")
    print("  3. fig3_svr_actual_vs_predicted.png")
    print("  4. fig4_sensitivity_analysis.png")
    print("  5. fig5_probabilistic_uncertainty_distributions.png")
    print("\nAll assets are ready for review and presentation.")


if __name__ == '__main__':
    main()
