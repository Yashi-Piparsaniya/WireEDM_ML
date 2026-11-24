# Wire EDM Performance Metrics Modeling under Parametric Uncertainty

This project implements the research work described in:
**"Machine Learning-Based Predictive Modeling of Wire EDM Performance Metrics Under Parametric Uncertainty"**
*Department of Mechanical Engineering / Engineering and Computational Mechanics, National Institute of Technology (NIT) Jamshedpur*

## Overview
Wire Electric Discharge Machining (WEDM) is a non-traditional precision machining technique used extensively in aerospace, automotive, and tooling industries. Process parameters inherently experience stochastic variations during operation due to spark gap instability, power fluctuations, and thermal phenomena.

This machine learning framework models:
1. **Cutting Rate (CR)** [mm/min]
2. **Surface Roughness (SR)** [$\mu$m Ra]

Under varying operating process parameters:
- **Pulse-On Time ($T_{on}$)**: 0.4 – 1.4 $\mu$s
- **Pulse-Off Time ($T_{off}$)**: 14 – 46 $\mu$s
- **Peak Current ($IP$)**: 70 – 210 A
- **Servo Voltage ($SV$)**: 16 – 80 V
- **Wire Feed Rate ($WF$)**: 2 – 12 m/min
- **Wire Tension ($WT$)**: 450 – 1600 g

## Key Features
- **Physics-Informed Synthetic Data Generator**: Generates 1,000 samples incorporating 10% Gaussian operational variability.
- **Machine Learning Regressors**:
  - Linear Regression (LR)
  - Decision Tree Regressor (DT)
  - Support Vector Regression (SVR with RBF kernel)
- **Model Evaluation**: Comprehensive metrics ($R^2$, RMSE, MAE) and publication-quality diagnostic plots.
- **Monte Carlo Sensitivity Analysis**: Assesses parameter influence using 10,000 MCS iterations at $\pm 5\%$ stochastic variation.
- **Probabilistic Uncertainty Modeling**: Evaluates output dispersion under 1%, 3%, and 5% parametric uncertainty and computes stochastic bounds.
- **Interactive Streamlit Web Dashboard**: Real-time parameter tuning, probabilistic bounds calculation, and scenario simulations.

---

## Directory Structure
```
WireEDM_ML_Project/
├── app.py                      # Interactive Streamlit Web Application
├── run_pipeline.py             # End-to-end execution script
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── data/                       # Generated dataset (train, test, full)
├── models/                     # Serialized trained models & scalers
├── results/
│   ├── figures/                # Publication-quality charts (Fig 1 - Fig 5)
│   └── tables/                 # Metrics, Sensitivity, and Stochastic Bounds CSVs
└── src/
    ├── __init__.py
    ├── data_generator.py       # Physics-based data synthesis + noise
    ├── models.py               # Model training & pipeline definitions
    ├── sensitivity.py          # Monte Carlo Sensitivity Analysis
    └── probabilistic.py        # Probabilistic uncertainty & bounds estimation
```

---

## Installation & Setup

1. Open a terminal / command prompt in this directory:
   ```bash
   cd WireEDM_ML_Project
   ```

2. (Optional) Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### 1. Run the Entire End-to-End Pipeline
To generate data, train all 3 models, evaluate performance, run sensitivity analysis, calculate probabilistic uncertainty bounds, and generate all figures:
```bash
python run_pipeline.py
```

### 2. Launch the Interactive Web Dashboard
An interactive GUI built with Streamlit allows you to test custom inputs, observe predicted CR and SR, inspect stochastic uncertainty bands, and view interactive charts:
```bash
streamlit run app.py
```

---

## Results Summary

| Model | Target | RMSE | $R^2$ | MAE |
| :--- | :--- | :---: | :---: | :---: |
| **Linear Regression** | Cutting Rate (CR) | ~0.45 | ~0.89 | ~0.31 |
| | Surface Roughness (SR) | ~0.32 | ~0.85 | ~0.28 |
| **Decision Tree** | Cutting Rate (CR) | ~0.41 | ~0.91 | ~0.29 |
| | Surface Roughness (SR) | ~0.31 | ~0.87 | ~0.27 |
| **Support Vector Machine (SVR)** | **Cutting Rate (CR)** | **~0.38** | **~0.93** | **~0.27** |
| | **Surface Roughness (SR)** | **~0.29** | **~0.90** | **~0.25** |

- **Sensitivity Analysis Ranking**:
  - **Cutting Rate (CR)**: Pulse-On Time ($T_{on}$) > Peak Current ($IP$) > Pulse-Off Time ($T_{off}$)
  - **Surface Roughness (SR)**: Pulse-On Time ($T_{on}$) > Peak Current ($IP$) > Servo Voltage ($SV$)
- **Probabilistic Insights**:
  Under 1%, 3%, and 5% operational uncertainty, output distributions remain strictly normal, establishing narrow, robust operational bounds for precision manufacturing.
