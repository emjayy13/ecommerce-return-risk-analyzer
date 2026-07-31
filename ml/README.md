# E-Commerce Customer Return Risk Analyzer - ML Subsystem (v2)

This directory contains the Machine Learning module for predicting customer return risk scores (0-100) based on historical customer behavior and independent ground-truth audit labels.

> **Full Master Technical & Interviewer Documentation**: See [docs/PROJECT_ML_DOCUMENTATION.md](file:///c:/Users/mohit/Desktop/ecommerce-return-risk-analyzer/docs/PROJECT_ML_DOCUMENTATION.md) for full pipeline details, metrics, and 12 interviewer Q&As.

---

## Architecture Overview

```
ml/
├── data/                                 # Datasets (raw, cleaned, simulated, aggregated, feedback)
│   ├── ecommerce_returns_synthetic_data.csv
│   ├── ecommerce_returns_clean.csv
│   ├── ecommerce_returns_simulated_customers.csv
│   ├── customer_features.csv
│   └── feedback_records.csv
├── models/                               # Active production model & versions
│   ├── customer_risk_model.pkl           # Active production model binary
│   ├── customer_risk_metadata.json       # Production metrics metadata
│   ├── retraining_history.json           # Continuous learning audit logs
│   └── versions/                         # Archived versioned models (model_vN.pkl)
├── src/                                  # ML Pipeline Source Code
│   ├── preprocess.py                     # Data cleaning & anomaly removal
│   ├── simulate_customers.py             # Customer-level grouping & independent ground-truth target
│   ├── aggregate_features.py            # Feature engineering (10 historical customer metrics)
│   ├── customer_risk_model.py            # Core ML Engine, preprocessor, and champion/challenger evaluator
│   ├── train.py                          # Full pipeline execution & baseline model exporter
│   ├── continuous_pipeline.py            # Retraining orchestrator
│   ├── predict.py                        # FastAPI REST Serving API (Continuous Learning v2)
│   └── debug/                            # Test & evaluation scripts
│       ├── test_live_api.py
│       ├── test_continuous_learning.py
│       └── test_20_requests.py
├── requirements.txt                      # Python dependencies
└── README.md                             # ML Module Overview
```

---

## ML Pipeline & Continuous Learning Steps

1. **Preprocessing (`src/preprocess.py`)**:
   - Cleans raw synthetic e-commerce return data.
   - Handles missing values, filters negative return durations and >90 day outliers.

2. **Customer Simulation & Ground-Truth (`src/simulate_customers.py`)**:
   - Maps raw orders to customer IDs (~1,350+ unique customer profiles).
   - Assigns independent ground-truth target `flagged_by_company` (No Data Leakage!).

3. **Feature Aggregation (`src/aggregate_features.py`)**:
   - Aggregates 10 historical customer features per customer: `total_orders`, `total_returns`, `return_ratio`, `avg_return_window`, `product_category_risk`, `vague_reason_count`, `mismatch_flag_history`, `customer_rating_behavior`, `previous_fraud_flags`, `account_age_days`.

4. **Model Training & Cross-Validation (`src/customer_risk_model.py` & `src/train.py`)**:
   - Preprocessing using `StandardScaler` for numeric and `OneHotEncoder` for categorical features.
   - Evaluates `LogisticRegression` (Champion), `RandomForestClassifier`, and `GradientBoostingClassifier` using **5-Fold Stratified Cross-Validation**.

5. **Continuous Learning v2 (`src/predict.py`)**:
   - `POST /predict`: Generates Risk Score (0-100) and Level (Low, Medium, High).
   - `POST /feedback`: Ingests verified audit ground-truth labels.
   - Auto-triggers threshold retraining every 10 feedback items.
   - **Champion vs. Challenger Gatekeeper**: Replaces production model only if the new Challenger model demonstrates superior ROC-AUC / F1 performance.

---

## Running API & Tests

### Install Dependencies
```bash
pip install -r ml/requirements.txt
```

### Run Model Training Pipeline
```bash
python ml/src/train.py
```

### Run Live Batch 20-Request Test & Metrics Evaluation
```bash
python ml/src/debug/test_20_requests.py
```

### Launch FastAPI Server
```bash
cd ml/src
python predict.py
```
*Access Swagger UI documentation at: `http://localhost:8000/docs`*
