# E-Commerce Return Risk Analyzer - ML Subsystem

This directory contains the Machine Learning module for predicting customer return risk scores based on order history, return patterns, and categorical behavior.

---

## 📌 Architecture Overview

```
ml/
├── data/                                 # Datasets (raw, cleaned, simulated, aggregated)
│   ├── ecommerce_returns_synthetic_data.csv
│   ├── ecommerce_returns_clean.csv
│   ├── ecommerce_returns_simulated_customers.csv
│   └── customer_features.csv
├── models/                               # Trained model artifacts & metadata
│   ├── return_risk_model.pkl
│   └── return_risk_model_metadata.json
├── src/                                  # ML Pipeline Source Code
│   ├── preprocess.py                     # Data cleaning & anomaly removal
│   ├── simulate_customers.py             # Customer-level grouping & order simulation
│   ├── aggregate_features.py            # Feature engineering & rule-based scoring label generation
│   ├── train.py                          # Model training & pipeline export
│   ├── predict.py                        # FastAPI prediction endpoint
│   └── debug/                            # Diagnostic & analysis scripts
├── requirements.txt                      # ML module python dependencies
└── README.md                             # ML Documentation
```

---

## 🛠️ ML Pipeline Steps

1. **Preprocessing (`src/preprocess.py`)**:
   - Cleans raw synthetic e-commerce return data.
   - Handles missing values, filters negative return durations and >90 day outliers.
   - Computes order-level metrics (e.g., `Effective_Order_Value`, `Discount_Percentage`).

2. **Customer Order Simulation (`src/simulate_customers.py`)**:
   - Maps raw orders to customer IDs (~1,385 unique customer profiles across 5,500+ transactions).

3. **Feature Aggregation (`src/aggregate_features.py`)**:
   - Aggregates metrics per customer: `total_orders`, `return_ratio`, `avg_return_window`, `vague_reason_count`, `most_common_category`, `mismatch_flag_history`.
   - Computes initial risk score labels (0 to 100) and risk level buckets (`Low`, `Medium`, `High`).

4. **Model Training (`src/train.py`)**:
   - Trains a `GradientBoostingRegressor` scikit-learn pipeline with `OneHotEncoder` and `StandardScaler`.
   - Saves the serialized model to `models/return_risk_model.pkl`.

5. **FastAPI Serving (`src/predict.py`)**:
   - Exposes REST endpoints to score return risk dynamically.

---

## 🚀 Running the API Standalone

### 1. Install Dependencies
```bash
pip install -r ml/requirements.txt
```

### 2. Launch FastAPI Server
```bash
cd ml/src
python predict.py
```
*The server will start at `http://localhost:8000`.*

### 3. API Documentation & Testing
- Open Swagger UI in browser: **`http://localhost:8000/docs`**
- Send a sample `POST /predict` request:

```json
{
  "total_orders": 8,
  "return_ratio": 0.35,
  "avg_return_window": 3.0,
  "vague_reason_count": 2,
  "most_common_category": "Clothing",
  "mismatch_flag_history": true
}
```

#### Sample Response:
```json
{
  "risk_score": 88.47,
  "risk_level": "High",
  "recommendation": "High risk - Require manual verification for return authorization"
}
```

---

## 🧪 Health Check Endpoint
- **`GET /health`**: Returns status and confirms model load status.
```json
{
  "status": "healthy",
  "model_loaded": true
}
```
