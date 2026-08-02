# ML Subsystem - Customer Return Risk Analyzer

Predicts customer return risk scores (0-100) using scikit-learn with continuous learning.

---

## Features

- Synthetic ecommerce data generation (5,400 orders, 10 categories)
- Data preprocessing and cleaning
- Customer-level feature engineering (10 numeric + 1 categorical)
- Model training with 3 candidates (RandomForest, GradientBoosting, LogisticRegression)
- Best model selection by ROC-AUC
- FastAPI REST API with 11 endpoints
- Feedback collection and automatic retraining (threshold: 10 items)
- Champion vs. Challenger model promotion
- Model versioning with archived versions

---

## Project Structure

```text
ml/
├── src/
│   ├── generate_data.py           # Synthetic data generation
│   ├── preprocess.py              # Data cleaning & feature engineering
│   ├── simulate_customers.py      # Customer profile simulation
│   ├── aggregate_features.py      # Customer-level feature aggregation
│   ├── customer_risk_model.py     # Core ML engine (train, predict, retrain)
│   ├── train.py                   # Pipeline orchestrator
│   ├── predict.py                 # FastAPI REST API
│   ├── continuous_pipeline.py     # Standalone retraining script
│   └── debug/                     # Test & evaluation scripts
├── data/                          # Generated after training (not in repo)
├── models/                        # Generated after training (not in repo)
│   └── versions/                  # Archived model versions
├── notebooks/                     # Jupyter notebooks (EDA)
└── requirements.txt
```

---

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the complete pipeline:

```bash
python src/train.py
```

Start the API:

```bash
python src/predict.py
```

API Documentation:

```text
http://localhost:8000/docs
```

---

## ML Pipeline

```text
Generate Data (5,400 orders)
        │
        ▼
Preprocess Data (clean, engineer features)
        │
        ▼
Simulate Customers (~1,350 profiles)
        │
        ▼
Aggregate Features (10 numeric + 1 categorical)
        │
        ▼
Train Model (3 candidates, best by ROC-AUC)
        │
        ▼
Serve Predictions (FastAPI)
```

---

## Model

Trains 3 candidate models and selects the best by ROC-AUC:

| Model | Parameters |
|-------|-----------|
| RandomForestClassifier | 200 trees, max_depth=6, balanced |
| GradientBoostingClassifier | 150 trees, lr=0.05, max_depth=3 |
| LogisticRegression | max_iter=1000, balanced, C=1.0 |

Evaluation: 80/20 stratified split + 5-Fold Stratified Cross-Validation.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Service status |
| GET | `/health` | Health check with model stats |
| GET | `/model/info` | Active model version and metrics |
| GET | `/model/metrics` | Detailed CV metrics breakdown |
| POST | `/predict` | Predict customer return risk (0-100) |
| POST | `/predict-new-customer` | Predict for new customers |
| POST | `/feedback` | Submit verified audit ground-truth label |
| POST | `/retrain` | Manually trigger model retraining |
| GET | `/retrain/history` | Retraining audit log |
| POST | `/update-history` | Update customer transaction history |
| GET | `/customer/{id}` | Customer profile lookup |

---

## Continuous Learning

1. Submit feedback via `POST /feedback`
2. Every 10 feedback items triggers automatic retraining
3. New model promoted only if it outperforms current model (ROC-AUC)
4. All model versions archived in `ml/models/versions/`

---

## Generated Files

Generated automatically by `python src/train.py` and excluded from Git:

```text
ml/data/
├── ecommerce_returns_synthetic_data.csv
├── ecommerce_returns_clean.csv
├── ecommerce_returns_simulated_customers.csv
├── customer_features.csv
└── feedback_records.csv

ml/models/
├── customer_risk_model.pkl
├── customer_risk_metadata.json
├── retraining_history.json
└── versions/
```

To regenerate:

```bash
python src/train.py
```

---

## Useful Commands

```bash
# Verify model exists
python -c "import os; print(os.path.exists('models/customer_risk_model.pkl'))"

# Check customer count
python -c "import joblib; d=joblib.load('models/customer_risk_model.pkl'); print(len(d['customer_db']))"

# View model metadata
python -c "import json; print(json.dumps(json.load(open('models/customer_risk_metadata.json')), indent=2))"
```

---

## Future Improvements

- Dockerfile for containerized deployment
- Environment variable configuration
- pytest integration with CI/CD
- MongoDB for persistent customer database
- React dashboard for analytics visualization
