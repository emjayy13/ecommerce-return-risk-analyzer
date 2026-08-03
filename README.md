# Customer Return Risk Analyzer

ML-powered system to detect potentially fraudulent customer return behavior with continuous learning and automated model retraining.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2+-orange.svg)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Table of Contents

- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [ML Pipeline](#ml-pipeline)
- [API Reference](#api-reference)
- [Continuous Learning](#continuous-learning)
- [Development](#development)
- [Testing](#testing)
- [Future Work](#future-work)
- [License](#license)

---

## Quick Start

### Prerequisites

- Python 3.10+
- pip

### Get Running in 30 Seconds

```bash
git clone <repo>
cd ecommerce-return-risk-analyzer
pip install -r ml/requirements.txt
python ml/src/train.py
python ml/src/predict.py
```

Server starts at `http://localhost:8000`.

### Verify Installation

```bash
# Check server health
curl http://localhost:8000/health

# Test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"customer_id": "CUST000063", "product_category": "Electronics", "return_reason": "Defective", "is_returned": true}'
```

---

## Architecture

```
Clone Repository
      │
      ▼
pip install -r requirements.txt
      │
      ▼
python ml/src/train.py              ← Data generation + Training
      │
      ├── generate_data.py          → Raw synthetic dataset (5,400 orders)
      ├── preprocess.py             → Clean dataset (anomaly removal, feature engineering)
      ├── simulate_customers.py     → Customer profiles (~1,350 customers)
      ├── aggregate_features.py     → 10 features per customer
      └── customer_risk_model.py    → Trained model.pkl (3 candidates, best by ROC-AUC)
      │
      ▼
python ml/src/predict.py            ← FastAPI serves predictions
      │
      ├── Validates model exists on startup
      ├── Returns clear error if missing
      └── Serves /predict, /feedback, /retrain endpoints
```

**Key Design Principle**: Training and prediction are completely separated. The API never trains.

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| API | FastAPI | REST API serving |
| ML | Python, scikit-learn, pandas | Model training & inference |

---

## Project Structure

```
ecommerce-return-risk-analyzer/
├── ml/
│   ├── data/                              # Generated datasets (not in repo)
│   │   ├── ecommerce_returns_synthetic_data.csv
│   │   ├── ecommerce_returns_clean.csv
│   │   ├── ecommerce_returns_simulated_customers.csv
│   │   ├── customer_features.csv
│   │   └── feedback_records.csv           # Runtime feedback
│   ├── models/                            # Trained models (not in repo)
│   │   ├── customer_risk_model.pkl        # Production model
│   │   ├── customer_risk_metadata.json    # Model metadata
│   │   ├── retraining_history.json        # Audit log
│   │   └── versions/                      # Versioned model archives
│   ├── src/
│   │   ├── generate_data.py               # Synthetic data generation
│   │   ├── preprocess.py                  # Data cleaning & feature engineering
│   │   ├── simulate_customers.py          # Customer profile simulation
│   │   ├── aggregate_features.py          # Customer-level feature aggregation
│   │   ├── customer_risk_model.py         # Core ML engine + continuous learning
│   │   ├── train.py                       # Pipeline orchestrator
│   │   ├── predict.py                     # FastAPI REST API (prediction only)
│   │   ├── continuous_pipeline.py         # Standalone retraining script
│   │   └── debug/                         # Test & evaluation scripts
│   ├── notebooks/                         # Jupyter notebooks (EDA)
│   └── requirements.txt                   # Python dependencies
└── docs/                                  # Project documentation
```

---

## ML Pipeline

### Pipeline Flow

| Step | Script | Input | Output | Description |
|------|--------|-------|--------|-------------|
| 1 | `generate_data.py` | - | `ecommerce_returns_synthetic_data.csv` | Generate 5,400 synthetic orders |
| 2 | `preprocess.py` | Raw CSV | `ecommerce_returns_clean.csv` | Clean data, engineer features |
| 3 | `simulate_customers.py` | Clean CSV | `ecommerce_returns_simulated_customers.csv` | Assign customer profiles + fraud labels |
| 4 | `aggregate_features.py` | Simulated CSV | `customer_features.csv` | 10 features per customer |
| 5 | `customer_risk_model.py` | Features CSV | `customer_risk_model.pkl` | Train 3 candidates, select best |

### Feature Engineering

**10 Numeric Features:**

| Feature | Description |
|---------|-------------|
| `total_orders` | Total orders placed by customer |
| `total_returns` | Total returns made by customer |
| `return_ratio` | total_returns / total_orders |
| `avg_return_window` | Average days between order and return |
| `product_category_risk` | Proportion of orders in high-risk categories |
| `vague_reason_count` | Count of "Defective" / "Not as described" returns |
| `mismatch_flag_history` | Whether customer ever claimed "Wrong item" |
| `customer_rating_behavior` | Simulated customer rating (1-5) |
| `previous_fraud_flags` | Historical fraud flags |
| `account_age_days` | Account age in days |

**1 Categorical Feature:** `most_common_category`

**Target Variable:** `flagged_by_company` (independent ground-truth fraud label, zero data leakage)

### Model Training

| Candidate | Parameters |
|-----------|-----------|
| RandomForestClassifier | 200 trees, max_depth=6, balanced |
| GradientBoostingClassifier | 150 trees, lr=0.05, max_depth=3 |
| LogisticRegression | max_iter=1000, balanced, C=1.0 |

**Evaluation:** 80/20 stratified split + 5-Fold Stratified CV. Best model selected by ROC-AUC.

---

## API Reference

### Base URL

```
http://localhost:8000
```

### Swagger Documentation

```
http://localhost:8000/docs
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Service status |
| GET | `/health` | Health check with model stats |
| GET | `/model/info` | Active model version and metrics |
| GET | `/model/metrics` | Detailed CV metrics breakdown |
| POST | `/predict` | Predict customer return risk (0-100) |
| POST | `/predict-new-customer` | Predict for new customers |
| POST | `/feedback` | Submit verified audit ground-truth label |
| POST | `/retrain` | Manually trigger model retraining |
| GET | `/retrain/history` | View retraining audit log |
| POST | `/update-history` | Update customer transaction history |
| GET | `/customer/{id}` | Look up customer profile |

### Example Requests

**Health Check:**
```bash
curl http://localhost:8000/health
```

**Predict Risk:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "CUST000063",
    "order_id": "ORD00007551",
    "product_category": "Clothing",
    "return_reason": "Defective",
    "is_returned": true
  }'
```

**Submit Feedback:**
```bash
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "CUST000063",
    "actual_fraud_label": 1,
    "notes": "Verified fraud after manual inspection"
  }'
```

**Manual Retrain:**
```bash
curl -X POST "http://localhost:8000/retrain?force=true"
```

**Get Customer Profile:**
```bash
curl http://localhost:8000/customer/CUST000063
```

---

## Continuous Learning

### How It Works

```
POST /feedback (x10)
        │
        ▼
Threshold Reached
        │
        ▼
Automatic Retraining
        │
        ▼
Champion vs. Challenger
        │
        ├── Challenger wins → Promote to production
        └── Champion wins  → Keep current model
```

1. **Store Predictions**: Every `/predict` request updates the in-memory customer database
2. **Collect Feedback**: `/feedback` endpoint stores verified ground-truth audit labels
3. **Threshold Trigger**: Every 10 feedback items, automatic retraining is triggered
4. **Champion vs. Challenger**: New model promoted if `challenger_auc >= champion_auc`, or if forced, or if no model exists yet
5. **Versioning**: Every retrained model saved to `ml/models/versions/` with timestamp

### Manual Operations

```bash
# Trigger manual retrain
curl -X POST "http://localhost:8000/retrain?force=true"

# View retraining history
curl http://localhost:8000/retrain/history

# Standalone retraining script
python ml/src/continuous_pipeline.py
```

---

## Development

### Setup Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r ml/requirements.txt
```

### Run Individual Pipeline Steps

```bash
# Step 1: Generate synthetic data only
python ml/src/generate_data.py

# Step 2: Preprocess raw data only
python ml/src/preprocess.py

# Step 3: Simulate customer profiles only
python ml/src/simulate_customers.py

# Step 4: Aggregate features only
python ml/src/aggregate_features.py

# Step 5: Retrain model (requires customer_features.csv)
python ml/src/continuous_pipeline.py
```

### Useful Commands

```bash
# Check generated file sizes
ls -lh ml/data/*.csv ml/models/*.pkl

# Verify model exists
python -c "import os; print('Model exists:', os.path.exists('ml/models/customer_risk_model.pkl'))"

# Check customer count in trained model
python -c "import joblib; d = joblib.load('ml/models/customer_risk_model.pkl'); print(f'Customers: {len(d[\"customer_db\"])}')"

# View model metadata
python -c "import json; print(json.dumps(json.load(open('ml/models/customer_risk_metadata.json')), indent=2))"

# View retraining history
python -c "import json; print(json.dumps(json.load(open('ml/models/retraining_history.json')), indent=2))"
```

---

## Testing

### Run Tests

```bash
# Smoke test (health, predict, customer profile)
python ml/src/debug/test_live_api.py

# Continuous learning test (submits 10 feedbacks, triggers retrain)
python ml/src/debug/test_continuous_learning.py

# Batch evaluation (20 requests, metrics calculation)
python ml/src/debug/test_20_requests.py
```

### Validate Pipeline Reproducibility

```bash
# Delete all generated files
rm -rf ml/data/*.csv ml/models/*.pkl ml/models/*.json ml/models/versions/*.pkl ml/models/versions/*_metrics.json

# Run pipeline from scratch
python ml/src/train.py

# Verify all files regenerated
ls -la ml/data/*.csv ml/models/*.pkl
```

---

## Future Work

Features planned but not yet implemented:

- [ ] **Dockerfile** - Containerized deployment for production
- [ ] **Environment Variables** - Configurable model/data paths via `.env`
- [ ] **Makefile** - Single `make setup` / `make run` commands
- [ ] **pytest Integration** - Automated test suite with CI/CD pipeline
- [ ] **uvicorn Workers** - Multi-worker production server configuration
- [ ] **MongoDB Integration** - Persistent customer database (currently in-memory)
- [ ] **Frontend Dashboard** - React + Recharts analytics visualization
- [ ] **CI/CD Pipeline** - GitHub Actions for automated testing and deployment

---

## License

MIT
