# Project Commands Reference - PowerShell

## 1. Project Setup (First Time)

```powershell
# Clone the repo
git clone https://github.com/21mohit-dotcom/ecommerce-return-risk-analyzer.git
cd ecommerce-return-risk-analyzer

# Install Python dependencies (ML + Backend)
pip install -r ml/requirements.txt

# Install frontend dependencies (when React is set up)
cd frontend
npm install
cd ..
```

---

## 2. ML Model Commands

```powershell
# Train the full ML pipeline (preprocessing + simulation + features + model training)
python ml/src/train.py

# Run 10 query test on ML API
python ml/src/debug/run_10_queries.py

# Run 20 request batch evaluation
python ml/src/debug/test_20_requests.py

# Test continuous learning pipeline
python ml/src/debug/test_continuous_learning.py

# Test live API endpoints
python ml/src/debug/test_live_api.py

# Check feature distribution
python ml/src/debug/check_distribution.py
```

---

## 3. Backend / API Commands

```powershell
# Start FastAPI server (runs on http://localhost:8000)
cd ml/src
python predict.py

# Or from project root
python ml/src/predict.py

# Access Swagger API docs
# Open browser: http://localhost:8000/docs

# Test API health
Invoke-RestMethod -Uri "http://localhost:8000/health" -Method GET

# Test model info
Invoke-RestMethod -Uri "http://localhost:8000/model/info" -Method GET

# Test prediction endpoint
$body = @{
    customer_id = "CUST000063"
    order_id = "ORD_TEST_001"
    product_category = "Clothing"
    return_reason = "Defective"
    is_returned = $true
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method POST -Body $body -ContentType "application/json"

# Test feedback endpoint
$feedback = @{
    customer_id = "CUST000063"
    actual_fraud_label = 1
    notes = "Confirmed fraud after manual audit"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/feedback" -Method POST -Body $feedback -ContentType "application/json"

# Trigger manual retrain
Invoke-RestMethod -Uri "http://localhost:8000/retrain?force=true" -Method POST

# Get retrain history
Invoke-RestMethod -Uri "http://localhost:8000/retrain/history" -Method GET

# Get customer profile
Invoke-RestMethod -Uri "http://localhost:8000/customer/CUST000063" -Method GET
```

---

## 4. Frontend Commands (When React is set up)

```powershell
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Run tests
npm test

# Lint check
npm run lint
```

---

## 5. Git Commands

```powershell
# Check current status
git status

# Create a new feature branch
git checkout -b feature/your-feature-name

# Stage files
git add .

# Or stage specific files
git add ml/src/predict.py
git add frontend/src/App.jsx

# Commit
git commit -m "your commit message"

# Push to GitHub
git push origin feature/your-feature-name

# Pull latest changes from main
git pull origin main

# Switch to main branch
git checkout main

# Merge feature branch into main
git merge feature/your-feature-name

# View commit history
git log --oneline -10

# View changes before committing
git diff

# Stash changes (save without committing)
git stash

# Apply stashed changes
git stash pop
```

---

## 6. Testing Commands

```powershell
# Run all ML tests
python ml/src/debug/test_live_api.py
python ml/src/debug/test_continuous_learning.py
python ml/src/debug/test_20_requests.py
python ml/src/debug/run_10_queries.py

# Run frontend tests (when React is set up)
cd frontend
npm test

# Run pytest (if pytest is added later)
python -m pytest ml/src/debug/ -v
```

---

## 7. API Endpoints Quick Reference

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/` | Server status |
| GET | `/health` | Health check |
| GET | `/model/info` | Model version and stats |
| GET | `/model/metrics` | Detailed CV metrics |
| POST | `/predict` | Predict customer risk |
| POST | `/predict-new-customer` | New customer prediction |
| POST | `/feedback` | Submit audit feedback |
| POST | `/retrain` | Manual retrain trigger |
| GET | `/retrain/history` | Retraining audit log |
| GET | `/customer/{id}` | Customer profile lookup |
| POST | `/update-history` | Update customer history |

---

## 8. Project Structure (Quick Reference)

```
ecommerce-return-risk-analyzer/
├── ml/                          # ML module (main code)
│   ├── src/                     # Python source files
│   │   ├── train.py             # Full pipeline trainer
│   │   ├── predict.py           # FastAPI server
│   │   ├── customer_risk_model.py  # Core ML engine
│   │   ├── preprocess.py        # Data cleaning
│   │   ├── simulate_customers.py   # Customer simulation
│   │   ├── aggregate_features.py   # Feature engineering
│   │   ├── continuous_pipeline.py  # Retraining
│   │   └── debug/               # Test scripts
│   ├── data/                    # Datasets
│   ├── models/                  # Trained model files
│   └── requirements.txt         # Python dependencies
├── frontend/                    # React app (to be built)
├── backend/                     # Backend (to be built)
├── docs/                        # Documentation
└── README.md
```

---

## 9. Model Files Reference

| File | Purpose |
|------|---------|
| `ml/models/customer_risk_model.pkl` | Active production model |
| `ml/models/customer_risk_metadata.json` | Model metrics and version info |
| `ml/models/retraining_history.json` | Retraining audit log |
| `ml/models/versions/` | Archived model versions |
| `ml/data/customer_features.csv` | Customer feature dataset |
| `ml/data/feedback_records.csv` | Feedback records for continuous learning |
