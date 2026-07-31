"""
Predict v2 Module - Alias to Customer Return Risk Analyzer API (predict.py).
"""

from predict import app, load_or_train_model, risk_model

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
