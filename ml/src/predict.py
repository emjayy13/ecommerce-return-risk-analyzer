import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict

app = FastAPI(
    title="E-Commerce Return Risk Analyzer - ML Prediction API",
    description="API for predicting customer return risk scores based on order and return history.",
    version="1.0.0"
)

# Load model pipeline
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'return_risk_model.pkl')

model = None


def load_model_binary():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"Successfully loaded model from {MODEL_PATH}")
    else:
        print(f"Warning: Model file not found at {MODEL_PATH}")


@app.on_event("startup")
def startup_event():
    load_model_binary()


class CustomerFeatureInput(BaseModel):
    total_orders: int = Field(..., description="Total number of orders placed by customer", example=10)
    return_ratio: float = Field(..., description="Ratio of returned orders (0.0 to 1.0)", example=0.25)
    avg_return_window: float = Field(..., description="Average days taken to return an order", example=4.5)
    vague_reason_count: int = Field(..., description="Count of vague return reasons provided", example=1)
    most_common_category: str = Field(..., description="Most frequent product category purchased", example="Clothing")
    mismatch_flag_history: bool = Field(..., description="Whether customer has history of wrong item claims", example=False)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "total_orders": 8,
                "return_ratio": 0.35,
                "avg_return_window": 3.0,
                "vague_reason_count": 2,
                "most_common_category": "Clothing",
                "mismatch_flag_history": True
            }
        }
    )


class RiskPredictionResponse(BaseModel):
    risk_score: float = Field(..., description="Predicted return risk score (0 - 100)")
    risk_level: str = Field(..., description="Risk category: Low, Medium, High")
    recommendation: str = Field(..., description="Suggested action based on risk level")


def determine_risk_level(score: float) -> str:
    if score <= 40:
        return "Low"
    elif score <= 70:
        return "Medium"
    else:
        return "High"


def get_recommendation(risk_level: str) -> str:
    if risk_level == "Low":
        return "Standard processing - Low risk customer"
    elif risk_level == "Medium":
        return "Monitor returns - Require specific return reasons"
    else:
        return "High risk - Require manual verification for return authorization"


@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "E-Commerce Return Risk Prediction API",
        "docs_url": "/docs"
    }


@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=RiskPredictionResponse)
def predict_risk(data: CustomerFeatureInput):
    global model
    if model is None:
        load_model_binary()
        if model is None:
            raise HTTPException(status_code=500, detail="ML model binary not found. Train model first.")

    input_df = pd.DataFrame([{
        'total_orders': data.total_orders,
        'return_ratio': data.return_ratio,
        'avg_return_window': data.avg_return_window,
        'vague_reason_count': data.vague_reason_count,
        'most_common_category': data.most_common_category,
        'mismatch_flag_history': int(data.mismatch_flag_history)
    }])

    raw_prediction = model.predict(input_df)[0]
    risk_score = round(float(max(0.0, min(100.0, raw_prediction))), 2)
    risk_level = determine_risk_level(risk_score)
    recommendation = get_recommendation(risk_level)

    return RiskPredictionResponse(
        risk_score=risk_score,
        risk_level=risk_level,
        recommendation=recommendation
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    