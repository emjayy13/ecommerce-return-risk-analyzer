import os
import joblib
import json
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict

app = FastAPI(
    title="E-Commerce Return Risk Analyzer - ML Prediction API",
    description="Continuous learning model for predicting customer return risk.",
    version="2.0.0"
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_V2_PATH = os.path.join(MODEL_DIR, 'return_risk_model_v2.pkl')
MODEL_V1_PATH = os.path.join(MODEL_DIR, 'return_risk_model.pkl')
META_PATH = os.path.join(MODEL_DIR, 'return_risk_model_v2_metadata.json')

model = None
metadata = None


def load_model():
    global model, metadata
    if os.path.exists(MODEL_V2_PATH):
        model = joblib.load(MODEL_V2_PATH)
        print(f"Loaded v2 model from {MODEL_V2_PATH}")
    elif os.path.exists(MODEL_V1_PATH):
        model = joblib.load(MODEL_V1_PATH)
        print(f"Loaded v1 model from {MODEL_V1_PATH}")
    else:
        print("Warning: No model file found")

    if os.path.exists(META_PATH):
        with open(META_PATH) as f:
            metadata = json.load(f)


@app.on_event("startup")
def startup_event():
    load_model()


class OrderInput(BaseModel):
    Product_Price: float = Field(..., example=150.0)
    Order_Quantity: int = Field(..., example=2)
    User_Age: int = Field(..., example=35)
    Discount_Applied: float = Field(..., example=10.0)
    Product_Category: str = Field(..., example="Electronics")
    User_Gender: str = Field(..., example="Male")
    Payment_Method: str = Field(..., example="Credit Card")
    Shipping_Method: str = Field(..., example="Standard")
    Order_Month: int = Field(default=6, example=6)
    Order_Day: int = Field(default=15, example=15)
    Order_DayOfWeek: int = Field(default=2, example=2)
    Is_Weekend: int = Field(default=0, example=0)
    Is_Clothing_Category: int = Field(default=0, example=0)
    High_Discount_Flag: int = Field(default=0, example=0)
    Is_Express_Shipping: int = Field(default=0, example=0)
    total_orders: int = Field(default=5, example=5)
    return_ratio: float = Field(default=0.1, example=0.1)
    avg_return_window: float = Field(default=0.0, example=0.0)
    vague_reason_count: int = Field(default=0, example=0)
    mismatch_flag_history: int = Field(default=0, example=0)
    category_return_rate: float = Field(default=0.107, example=0.107)
    category_avg_price: float = Field(default=252.0, example=252.0)
    category_order_count: int = Field(default=1100, example=1100)
    location_return_rate: float = Field(default=0.107, example=0.107)
    location_order_count: int = Field(default=56, example=56)
    payment_return_rate: float = Field(default=0.107, example=0.107)
    payment_order_count: int = Field(default=1386, example=1386)
    shipping_return_rate: float = Field(default=0.107, example=0.107)
    shipping_order_count: int = Field(default=1847, example=1847)
    price_percentile: float = Field(default=0.5, example=0.5)
    quantity_percentile: float = Field(default=0.5, example=0.5)
    discount_percentile: float = Field(default=0.5, example=0.5)
    price_x_quantity: float = Field(default=300.0, example=300.0)
    discount_ratio: float = Field(default=0.067, example=0.067)
    price_deviation_from_category: float = Field(default=0.0, example=0.0)

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "Product_Price": 150.0, "Order_Quantity": 2, "User_Age": 35,
            "Discount_Applied": 10.0, "Product_Category": "Electronics",
            "User_Gender": "Male", "Payment_Method": "Credit Card",
            "Shipping_Method": "Standard"
        }
    })


class PredictionResponse(BaseModel):
    return_probability: float = Field(..., description="Probability of return (0-1)")
    risk_level: str = Field(..., description="Risk category: Low, Medium, High")
    confidence: float = Field(..., description="Model confidence in prediction")
    recommendation: str = Field(..., description="Suggested action")
    model_version: str = Field(..., description="Model version used")


def get_risk_level(prob: float) -> str:
    if prob < 0.2:
        return "Low"
    elif prob < 0.5:
        return "Medium"
    else:
        return "High"


def get_recommendation(risk_level: str, prob: float) -> str:
    if risk_level == "Low":
        return "Standard processing - Low return risk customer"
    elif risk_level == "Medium":
        return "Monitor returns - Verify return reason before approval"
    else:
        return "High risk - Manual review required for return authorization"


@app.get("/")
def root():
    return {"status": "online", "service": "Return Risk Analyzer v2", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None, "version": "2.0"}


@app.get("/model/info")
def model_info():
    if metadata:
        return metadata
    return {"error": "No metadata available"}


@app.post("/predict", response_model=PredictionResponse)
def predict(data: OrderInput):
    global model
    if model is None:
        load_model()
        if model is None:
            raise HTTPException(status_code=500, detail="Model not found. Train first.")

    input_dict = data.model_dump()
    input_df = pd.DataFrame([input_dict])

    try:
        prob = model.predict_proba(input_df)[0][1]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

    risk_level = get_risk_level(prob)
    confidence = max(prob, 1 - prob)

    return PredictionResponse(
        return_probability=round(float(prob), 4),
        risk_level=risk_level,
        confidence=round(float(confidence), 4),
        recommendation=get_recommendation(risk_level, prob),
        model_version=metadata.get('version', 'unknown') if metadata else 'v1'
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
