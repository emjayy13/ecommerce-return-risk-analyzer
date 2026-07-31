import os
import json
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional

app = FastAPI(
    title="E-Commerce Customer Return Risk API",
    description="Continuous learning model for customer return risk scoring based on historical behavior.",
    version="2.0.0"
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'customer_risk_model.pkl')

risk_model = None


def load_model():
    global risk_model
    from src.customer_risk_model import CustomerRiskModel
    risk_model = CustomerRiskModel()
    if risk_model.load_model():
        print(f"Model loaded: {risk_model.version}")
    else:
        print("No saved model found. Training...")
        orders_df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'order_features.csv'))
        customer_df = risk_model.build_features_from_orders(orders_df)
        risk_model.train(customer_df)


@app.on_event("startup")
def startup():
    load_model()


class ReturnRequest(BaseModel):
    customer_id: str = Field(..., description="Customer ID", example="CUST000063")
    order_id: str = Field(..., description="Order ID", example="ORD00007551")
    product_category: str = Field(default="Electronics", example="Electronics")
    return_reason: str = Field(default="Changed mind", example="Changed mind")
    is_returned: bool = Field(default=True, example=True)

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "customer_id": "CUST000063",
            "order_id": "ORD00007551",
            "product_category": "Clothing",
            "return_reason": "Defective",
            "is_returned": True
        }
    })


class RiskResponse(BaseModel):
    customer_id: str
    risk_score: float = Field(..., description="Risk score 0-100")
    risk_level: str = Field(..., description="Low/Medium/High")
    return_probability: float = Field(..., description="Probability of return 0-1")
    prediction_method: str = Field(..., description="ml_model or rule_based")
    model_version: str
    customer_history: dict = Field(..., description="Customer's historical features")
    recommendation: str


class NewCustomerFlow(BaseModel):
    customer_id: str
    risk_score: float
    risk_level: str
    prediction_method: str
    message: str
    recommendation: str


def get_recommendation(level, score):
    if level == 'Low':
        return "Standard processing - Approve return automatically"
    elif level == 'Medium':
        return "Monitor - Require valid return reason before approval"
    else:
        return "High risk - Flag for manual review by fraud team"


@app.get("/")
def root():
    return {"status": "online", "service": "Customer Return Risk API v2", "docs": "/docs"}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": risk_model is not None,
        "model_version": risk_model.version if risk_model else None,
        "total_customers": len(risk_model.customer_db) if risk_model else 0
    }


@app.post("/predict", response_model=RiskResponse)
def predict_risk(request: ReturnRequest):
    """Predict return risk for a customer based on their historical behavior."""
    if risk_model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    customer_features = risk_model.get_customer_features(request.customer_id)

    if customer_features is None:
        raise HTTPException(
            status_code=404,
            detail=f"Customer {request.customer_id} not found. Use /predict-new-customer for new customers."
        )

    result = risk_model.predict_risk(customer_features)

    return RiskResponse(
        customer_id=request.customer_id,
        risk_score=result['risk_score'],
        risk_level=result['risk_level'],
        return_probability=result['return_probability'],
        prediction_method=result['prediction_method'],
        model_version=result['model_version'],
        customer_history=customer_features,
        recommendation=get_recommendation(result['risk_level'], result['risk_score'])
    )


@app.post("/predict-new-customer", response_model=NewCustomerFlow)
def predict_new_customer(request: ReturnRequest):
    """Handle return request for a new customer with no history."""
    default_features = risk_model.get_default_features(request.customer_id) if risk_model else {
        'total_orders': 0, 'return_ratio': 0.0, 'avg_return_window': 0.0,
        'vague_reason_count': 0, 'mismatch_flag_history': 0, 'most_common_category': 'Unknown'
    }

    if risk_model:
        result = risk_model.predict_risk(default_features)
    else:
        result = {'risk_score': 0, 'risk_level': 'Low', 'prediction_method': 'rule_based'}

    if risk_model:
        risk_model.update_customer_history(request.customer_id, {
            'is_returned': request.is_returned,
            'return_reason': request.return_reason
        })

    return NewCustomerFlow(
        customer_id=request.customer_id,
        risk_score=result['risk_score'],
        risk_level=result['risk_level'],
        prediction_method='new_customer_default',
        message="New customer - using default features. History will be built for future predictions.",
        recommendation="New customer - Apply standard return policy. Start building history."
    )


@app.post("/update-history")
def update_history(request: ReturnRequest):
    """Update customer history after a return is processed."""
    if risk_model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    risk_model.update_customer_history(request.customer_id, {
        'is_returned': request.is_returned,
        'return_reason': request.return_reason
    })

    return {
        "status": "updated",
        "customer_id": request.customer_id,
        "message": "Customer history updated"
    }


@app.get("/customer/{customer_id}")
def get_customer(customer_id: str):
    """Get customer features and risk assessment."""
    if risk_model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    features = risk_model.get_customer_features(customer_id)
    if features is None:
        return {"customer_id": customer_id, "status": "new_customer", "features": risk_model.get_default_features(customer_id)}

    result = risk_model.predict_risk(features)
    return {
        "customer_id": customer_id,
        "status": "existing_customer",
        "features": features,
        "risk_assessment": result
    }


@app.get("/model/metrics")
def model_metrics():
    """Get model performance metrics."""
    if risk_model is None:
        return {"error": "No model loaded"}
    return {
        "version": risk_model.version,
        "metrics": risk_model.metrics,
        "total_customers": len(risk_model.customer_db)
    }


@app.post("/retrain")
def retrain_model():
    """Retrain model with all accumulated data."""
    if risk_model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    orders_df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'order_features.csv'))
    customer_df = risk_model.build_features_from_orders(orders_df)
    results = risk_model.train(customer_df)

    return {
        "status": "retrained",
        "version": risk_model.version,
        "metrics": {name: {'auc_roc': r['auc_roc'], 'f1': r['f1']} for name, r in results.items()}
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
