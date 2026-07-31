import os
import sys
import pandas as pd
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict

# Ensure current src directory is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from customer_risk_model import CustomerRiskModel

app = FastAPI(
    title="Customer Return Risk Analyzer API",
    description="API for evaluating customer return risk scores (0–100) based on historical customer return patterns.",
    version="3.0.0"
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

risk_model = CustomerRiskModel()


def load_or_train_model():
    """Loads existing trained model or runs training pipeline if missing."""
    global risk_model
    if not risk_model.load_model():
        print("Model file not found. Running initial training pipeline...")
        features_csv = os.path.join(BASE_DIR, 'data', 'customer_features.csv')
        if os.path.exists(features_csv):
            customer_df = pd.read_csv(features_csv)
            risk_model.train(customer_df)
        else:
            from train import run_full_ml_pipeline
            risk_model = run_full_ml_pipeline()


@app.on_event("startup")
def startup_event():
    load_or_train_model()


class ReturnRequest(BaseModel):
    customer_id: str = Field(..., description="Unique customer identifier", example="CUST000063")
    order_id: Optional[str] = Field(None, description="Order identifier", example="ORD00007551")
    product_category: Optional[str] = Field("Electronics", description="Category of purchased item", example="Electronics")
    return_reason: Optional[str] = Field("Changed mind", description="Customer provided return reason", example="Defective")
    is_returned: bool = Field(True, description="Whether this request is a return event")

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "customer_id": "CUST000063",
            "order_id": "ORD00007551",
            "product_category": "Clothing",
            "return_reason": "Defective",
            "is_returned": True
        }
    })


class RiskPredictionResponse(BaseModel):
    customer_id: str
    risk_score: float = Field(..., description="Customer return risk score from 0 to 100")
    risk_level: str = Field(..., description="Risk tier: Low, Medium, High")
    risk_probability: float = Field(..., description="Probability score between 0.0 and 1.0")
    prediction_method: str = Field(..., description="Method used: ml_model, new_customer_baseline, or rule_based_fallback")
    model_version: str = Field(..., description="Trained model version identifier")
    recommendation: str = Field(..., description="Actionable recommendation for merchant")
    customer_history: Dict[str, Any] = Field(..., description="Historical customer features from database")


class NewCustomerResponse(BaseModel):
    customer_id: str
    risk_score: float
    risk_level: str
    prediction_method: str
    message: str
    recommendation: str


@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "Customer Return Risk Analyzer API",
        "version": "3.0.0",
        "docs_url": "/docs"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": risk_model.model is not None,
        "model_version": risk_model.version,
        "total_customers_in_db": len(risk_model.customer_db)
    }


@app.get("/model/metrics")
def get_model_metrics():
    """Returns model performance metrics and cross-validation evaluation."""
    if risk_model.model is None:
        raise HTTPException(status_code=500, detail="Model not initialized")
    return {
        "version": risk_model.version,
        "metrics": risk_model.metrics,
        "cv_results": risk_model.cv_results,
        "total_customers": len(risk_model.customer_db)
    }


@app.post("/predict", response_model=RiskPredictionResponse)
def predict_customer_risk(request: ReturnRequest):
    """
    Predicts risk score (0–100) and risk level (Low, Medium, High) for a customer's return request.
    Fetches historical features from database if existing, or uses sensible default baselines for new customers.
    """
    if risk_model.model is None:
        load_or_train_model()

    customer_features = risk_model.get_customer_features(request.customer_id)

    if customer_features is None:
        # Handling new customer flow
        default_features = risk_model.get_default_features(request.customer_id)
        prediction = risk_model.predict_risk(default_features)
        prediction['prediction_method'] = 'new_customer_baseline'
        
        # Initialize/collect history for future predictions
        updated_features = risk_model.update_customer_history(request.customer_id, request.model_dump())

        return RiskPredictionResponse(
            customer_id=request.customer_id,
            risk_score=prediction['risk_score'],
            risk_level=prediction['risk_level'],
            risk_probability=prediction['risk_probability'],
            prediction_method=prediction['prediction_method'],
            model_version=prediction['model_version'],
            recommendation="New customer - Standard processing. Transaction recorded for future history.",
            customer_history=updated_features
        )

    # Existing customer prediction flow
    prediction = risk_model.predict_risk(customer_features)
    
    # Update customer history after prediction for continuous history building
    updated_features = risk_model.update_customer_history(request.customer_id, request.model_dump())

    return RiskPredictionResponse(
        customer_id=request.customer_id,
        risk_score=prediction['risk_score'],
        risk_level=prediction['risk_level'],
        risk_probability=prediction['risk_probability'],
        prediction_method=prediction['prediction_method'],
        model_version=prediction['model_version'],
        recommendation=prediction['recommendation'],
        customer_history=customer_features
    )


@app.post("/predict-new-customer", response_model=NewCustomerResponse)
def predict_new_customer(request: ReturnRequest):
    """Explicit endpoint to handle brand-new customer return requests and initialize history."""
    default_features = risk_model.get_default_features(request.customer_id)
    prediction = risk_model.predict_risk(default_features)
    
    # Register customer and collect history
    risk_model.update_customer_history(request.customer_id, request.model_dump())

    return NewCustomerResponse(
        customer_id=request.customer_id,
        risk_score=prediction['risk_score'],
        risk_level=prediction['risk_level'],
        prediction_method='new_customer_default',
        message="New customer record created. Default features applied for initial score.",
        recommendation="Apply standard return policy. Starting customer history collection."
    )


@app.post("/update-history")
def update_history(request: ReturnRequest):
    """Updates customer transaction and return history in the database."""
    updated_features = risk_model.update_customer_history(request.customer_id, request.model_dump())
    return {
        "status": "success",
        "customer_id": request.customer_id,
        "message": "Customer history updated successfully",
        "current_total_orders": updated_features.get('total_orders'),
        "current_return_ratio": updated_features.get('return_ratio')
    }


@app.get("/customer/{customer_id}")
def get_customer_profile(customer_id: str):
    """Lookup customer profile and current risk score evaluation."""
    features = risk_model.get_customer_features(customer_id)
    if features is None:
        return {
            "customer_id": customer_id,
            "status": "new_customer",
            "features": risk_model.get_default_features(customer_id),
            "risk_assessment": risk_model.predict_risk(risk_model.get_default_features(customer_id))
        }

    prediction = risk_model.predict_risk(features)
    return {
        "customer_id": customer_id,
        "status": "existing_customer",
        "features": features,
        "risk_assessment": prediction
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)