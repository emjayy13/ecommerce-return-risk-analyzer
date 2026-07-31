import os
import sys
import pandas as pd
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field, ConfigDict

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from customer_risk_model import CustomerRiskModel

app = FastAPI(
    title="Customer Return Risk Analyzer API - Continuous Learning v2",
    description="API for predicting customer return risk scores (0–100) and managing continuous learning retraining pipelines.",
    version="3.1.0"
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

risk_model = CustomerRiskModel(retrain_threshold=10)


def load_or_train_model():
    """Loads active production model or runs initial training if missing."""
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


class FeedbackRequest(BaseModel):
    customer_id: str = Field(..., description="Customer ID", example="CUST000063")
    actual_fraud_label: int = Field(..., description="Actual ground-truth audit label (1 = Fraud/Abuse, 0 = Legitimate)", example=1)
    notes: Optional[str] = Field("Manual audit outcome", description="Audit investigation notes", example="Company audit confirmed synthetic item switch")

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "customer_id": "CUST000063",
            "actual_fraud_label": 1,
            "notes": "Verified fraud after manual return inspection"
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
        "service": "Customer Return Risk Analyzer API - Continuous Learning v2",
        "version": "3.1.0",
        "docs_url": "/docs"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": risk_model.model is not None,
        "model_version": risk_model.version,
        "total_customers_in_db": len(risk_model.customer_db),
        "unprocessed_feedback_count": risk_model.unprocessed_feedback_count,
        "retrain_threshold": risk_model.retrain_threshold
    }


@app.get("/model/info")
def get_model_info():
    """Returns active model details, version, feedback stats, and current metrics."""
    if risk_model.model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    return {
        "active_model_version": risk_model.version,
        "metrics": risk_model.metrics,
        "total_customers_in_db": len(risk_model.customer_db),
        "unprocessed_feedback_count": risk_model.unprocessed_feedback_count,
        "retrain_threshold": risk_model.retrain_threshold,
        "retraining_runs_count": len(risk_model.get_retraining_history())
    }


@app.get("/model/metrics")
def get_model_metrics():
    """Returns detailed evaluation metrics and cross-validation breakdown."""
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
        default_features = risk_model.get_default_features(request.customer_id)
        prediction = risk_model.predict_risk(default_features)
        prediction['prediction_method'] = 'new_customer_baseline'
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

    prediction = risk_model.predict_risk(customer_features)
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
    risk_model.update_customer_history(request.customer_id, request.model_dump())

    return NewCustomerResponse(
        customer_id=request.customer_id,
        risk_score=prediction['risk_score'],
        risk_level=prediction['risk_level'],
        prediction_method='new_customer_default',
        message="New customer record created. Default features applied for initial score.",
        recommendation="Apply standard return policy. Starting customer history collection."
    )


@app.post("/feedback")
def submit_ground_truth_feedback(feedback: FeedbackRequest):
    """
    Submits verified ground-truth audit label for a customer return request.
    Appends to historical dataset and triggers automated retraining if threshold is reached.
    """
    if risk_model.model is None:
        load_or_train_model()

    result = risk_model.record_ground_truth_feedback(
        customer_id=feedback.customer_id,
        actual_fraud_label=feedback.actual_fraud_label,
        notes=feedback.notes
    )
    return result


@app.post("/retrain")
def trigger_retraining(force: bool = True):
    """
    Manually triggers automated retraining & Champion vs. Challenger model evaluation.
    Promotes challenger model only if it outperforms the active production champion.
    """
    if risk_model.model is None:
        load_or_train_model()

    audit_entry = risk_model.retrain_pipeline(force=force, reason="Triggered via POST /retrain")
    return {
        "status": "completed",
        "retraining_audit": audit_entry,
        "active_model_version": risk_model.version
    }


@app.get("/retrain/history")
def get_retraining_history():
    """Returns complete audit log history of all past continuous retraining runs."""
    return {
        "total_retrain_runs": len(risk_model.get_retraining_history()),
        "retraining_history": risk_model.get_retraining_history()
    }


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