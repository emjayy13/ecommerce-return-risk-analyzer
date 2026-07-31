import sys
import os
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from predict import app, load_or_train_model


def test_api():
    load_or_train_model()
    
    with TestClient(app) as client:
        print("=" * 60)
        print("  TESTING FASTAPI CUSTOMER RETURN RISK ANALYZER ENDPOINTS")
        print("=" * 60)

        # 1. Health check
        print("\n1. GET /health")
        res = client.get("/health")
        print("Status:", res.status_code)
        print("Response:", res.json())
        assert res.status_code == 200
        assert res.json()["model_loaded"] is True

        # 2. Model Metrics
        print("\n2. GET /model/metrics")
        res = client.get("/model/metrics")
        print("Status:", res.status_code)
        metrics_data = res.json()
        print(f"Version: {metrics_data.get('version')}")
        print(f"Metrics summary: Accuracy={metrics_data.get('metrics', {}).get('accuracy')}, ROC-AUC={metrics_data.get('metrics', {}).get('roc_auc')}")
        assert res.status_code == 200

        # 3. Existing customer prediction
        print("\n3. POST /predict (Existing Customer)")
        req_payload = {
            "customer_id": "CUST000063",
            "order_id": "ORD00007551",
            "product_category": "Clothing",
            "return_reason": "Defective",
            "is_returned": True
        }
        res = client.post("/predict", json=req_payload)
        print("Status:", res.status_code)
        print("Response:", json.dumps(res.json(), indent=2))
        assert res.status_code == 200
        pred = res.json()
        assert 0 <= pred["risk_score"] <= 100
        assert pred["risk_level"] in ["Low", "Medium", "High"]

        # 4. New customer prediction
        print("\n4. POST /predict (Brand-New Customer)")
        new_payload = {
            "customer_id": "CUST_BRAND_NEW_9999",
            "order_id": "ORD_NEW_001",
            "product_category": "Electronics",
            "return_reason": "Changed mind",
            "is_returned": True
        }
        res = client.post("/predict", json=new_payload)
        print("Status:", res.status_code)
        print("Response:", json.dumps(res.json(), indent=2))
        assert res.status_code == 200
        new_pred = res.json()
        assert new_pred["prediction_method"] == "new_customer_baseline"

        # 5. Customer Profile Lookup
        print("\n5. GET /customer/CUST_BRAND_NEW_9999")
        res = client.get("/customer/CUST_BRAND_NEW_9999")
        print("Status:", res.status_code)
        print("Response:", json.dumps(res.json(), indent=2))
        assert res.status_code == 200
        assert res.json()["status"] == "existing_customer"
        assert res.json()["features"]["total_orders"] >= 1

        print("\n" + "=" * 60)
        print("  ALL API ENDPOINT TESTS PASSED SUCCESSFULLY!")
        print("=" * 60)


if __name__ == "__main__":
    test_api()
