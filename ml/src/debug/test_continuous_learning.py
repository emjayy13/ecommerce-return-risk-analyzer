import sys
import os
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from predict import app, load_or_train_model, risk_model


def test_continuous_learning():
    print("=" * 70)
    print("  TESTING CONTINUOUS LEARNING V2 PIPELINE & FASTAPI ENDPOINTS")
    print("=" * 70)

    load_or_train_model()

    with TestClient(app) as client:
        # 1. Health & Model Info
        print("\n1. GET /health & GET /model/info")
        res_health = client.get("/health")
        print("Health Response:", res_health.json())
        assert res_health.status_code == 200
        assert res_health.json()["model_loaded"] is True

        res_info = client.get("/model/info")
        print("Model Info Response:", res_info.json())
        assert res_info.status_code == 200

        # 2. Existing Customer Prediction
        print("\n2. POST /predict (Existing Customer)")
        res_pred = client.post("/predict", json={
            "customer_id": "CUST000063",
            "order_id": "ORD00007551",
            "product_category": "Clothing",
            "return_reason": "Defective",
            "is_returned": True
        })
        print("Prediction Response:", json.dumps(res_pred.json(), indent=2))
        assert res_pred.status_code == 200

        # 3. Submit Ground-Truth Feedback Records (Simulating 10 feedback items to hit threshold)
        print("\n3. Submitting 10 Verified Ground-Truth Feedback Items (Threshold Trigger)...")
        feedback_results = []
        for i in range(1, 11):
            fb_payload = {
                "customer_id": f"CUST_FB_{str(i).zfill(3)}",
                "actual_fraud_label": 1 if i % 3 == 0 else 0,
                "notes": f"Manual company audit check #{i}"
            }
            res_fb = client.post("/feedback", json=fb_payload)
            assert res_fb.status_code == 200
            fb_data = res_fb.json()
            feedback_results.append(fb_data)
            print(f"  Feedback #{i}: Customer={fb_payload['customer_id']} | Unprocessed Count={fb_data['unprocessed_feedback_count']}/10 | Retrained={fb_data['retrain_triggered']}")

        last_fb = feedback_results[-1]
        assert last_fb["retrain_triggered"] is True
        print("\nAutomated threshold retraining triggered successfully!")
        print("Retrain Result Audit:", json.dumps(last_fb["retrain_result"], indent=2))

        # 4. Manual Retrain Trigger
        print("\n4. POST /retrain (Manual Trigger Test)")
        res_retrain = client.post("/retrain", params={"force": True})
        print("Manual Retrain Response:", json.dumps(res_retrain.json(), indent=2))
        assert res_retrain.status_code == 200

        # 5. GET /retrain/history
        print("\n5. GET /retrain/history")
        res_hist = client.get("/retrain/history")
        print("Retrain History Response:", json.dumps(res_hist.json(), indent=2))
        assert res_hist.status_code == 200
        assert res_hist.json()["total_retrain_runs"] >= 2

        # 6. Verify Model Version Binaries on Disk
        # Correct path for ml/models/versions
        src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ml_dir = os.path.dirname(src_dir)
        versions_dir = os.path.join(ml_dir, 'models', 'versions')
        version_files = os.listdir(versions_dir) if os.path.exists(versions_dir) else []
        print(f"\n6. Saved Version Files in {versions_dir}: {version_files}")
        assert len(version_files) > 0

        print("\n" + "=" * 70)
        print("  ALL CONTINUOUS LEARNING V2 TESTS PASSED SUCCESSFULLY!")
        print("=" * 70)


if __name__ == "__main__":
    test_continuous_learning()
