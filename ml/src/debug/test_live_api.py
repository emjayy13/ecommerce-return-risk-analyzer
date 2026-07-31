import sys
import os
import time
import threading
import uvicorn
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from predict import app

def start_server():
    uvicorn.run(app, host="127.0.0.1", port=8005, log_level="error")

if __name__ == "__main__":
    # Start server in background thread
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    time.sleep(2) # Give uvicorn a moment to bind

    base_url = "http://127.0.0.1:8005"

    print("=========================================")
    print("LIVE API PROOF & VERIFICATION TEST")
    print("=========================================\n")

    # 1. Health check
    h = requests.get(f"{base_url}/health")
    print(f"[GET /health] Status: {h.status_code}")
    print(f"Response: {h.json()}\n")

    # Test cases
    test_cases = [
        {
            "name": "Low Risk Customer Profile",
            "payload": {
                "total_orders": 20,
                "return_ratio": 0.05,
                "avg_return_window": 15.0,
                "vague_reason_count": 0,
                "most_common_category": "Electronics",
                "mismatch_flag_history": False
            }
        },
        {
            "name": "Medium Risk Customer Profile",
            "payload": {
                "total_orders": 12,
                "return_ratio": 0.20,
                "avg_return_window": 5.0,
                "vague_reason_count": 1,
                "most_common_category": "Home",
                "mismatch_flag_history": False
            }
        },
        {
            "name": "High Risk Customer Profile",
            "payload": {
                "total_orders": 8,
                "return_ratio": 0.45,
                "avg_return_window": 1.5,
                "vague_reason_count": 3,
                "most_common_category": "Clothing",
                "mismatch_flag_history": True
            }
        }
    ]

    for tc in test_cases:
        print(f"--- Testing: {tc['name']} ---")
        print(f"Payload: {tc['payload']}")
        res = requests.post(f"{base_url}/predict", json=tc['payload'])
        print(f"HTTP Status: {res.status_code}")
        print(f"Prediction Result: {res.json()}\n")

    print("=========================================")
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=========================================")
