import os
import json
import joblib

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.dirname(SRC_DIR)
model_path = os.path.join(BASE_DIR, 'models', 'customer_risk_model.pkl')
metadata_path = os.path.join(BASE_DIR, 'models', 'customer_risk_metadata.json')

if not os.path.exists(model_path):
    print("Model not found. Run: python ml/src/train.py")
    exit(1)

data = joblib.load(model_path)

print("=========================================")
print("MODEL EVALUATION SUMMARY")
print("=========================================")
print(f"Version       : {data.get('version', 'N/A')}")
print(f"Best Model    : {data.get('metrics', {}).get('best_model_name', 'N/A')}")
print(f"Accuracy      : {data.get('metrics', {}).get('accuracy', 'N/A')}")
print(f"Precision     : {data.get('metrics', {}).get('precision', 'N/A')}")
print(f"Recall        : {data.get('metrics', {}).get('recall', 'N/A')}")
print(f"F1-Score      : {data.get('metrics', {}).get('f1_score', 'N/A')}")
print(f"ROC-AUC       : {data.get('metrics', {}).get('roc_auc', 'N/A')}")
print(f"PR-AUC        : {data.get('metrics', {}).get('pr_auc', 'N/A')}")
print(f"Customers     : {len(data.get('customer_db', {}))}")
print(f"Confusion Mat : {data.get('metrics', {}).get('confusion_matrix', 'N/A')}")
print("=========================================")
