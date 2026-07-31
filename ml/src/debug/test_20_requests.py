import sys
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from predict import app, load_or_train_model


def run_batch_20_requests_test():
    print("=" * 105)
    print("  RUNNING LIVE BATCH TEST: 20 CUSTOMER RETURN REQUESTS & METRIC EVALUATION")
    print("=" * 105)

    load_or_train_model()

    # Determine correct path for ml/data/customer_features.csv
    src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ml_dir = os.path.dirname(src_dir)
    features_csv = os.path.join(ml_dir, 'data', 'customer_features.csv')
    df = pd.read_csv(features_csv)

    # Pick 20 representative customers (mix of normal and flagged fraud customers)
    sample_fraud = df[df['flagged_by_company'] == 1].sample(n=6, random_state=42)
    sample_normal = df[df['flagged_by_company'] == 0].sample(n=14, random_state=42)
    sample_20 = pd.concat([sample_fraud, sample_normal]).sample(frac=1.0, random_state=42).reset_index(drop=True)

    categories = ['Clothing', 'Electronics', 'Shoes', 'Fashion', 'Books', 'Home', 'Toys']
    reasons = ['Defective', 'Not as described', 'Wrong item', 'Changed mind', 'Size issue']

    results_table = []
    y_true = []
    y_pred_binary = []
    y_pred_probs = []

    with TestClient(app) as client:
        for idx, row in sample_20.iterrows():
            req_num = idx + 1
            cid = row['Customer_ID']
            actual_label = int(row['flagged_by_company'])
            category = str(row['most_common_category']) if row['most_common_category'] != 'Unknown' else np.random.choice(categories)
            reason = np.random.choice(reasons)

            payload = {
                "customer_id": cid,
                "order_id": f"ORD_TEST_{req_num:03d}",
                "product_category": category,
                "return_reason": reason,
                "is_returned": True
            }

            response = client.post("/predict", json=payload)
            assert response.status_code == 200
            res_data = response.json()

            risk_score = res_data['risk_score']
            risk_level = res_data['risk_level']
            prob = res_data['risk_probability']

            # Classification decision threshold (prob >= 0.35 -> High/Medium Risk = Fraud prediction)
            pred_binary = 1 if prob >= 0.35 else 0
            is_correct = (pred_binary == actual_label)

            y_true.append(actual_label)
            y_pred_binary.append(pred_binary)
            y_pred_probs.append(prob)

            results_table.append({
                "Req": req_num,
                "Customer_ID": cid,
                "Category": category,
                "Reason": reason,
                "Actual": "Fraud" if actual_label == 1 else "Normal",
                "Risk_Score": risk_score,
                "Risk_Level": risk_level,
                "Predicted": "Fraud" if pred_binary == 1 else "Normal",
                "Match": "PASS" if is_correct else "FAIL"
            })

    acc = accuracy_score(y_true, y_pred_binary) * 100.0
    prec = precision_score(y_true, y_pred_binary, zero_division=0) * 100.0
    rec = recall_score(y_true, y_pred_binary, zero_division=0) * 100.0
    f1 = f1_score(y_true, y_pred_binary, zero_division=0) * 100.0
    roc_auc = roc_auc_score(y_true, y_pred_probs) if len(set(y_true)) > 1 else 1.0
    cm = confusion_matrix(y_true, y_pred_binary)

    print("\n" + "-" * 105)
    print(f"{'#':<4} {'Customer ID':<13} {'Category':<13} {'Reason':<18} {'Actual':<8} {'Score':<8} {'Level':<8} {'Predicted':<9} {'Status':<6}")
    print("-" * 105)
    for r in results_table:
        print(f"{r['Req']:<4} {r['Customer_ID']:<13} {r['Category']:<13} {r['Reason']:<18} {r['Actual']:<8} {r['Risk_Score']:<8.2f} {r['Risk_Level']:<8} {r['Predicted']:<9} {r['Match']:<6}")
    print("-" * 105)

    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    print("\n" + "=" * 105)
    print("  TOTAL 20-REQUEST BATCH EVALUATION METRICS")
    print("=" * 105)
    print(f"  - Total Requests Tested   : 20")
    print(f"  - Total Accuracy           : {acc:.2f}% ({sum(np.array(y_true) == np.array(y_pred_binary))}/20 Correct)")
    print(f"  - Precision                : {prec:.2f}%")
    print(f"  - Recall                   : {rec:.2f}%")
    print(f"  - F1-Score                 : {f1:.2f}%")
    print(f"  - ROC-AUC Score            : {roc_auc:.4f}")
    print(f"  - Confusion Matrix Breakdown:")
    print(f"      - True Negatives  (Normal correctly identified) : {tn}")
    print(f"      - True Positives  (Fraud correctly identified)  : {tp}")
    print(f"      - False Positives (Normal flagged as Fraud)    : {fp}")
    print(f"      - False Negatives (Fraud missed as Normal)     : {fn}")

    levels = [r['Risk_Level'] for r in results_table]
    print("\n  - Risk Level Breakdown:")
    print(f"      - Low Risk    : {levels.count('Low')} requests")
    print(f"      - Medium Risk : {levels.count('Medium')} requests")
    print(f"      - High Risk   : {levels.count('High')} requests")
    print("=" * 105)

    eval_json_path = os.path.join(ml_dir, 'models', 'batch_20_test_results.json')
    with open(eval_json_path, 'w') as f:
        json.dump({
            "total_requests": 20,
            "overall_accuracy_percentage": round(acc, 2),
            "precision_percentage": round(prec, 2),
            "recall_percentage": round(rec, 2),
            "f1_score_percentage": round(f1, 2),
            "roc_auc_score": round(roc_auc, 4),
            "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
            "detailed_requests": results_table
        }, f, indent=2)
    print(f"\nDetailed evaluation results exported to: {eval_json_path}\n")


if __name__ == "__main__":
    run_batch_20_requests_test()
