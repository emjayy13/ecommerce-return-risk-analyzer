import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, r2_score

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.dirname(SRC_DIR)
data_path = os.path.join(BASE_DIR, 'data', 'customer_features.csv')
model_path = os.path.join(BASE_DIR, 'models', 'return_risk_model.pkl')

df = pd.read_csv(data_path)

X = df[['most_common_category', 'total_orders', 'return_ratio', 'avg_return_window', 'vague_reason_count', 'mismatch_flag_history']].copy()
X['mismatch_flag_history'] = X['mismatch_flag_history'].astype(int)
y = df['risk_score_label']

# Split 80/20 train/test
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

# Load trained model
model = joblib.load(model_path)
preds = model.predict(X_te)

# Regression metrics
mae = mean_absolute_error(y_te, preds)
r2 = r2_score(y_te, preds)

# Risk Level Classification metrics
def bucket(s):
    if s <= 40:
        return 'Low'
    elif s <= 70:
        return 'Medium'
    return 'High'

y_te_b = y_te.apply(bucket)
preds_b = [bucket(s) for s in preds]
acc = accuracy_score(y_te_b, preds_b)

print("=========================================")
print("MODEL EVALUATION & ACCURACY METRICS")
print("=========================================")
print(f"Total Test Samples Evaluated: {len(X_te)} customer profiles")
print(f"Mean Absolute Error (MAE): {mae:.2f} (out of 100 max risk score)")
print(f"R² Score (Variance Explained): {r2:.4f} ({r2*100:.2f}%)")
print(f"Risk Bucket Classification Accuracy: {acc*100:.2f}%\n")
print("Detailed Classification Report (Low / Medium / High Risk):")
print(classification_report(y_te_b, preds_b))
print("=========================================")
