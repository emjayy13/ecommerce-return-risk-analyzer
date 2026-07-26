import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

CATEGORICAL_COLS = ['most_common_category']
NUMERIC_COLS = ['total_orders', 'return_ratio', 'avg_return_window', 'vague_reason_count']
BOOLEAN_COLS = ['mismatch_flag_history']


def train_model(data_path: str, model_output_path: str):
    df = pd.read_csv(data_path)
    print(f"Loaded customer-level data: {df.shape}  <-- must be ~1385 rows")

    X = df[CATEGORICAL_COLS + NUMERIC_COLS + BOOLEAN_COLS].copy()
    X['mismatch_flag_history'] = X['mismatch_flag_history'].astype(int)
    y = df['risk_score_label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer(transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_COLS),
        ('num', StandardScaler(), NUMERIC_COLS + BOOLEAN_COLS)
    ])

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, max_depth=3, random_state=42))
    ])

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    print("\n--- MODEL PERFORMANCE ---")
    print(f"MAE: {mean_absolute_error(y_test, preds):.2f}")
    print(f"R2 Score: {r2_score(y_test, preds):.3f}")

    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(pipeline, model_output_path)
    print(f"Model saved to: {model_output_path}")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_csv = os.path.join(base_dir, 'data', 'customer_features.csv')
    model_path = os.path.join(base_dir, 'models', 'return_risk_model.pkl')
    train_model(data_csv, model_path)
