import os
import json
import time
import joblib
import logging
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report
)
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
VERSION_DIR = os.path.join(MODEL_DIR, 'versions')

CUSTOMER_FEATURES = [
    'total_orders', 'return_ratio', 'avg_return_window',
    'vague_reason_count', 'mismatch_flag_history'
]
CUSTOMER_CAT_FEATURES = ['most_common_category']

RULE_THRESHOLDS = {
    'return_ratio_high': 0.3,
    'return_ratio_medium': 0.15,
    'vague_reason_high': 2,
    'vague_reason_medium': 1,
    'fast_return_days': 2
}


class CustomerRiskModel:
    """Customer-level return risk prediction model with continuous learning."""

    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.version = None
        self.metrics = {}
        self.customer_db = {}

    def build_features_from_orders(self, orders_df):
        """Build customer-level features from order-level data."""
        logger.info(f"Building customer features from {len(orders_df)} orders...")

        grouped = orders_df.groupby('Customer_ID').agg(
            total_orders=('Order_ID', 'count'),
            total_returns=('Is_Returned', 'sum'),
            avg_return_window=('Days_to_Return', lambda x: x[x > 0].mean() if (x > 0).any() else 0),
            most_common_category=('Product_Category', lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown'),
        ).reset_index()

        grouped['return_ratio'] = (grouped['total_returns'] / grouped['total_orders']).round(3)
        grouped['avg_return_window'] = grouped['avg_return_window'].fillna(0).round(1)

        vague_reasons = ['Defective', 'Not as described']
        grouped['vague_reason_count'] = orders_df[
            orders_df['Return_Reason'].isin(vague_reasons)
        ].groupby('Customer_ID').size().reindex(grouped['Customer_ID'], fill_value=0).values

        mismatch_reasons = ['Wrong item']
        mismatch_flag = orders_df[
            orders_df['Return_Reason'].isin(mismatch_reasons)
        ].groupby('Customer_ID').size() > 0
        grouped['mismatch_flag_history'] = grouped['Customer_ID'].map(mismatch_flag).fillna(False).astype(int)

        logger.info(f"Customer features built: {grouped.shape[0]} customers")
        return grouped

    def rule_based_score(self, row):
        """Rule-based risk score for new customers or fallback."""
        score = 0
        if row['return_ratio'] > RULE_THRESHOLDS['return_ratio_high']:
            score += 30
        elif row['return_ratio'] > RULE_THRESHOLDS['return_ratio_medium']:
            score += 15

        if 0 < row['avg_return_window'] < RULE_THRESHOLDS['fast_return_days']:
            score += 15

        if row['most_common_category'] in ['Clothing', 'Shoes', 'Fashion']:
            score += 20

        if row['vague_reason_count'] >= RULE_THRESHOLDS['vague_reason_high']:
            score += 25
        elif row['vague_reason_count'] == RULE_THRESHOLDS['vague_reason_medium']:
            score += 10

        if row['mismatch_flag_history']:
            score += 15

        return min(score, 100)

    def get_default_features(self, customer_id=None):
        """Default features for a new customer with no history."""
        return {
            'Customer_ID': customer_id or f"NEW_{int(time.time())}",
            'total_orders': 0,
            'return_ratio': 0.0,
            'avg_return_window': 0.0,
            'vague_reason_count': 0,
            'mismatch_flag_history': 0,
            'most_common_category': 'Unknown'
        }

    def prepare_training_data(self, customer_df):
        """Prepare features and labels for training."""
        all_features = CUSTOMER_FEATURES + CUSTOMER_CAT_FEATURES
        X_num = customer_df[all_features].copy()
        X_num['mismatch_flag_history'] = X_num['mismatch_flag_history'].astype(int)

        preprocessor = ColumnTransformer(transformers=[
            ('num', StandardScaler(), CUSTOMER_FEATURES),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CUSTOMER_CAT_FEATURES)
        ])

        X = preprocessor.fit_transform(X_num)
        y = (customer_df['return_ratio'] > 0.15).astype(int).values

        return X, y, preprocessor

    def train(self, customer_df):
        """Train the customer risk model."""
        logger.info("Training customer risk model...")

        X, y, preprocessor = self.prepare_training_data(customer_df)
        self.preprocessor = preprocessor

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        smote = SMOTE(random_state=42, sampling_strategy=0.5)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        logger.info(f"After SMOTE: {X_train_res.shape[0]} samples (was {X_train.shape[0]})")

        models = {
            'LogisticRegression': LogisticRegression(
                max_iter=1000, class_weight='balanced', random_state=42, C=0.5
            ),
            'RandomForest': RandomForestClassifier(
                n_estimators=200, max_depth=8, class_weight='balanced', random_state=42, n_jobs=-1
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42
            ),
        }

        results = {}
        for name, model in models.items():
            start = time.time()
            model.fit(X_train_res, y_train_res)
            train_time = time.time() - start

            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            results[name] = {
                'model': model,
                'accuracy': accuracy_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'auc_roc': roc_auc_score(y_test, y_prob),
                'pr_auc': average_precision_score(y_test, y_prob),
                'train_time': train_time
            }
            logger.info(f"  {name}: AUC={results[name]['auc_roc']:.4f} F1={results[name]['f1']:.4f}")

        best_name = max(results, key=lambda k: results[k]['auc_roc'])
        self.model = results[best_name]['model']
        self.version = f"{best_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.metrics = results[best_name]

        logger.info(f"Best model: {best_name} (AUC-ROC: {self.metrics['auc_roc']:.4f})")

        self._build_customer_db(customer_df)
        self._save_model()

        return results

    def predict_risk(self, customer_features):
        """Predict risk score for a customer. Returns score 0-100 and level."""
        if self.model is None:
            return self._rule_based_prediction(customer_features)

        X_num = pd.DataFrame([{
            'total_orders': customer_features.get('total_orders', 0),
            'return_ratio': customer_features.get('return_ratio', 0.0),
            'avg_return_window': customer_features.get('avg_return_window', 0.0),
            'vague_reason_count': customer_features.get('vague_reason_count', 0),
            'mismatch_flag_history': int(customer_features.get('mismatch_flag_history', 0)),
            'most_common_category': customer_features.get('most_common_category', 'Unknown')
        }])

        try:
            X = self.preprocessor.transform(X_num)
            prob = self.model.predict_proba(X)[0][1]
            score = round(float(prob * 100), 2)
        except Exception as e:
            logger.warning(f"Model prediction failed: {e}. Using rule-based.")
            return self._rule_based_prediction(customer_features)

        score = max(0, min(100, score))
        level = self._score_to_level(score)

        return {
            'risk_score': score,
            'risk_level': level,
            'return_probability': round(float(prob), 4),
            'model_version': self.version,
            'prediction_method': 'ml_model'
        }

    def _rule_based_prediction(self, features):
        """Rule-based fallback for new customers."""
        score = self.rule_based_score(features)
        return {
            'risk_score': score,
            'risk_level': self._score_to_level(score),
            'return_probability': round(score / 100, 4),
            'model_version': 'rule_based',
            'prediction_method': 'rule_based'
        }

    def _score_to_level(self, score):
        if score < 25:
            return 'Low'
        elif score < 50:
            return 'Medium'
        else:
            return 'High'

    def _build_customer_db(self, customer_df):
        """Build in-memory customer database for quick lookup."""
        for _, row in customer_df.iterrows():
            self.customer_db[row['Customer_ID']] = {
                'total_orders': row['total_orders'],
                'return_ratio': row['return_ratio'],
                'avg_return_window': row['avg_return_window'],
                'vague_reason_count': row['vague_reason_count'],
                'mismatch_flag_history': int(row['mismatch_flag_history']),
                'most_common_category': row['most_common_category']
            }
        logger.info(f"Customer DB built: {len(self.customer_db)} customers")

    def get_customer_features(self, customer_id):
        """Fetch customer features from DB. Returns None if new customer."""
        return self.customer_db.get(customer_id, None)

    def update_customer_history(self, customer_id, new_order):
        """Update customer history after a new order/return."""
        if customer_id in self.customer_db:
            cust = self.customer_db[customer_id]
            cust['total_orders'] += 1
            if new_order.get('is_returned', False):
                cust['total_returns'] = cust.get('total_returns', 0) + 1
                if new_order.get('return_reason') in ['Defective', 'Not as described']:
                    cust['vague_reason_count'] += 1
                if new_order.get('return_reason') == 'Wrong item':
                    cust['mismatch_flag_history'] = 1
            total = cust['total_orders']
            returns = cust.get('total_returns', 0)
            cust['return_ratio'] = round(returns / total, 3) if total > 0 else 0
        else:
            self.customer_db[customer_id] = self.get_default_features(customer_id)
            self.update_customer_history(customer_id, new_order)

    def incremental_train(self, new_customer_df):
        """Update model with new data without full retrain."""
        if self.model is None:
            logger.warning("No model to update. Run train() first.")
            return

        logger.info(f"Incremental update with {len(new_customer_df)} new customers...")

        X_new, y_new, _ = self.prepare_training_data(new_customer_df)

        smote = SMOTE(random_state=42, sampling_strategy=0.5)
        X_res, y_res = smote.fit_resample(X_new, y_new)

        self.model.fit(X_res, y_res)

        self._build_customer_db(new_customer_df)
        logger.info("Incremental update complete")

    def _save_model(self):
        """Save model and metadata."""
        os.makedirs(VERSION_DIR, exist_ok=True)

        model_path = os.path.join(MODEL_DIR, 'customer_risk_model.pkl')
        joblib.dump({
            'model': self.model,
            'preprocessor': self.preprocessor,
            'version': self.version,
            'metrics': self.metrics,
            'customer_db': self.customer_db,
            'customer_features': CUSTOMER_FEATURES,
            'customer_cat_features': CUSTOMER_CAT_FEATURES,
            'rule_thresholds': RULE_THRESHOLDS
        }, model_path)

        version_path = os.path.join(VERSION_DIR, f"{self.version}.pkl")
        joblib.dump(self.model, version_path)

        meta_path = os.path.join(MODEL_DIR, 'customer_risk_model_metadata.json')
        with open(meta_path, 'w') as f:
            json.dump({
                'version': self.version,
                'timestamp': datetime.now().isoformat(),
                'metrics': self.metrics,
                'features': CUSTOMER_FEATURES + CUSTOMER_CAT_FEATURES,
                'total_customers': len(self.customer_db)
            }, f, indent=2, default=str)

        logger.info(f"Model saved: {model_path}")

    def load_model(self):
        """Load saved model."""
        model_path = os.path.join(MODEL_DIR, 'customer_risk_model.pkl')
        if os.path.exists(model_path):
            data = joblib.load(model_path)
            self.model = data['model']
            self.preprocessor = data['preprocessor']
            self.version = data['version']
            self.metrics = data['metrics']
            self.customer_db = data.get('customer_db', {})
            logger.info(f"Model loaded: {self.version} | Customers: {len(self.customer_db)}")
            return True
        return False


def run_training():
    """Full training pipeline."""
    logger.info("="*60)
    logger.info("  CUSTOMER RISK MODEL - TRAINING PIPELINE")
    logger.info("="*60)

    orders_df = pd.read_csv(os.path.join(DATA_DIR, 'order_features.csv'))
    logger.info(f"Loaded {len(orders_df)} orders")

    model = CustomerRiskModel()
    customer_df = model.build_features_from_orders(orders_df)

    results = model.train(customer_df)

    logger.info("\n" + "="*60)
    logger.info("  TRAINING COMPLETE")
    logger.info("="*60)
    logger.info(f"  Model version: {model.version}")
    logger.info(f"  Total customers: {len(model.customer_db)}")
    for name, res in results.items():
        logger.info(f"  {name}: AUC={res['auc_roc']:.4f} F1={res['f1']:.4f}")

    return model


if __name__ == "__main__":
    run_training()
