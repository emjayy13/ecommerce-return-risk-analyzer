import os
import json
import time
import joblib
import logging
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
VERSION_DIR = os.path.join(MODEL_DIR, 'versions')

NUMERIC_FEATURES = [
    'total_orders', 'total_returns', 'return_ratio', 'avg_return_window',
    'product_category_risk', 'vague_reason_count', 'mismatch_flag_history',
    'customer_rating_behavior', 'previous_fraud_flags', 'account_age_days'
]
CATEGORICAL_FEATURES = ['most_common_category']
TARGET_COLUMN = 'flagged_by_company'


class CustomerRiskModel:
    """
    Customer Return Risk Model for predicting fraud/abuse risk scores (0–100)
    for e-commerce customers based on historical return and entity attributes.
    Uses independent ground-truth target labels to prevent data leakage.
    """

    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.version = None
        self.metrics = {}
        self.cv_results = {}
        self.customer_db = {}
        self.feature_names = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    def get_default_features(self, customer_id: str = None) -> dict:
        """Sensible default baseline features for a brand-new customer with no history."""
        return {
            'Customer_ID': customer_id or f"NEW_{int(time.time())}",
            'total_orders': 0,
            'total_returns': 0,
            'return_ratio': 0.0,
            'avg_return_window': 0.0,
            'product_category_risk': 0.0,
            'vague_reason_count': 0,
            'mismatch_flag_history': 0,
            'customer_rating_behavior': 5.0,
            'previous_fraud_flags': 0,
            'account_age_days': 0,
            'most_common_category': 'Unknown'
        }

    def prepare_data(self, customer_df: pd.DataFrame):
        """Prepares feature matrix X and target y."""
        X_df = customer_df[self.feature_names].copy()
        X_df['mismatch_flag_history'] = X_df['mismatch_flag_history'].astype(int)

        if TARGET_COLUMN in customer_df.columns:
            y = customer_df[TARGET_COLUMN].astype(int).values
        elif 'is_fraud' in customer_df.columns:
            y = customer_df['is_fraud'].astype(int).values
        else:
            raise KeyError(f"Target column '{TARGET_COLUMN}' not found in dataframe.")

        return X_df, y

    def train(self, customer_df: pd.DataFrame):
        """
        Trains and evaluates candidate classifiers using 5-fold Cross-Validation
        and a stratified test set. Selects the best performing model based on ROC-AUC.
        """
        logger.info("Preparing data for training Customer Return Risk Model...")
        X_df, y = self.prepare_data(customer_df)

        # Preprocessing Pipeline
        self.preprocessor = ColumnTransformer(transformers=[
            ('num', StandardScaler(), NUMERIC_FEATURES),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_FEATURES)
        ])

        # Train/test split with stratification
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X_df, y, test_size=0.2, random_state=42, stratify=y
        )

        X_train = self.preprocessor.fit_transform(X_train_raw)
        X_test = self.preprocessor.transform(X_test_raw)

        candidate_models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=200, max_depth=6, random_state=42, class_weight='balanced', n_jobs=-1
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=150, learning_rate=0.05, max_depth=3, random_state=42
            ),
            'LogisticRegression': LogisticRegression(
                max_iter=1000, class_weight='balanced', random_state=42, C=1.0
            )
        }

        results = {}
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for name, clf in candidate_models.items():
            start_time = time.time()
            
            # Cross-validation scores on training set
            cv_metrics = cross_validate(
                clf, X_train, y_train, cv=skf,
                scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
            )

            # Fit on full training set
            clf.fit(X_train, y_train)
            train_duration = round(time.time() - start_time, 3)

            # Evaluate on unseen holdout test set
            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)[:, 1]

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            roc_auc = roc_auc_score(y_test, y_prob)
            pr_auc = average_precision_score(y_test, y_prob)
            cm = confusion_matrix(y_test, y_pred).tolist()

            results[name] = {
                'model': clf,
                'accuracy': round(float(acc), 4),
                'precision': round(float(prec), 4),
                'recall': round(float(rec), 4),
                'f1_score': round(float(f1), 4),
                'roc_auc': round(float(roc_auc), 4),
                'pr_auc': round(float(pr_auc), 4),
                'confusion_matrix': cm,
                'cv_scores': {
                    'mean_accuracy': round(float(np.mean(cv_metrics['test_accuracy'])), 4),
                    'mean_precision': round(float(np.mean(cv_metrics['test_precision'])), 4),
                    'mean_recall': round(float(np.mean(cv_metrics['test_recall'])), 4),
                    'mean_f1': round(float(np.mean(cv_metrics['test_f1'])), 4),
                    'mean_roc_auc': round(float(np.mean(cv_metrics['test_roc_auc'])), 4)
                },
                'train_duration_sec': train_duration
            }

            logger.info(
                f"  {name:18s} | Test Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | "
                f"F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f} | CV ROC-AUC: {results[name]['cv_scores']['mean_roc_auc']:.4f}"
            )

        # Select best model by ROC-AUC
        best_name = max(results, key=lambda k: results[k]['roc_auc'])
        self.model = results[best_name]['model']
        self.version = f"v3_{best_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.metrics = {
            'best_model_name': best_name,
            'accuracy': results[best_name]['accuracy'],
            'precision': results[best_name]['precision'],
            'recall': results[best_name]['recall'],
            'f1_score': results[best_name]['f1_score'],
            'roc_auc': results[best_name]['roc_auc'],
            'pr_auc': results[best_name]['pr_auc'],
            'confusion_matrix': results[best_name]['confusion_matrix'],
            'cv_scores': results[best_name]['cv_scores']
        }
        self.cv_results = {name: res['cv_scores'] for name, res in results.items()}

        logger.info(f"Selected Best Model: {best_name} (ROC-AUC: {self.metrics['roc_auc']})")

        self._build_customer_db(customer_df)
        self._save_model()

        return results

    def predict_risk(self, customer_features: dict) -> dict:
        """
        Generates risk prediction for a customer given historical features.
        Returns risk_score (0–100), risk_level ('Low'/'Medium'/'High'), and recommendation.
        """
        # Fallback to defaults if missing fields
        full_features = self.get_default_features()
        full_features.update(customer_features)

        if self.model is None or self.preprocessor is None:
            return self._rule_based_fallback(full_features)

        input_df = pd.DataFrame([{
            'total_orders': full_features.get('total_orders', 0),
            'total_returns': full_features.get('total_returns', 0),
            'return_ratio': float(full_features.get('return_ratio', 0.0)),
            'avg_return_window': float(full_features.get('avg_return_window', 0.0)),
            'product_category_risk': float(full_features.get('product_category_risk', 0.0)),
            'vague_reason_count': int(full_features.get('vague_reason_count', 0)),
            'mismatch_flag_history': int(full_features.get('mismatch_flag_history', 0)),
            'customer_rating_behavior': float(full_features.get('customer_rating_behavior', 5.0)),
            'previous_fraud_flags': int(full_features.get('previous_fraud_flags', 0)),
            'account_age_days': int(full_features.get('account_age_days', 0)),
            'most_common_category': str(full_features.get('most_common_category', 'Unknown'))
        }])

        try:
            X_trans = self.preprocessor.transform(input_df)
            prob = self.model.predict_proba(X_trans)[0][1]
            risk_score = round(float(prob * 100), 2)
            risk_score = max(0.0, min(100.0, risk_score))
            risk_level = self._score_to_level(risk_score)
            method = 'ml_model'
        except Exception as e:
            logger.warning(f"Model prediction failed ({e}). Using rule-based fallback.")
            return self._rule_based_fallback(full_features)

        return {
            'risk_score': risk_score,
            'risk_level': risk_level,
            'risk_probability': round(float(prob), 4),
            'prediction_method': method,
            'model_version': self.version or 'v3_production',
            'recommendation': self._get_recommendation(risk_level)
        }

    def _rule_based_fallback(self, features: dict) -> dict:
        """Rule-based risk calculation fallback for unseen/new customers."""
        score = 10.0  # Base low risk score for new customer
        if features.get('previous_fraud_flags', 0) > 0:
            score += 30.0 * features['previous_fraud_flags']
        if features.get('return_ratio', 0.0) > 0.3:
            score += 25.0
        if features.get('vague_reason_count', 0) >= 2:
            score += 15.0
        if features.get('mismatch_flag_history', 0) > 0:
            score += 15.0
        
        score = round(float(min(score, 100.0)), 2)
        risk_level = self._score_to_level(score)

        return {
            'risk_score': score,
            'risk_level': risk_level,
            'risk_probability': round(score / 100.0, 4),
            'prediction_method': 'rule_based_fallback',
            'model_version': 'fallback_v1',
            'recommendation': self._get_recommendation(risk_level)
        }

    @staticmethod
    def _score_to_level(score: float) -> str:
        if score < 35.0:
            return 'Low'
        elif score < 70.0:
            return 'Medium'
        else:
            return 'High'

    @staticmethod
    def _get_recommendation(risk_level: str) -> str:
        if risk_level == 'Low':
            return 'Standard processing - Low risk customer. Instant return authorization approved.'
        elif risk_level == 'Medium':
            return 'Monitor returns - Require specific item verification before return authorization.'
        else:
            return 'High risk - Flag for manual review by fraud mitigation team before return authorization.'

    def _build_customer_db(self, customer_df: pd.DataFrame):
        """Populates in-memory customer feature lookup database."""
        for _, row in customer_df.iterrows():
            cid = row['Customer_ID']
            self.customer_db[cid] = {
                'Customer_ID': cid,
                'total_orders': int(row['total_orders']),
                'total_returns': int(row['total_returns']),
                'return_ratio': float(row['return_ratio']),
                'avg_return_window': float(row['avg_return_window']),
                'product_category_risk': float(row['product_category_risk']),
                'vague_reason_count': int(row['vague_reason_count']),
                'mismatch_flag_history': int(row['mismatch_flag_history']),
                'customer_rating_behavior': float(row['customer_rating_behavior']),
                'previous_fraud_flags': int(row['previous_fraud_flags']),
                'account_age_days': int(row['account_age_days']),
                'most_common_category': str(row['most_common_category']),
                'flagged_by_company': int(row['flagged_by_company'])
            }
        logger.info(f"Built customer database lookup with {len(self.customer_db)} customers.")

    def get_customer_features(self, customer_id: str) -> dict:
        """Retrieves customer record from database lookup. Returns None if customer is new."""
        return self.customer_db.get(customer_id, None)

    def update_customer_history(self, customer_id: str, new_event: dict) -> dict:
        """
        Updates customer history when a new purchase/return request occurs,
        allowing future predictions to leverage updated behavior.
        """
        if customer_id not in self.customer_db:
            self.customer_db[customer_id] = self.get_default_features(customer_id)

        cust = self.customer_db[customer_id]
        cust['total_orders'] += 1
        
        if new_event.get('is_returned', False):
            cust['total_returns'] += 1
            reason = new_event.get('return_reason', '')
            if reason in ['Defective', 'Not as described']:
                cust['vague_reason_count'] += 1
            if reason == 'Wrong item':
                cust['mismatch_flag_history'] = 1

        cust['return_ratio'] = round(cust['total_returns'] / cust['total_orders'], 4)
        category = new_event.get('product_category')
        if category:
            cust['most_common_category'] = category

        return cust

    def _save_model(self):
        """Saves model binary and metadata JSON artifacts."""
        os.makedirs(VERSION_DIR, exist_ok=True)
        os.makedirs(MODEL_DIR, exist_ok=True)

        model_path = os.path.join(MODEL_DIR, 'customer_risk_model.pkl')
        joblib.dump({
            'model': self.model,
            'preprocessor': self.preprocessor,
            'version': self.version,
            'metrics': self.metrics,
            'cv_results': self.cv_results,
            'customer_db': self.customer_db,
            'numeric_features': NUMERIC_FEATURES,
            'categorical_features': CATEGORICAL_FEATURES
        }, model_path)

        meta_path = os.path.join(MODEL_DIR, 'customer_risk_metadata.json')
        with open(meta_path, 'w') as f:
            json.dump({
                'version': self.version,
                'updated_at': datetime.now().isoformat(),
                'metrics': self.metrics,
                'cv_results': self.cv_results,
                'features': NUMERIC_FEATURES + CATEGORICAL_FEATURES,
                'total_customers_in_db': len(self.customer_db)
            }, f, indent=2)

        logger.info(f"Successfully saved Customer Risk Model to {model_path} and metadata to {meta_path}")

    def load_model(self) -> bool:
        """Loads saved model binary and metadata."""
        model_path = os.path.join(MODEL_DIR, 'customer_risk_model.pkl')
        if os.path.exists(model_path):
            try:
                data = joblib.load(model_path)
                self.model = data['model']
                self.preprocessor = data['preprocessor']
                self.version = data['version']
                self.metrics = data['metrics']
                self.cv_results = data.get('cv_results', {})
                self.customer_db = data.get('customer_db', {})
                logger.info(f"Loaded Customer Risk Model ({self.version}) with {len(self.customer_db)} customers.")
                return True
            except Exception as e:
                logger.error(f"Failed to load model from {model_path}: {e}")
                return False
        return False
