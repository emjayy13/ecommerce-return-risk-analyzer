import os
import json
import time
import joblib
import logging
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
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
LOG_DIR = os.path.join(BASE_DIR, 'logs')

NUMERIC_FEATURES = [
    'total_orders', 'total_returns', 'return_ratio', 'avg_return_window',
    'product_category_risk', 'vague_reason_count', 'mismatch_flag_history',
    'customer_rating_behavior', 'previous_fraud_flags', 'account_age_days'
]
CATEGORICAL_FEATURES = ['most_common_category']
TARGET_COLUMN = 'flagged_by_company'


class CustomerRiskModel:
    """
    Continuous Learning Customer Return Risk Model v2.
    Predicts fraud/abuse risk scores (0–100) based on historical return attributes.
    Supports automated feedback collection, threshold-triggered retraining,
    Champion vs. Challenger model promotion, and versioned audit history.
    """

    def __init__(self, retrain_threshold: int = 10):
        self.model = None
        self.preprocessor = None
        self.version = None
        self.metrics = {}
        self.cv_results = {}
        self.customer_db = {}
        self.feature_names = NUMERIC_FEATURES + CATEGORICAL_FEATURES
        self.retrain_threshold = retrain_threshold
        self.unprocessed_feedback_count = 0
        self.retraining_history = []
        self.feedback_file = os.path.join(DATA_DIR, 'feedback_records.csv')
        self.history_file = os.path.join(MODEL_DIR, 'retraining_history.json')

    def get_default_features(self, customer_id: Optional[str] = None) -> dict:
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

    def prepare_data(self, customer_df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """Prepares feature matrix X and target y from customer dataframe."""
        X_df = customer_df[self.feature_names].copy()
        X_df['mismatch_flag_history'] = X_df['mismatch_flag_history'].astype(int)

        if TARGET_COLUMN in customer_df.columns:
            y = customer_df[TARGET_COLUMN].astype(int).values
        elif 'is_fraud' in customer_df.columns:
            y = customer_df['is_fraud'].astype(int).values
        else:
            raise KeyError(f"Target column '{TARGET_COLUMN}' not found in dataframe.")

        return X_df, y

    def train(self, customer_df: pd.DataFrame) -> dict:
        """Initial baseline model training across candidate classifiers."""
        logger.info("Training initial Customer Return Risk Model baseline...")
        X_df, y = self.prepare_data(customer_df)

        self.preprocessor = ColumnTransformer(transformers=[
            ('num', StandardScaler(), NUMERIC_FEATURES),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_FEATURES)
        ])

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
            cv_metrics = cross_validate(
                clf, X_train, y_train, cv=skf,
                scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
            )

            clf.fit(X_train, y_train)
            train_duration = round(time.time() - start_time, 3)

            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)[:, 1]

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            roc_auc = roc_auc_score(y_test, y_prob)
            pr_auc = average_precision_score(y_test, y_prob)

            results[name] = {
                'model': clf,
                'accuracy': round(float(acc), 4),
                'precision': round(float(prec), 4),
                'recall': round(float(rec), 4),
                'f1_score': round(float(f1), 4),
                'roc_auc': round(float(roc_auc), 4),
                'pr_auc': round(float(pr_auc), 4),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
                'cv_scores': {
                    'mean_accuracy': round(float(np.mean(cv_metrics['test_accuracy'])), 4),
                    'mean_precision': round(float(np.mean(cv_metrics['test_precision'])), 4),
                    'mean_recall': round(float(np.mean(cv_metrics['test_recall'])), 4),
                    'mean_f1': round(float(np.mean(cv_metrics['test_f1'])), 4),
                    'mean_roc_auc': round(float(np.mean(cv_metrics['test_roc_auc'])), 4)
                },
                'train_duration_sec': train_duration
            }

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

        self._build_customer_db(customer_df)
        self._save_model()
        return results

    def predict_risk(self, customer_features: dict) -> dict:
        """Generates risk score (0–100) and level ('Low'/'Medium'/'High') for a customer."""
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
            logger.warning(f"Model prediction failed ({e}). Using fallback.")
            return self._rule_based_fallback(full_features)

        return {
            'risk_score': risk_score,
            'risk_level': risk_level,
            'risk_probability': round(float(prob), 4),
            'prediction_method': method,
            'model_version': self.version or 'v3_production',
            'recommendation': self._get_recommendation(risk_level)
        }

    def record_ground_truth_feedback(self, customer_id: str, actual_fraud_label: int, notes: Optional[str] = None) -> dict:
        """
        Appends actual verified return/audit outcome label to dataset and updates customer DB.
        Triggers automated retraining if the unprocessed feedback count reaches threshold.
        """
        logger.info(f"Recording feedback for customer {customer_id}: actual_fraud_label={actual_fraud_label}")

        if customer_id in self.customer_db:
            cust = self.customer_db[customer_id]
            cust['flagged_by_company'] = int(actual_fraud_label)
        else:
            cust = self.get_default_features(customer_id)
            cust['flagged_by_company'] = int(actual_fraud_label)
            self.customer_db[customer_id] = cust

        feedback_entry = {
            'Customer_ID': customer_id,
            'total_orders': cust.get('total_orders', 1),
            'total_returns': cust.get('total_returns', 0),
            'return_ratio': cust.get('return_ratio', 0.0),
            'avg_return_window': cust.get('avg_return_window', 0.0),
            'product_category_risk': cust.get('product_category_risk', 0.0),
            'vague_reason_count': cust.get('vague_reason_count', 0),
            'mismatch_flag_history': cust.get('mismatch_flag_history', 0),
            'customer_rating_behavior': cust.get('customer_rating_behavior', 5.0),
            'previous_fraud_flags': cust.get('previous_fraud_flags', 0),
            'account_age_days': cust.get('account_age_days', 0),
            'most_common_category': cust.get('most_common_category', 'Unknown'),
            'flagged_by_company': int(actual_fraud_label),
            'timestamp': datetime.now().isoformat(),
            'notes': notes or "Verified company audit outcome"
        }

        os.makedirs(DATA_DIR, exist_ok=True)
        fb_df = pd.DataFrame([feedback_entry])
        if os.path.exists(self.feedback_file):
            fb_df.to_csv(self.feedback_file, mode='a', header=False, index=False)
        else:
            fb_df.to_csv(self.feedback_file, mode='w', header=True, index=False)

        self.unprocessed_feedback_count += 1
        logger.info(f"Feedback recorded. Unprocessed feedback count: {self.unprocessed_feedback_count}/{self.retrain_threshold}")

        retrain_triggered = False
        retrain_result = None
        if self.unprocessed_feedback_count >= self.retrain_threshold:
            logger.info("Retrain threshold reached! Initiating automated retraining pipeline...")
            retrain_result = self.retrain_pipeline(reason="Threshold reached")
            retrain_triggered = True

        return {
            "status": "feedback_recorded",
            "customer_id": customer_id,
            "actual_fraud_label": actual_fraud_label,
            "unprocessed_feedback_count": self.unprocessed_feedback_count,
            "retrain_threshold": self.retrain_threshold,
            "retrain_triggered": retrain_triggered,
            "retrain_result": retrain_result
        }

    def retrain_pipeline(self, force: bool = False, reason: str = "Manual trigger") -> dict:
        """
        Automated retraining pipeline:
        1. Merges base customer dataset + feedback records.
        2. Fits candidate Challenger models with Stratified K-Fold CV.
        3. Compares Challenger vs active Champion model performance.
        4. Promotes Challenger to production ONLY if ROC-AUC / F1 is superior.
        5. Logs retraining audit history & archives model version.
        """
        logger.info(f"Executing continuous learning retraining pipeline ({reason})...")

        features_csv = os.path.join(DATA_DIR, 'customer_features.csv')
        if not os.path.exists(features_csv):
            raise FileNotFoundError(f"Base customer features file not found at {features_csv}")

        df_base = pd.read_csv(features_csv)

        if os.path.exists(self.feedback_file):
            df_fb = pd.read_csv(self.feedback_file)
            fb_cols = [c for c in df_base.columns if c in df_fb.columns]
            df_fb_aligned = df_fb[fb_cols].copy()
            df_combined = pd.concat([df_base, df_fb_aligned], ignore_index=True)
            df_combined = df_combined.drop_duplicates(subset=['Customer_ID'], keep='last')
        else:
            df_combined = df_base

        total_dataset_size = len(df_combined)
        logger.info(f"Retraining dataset combined size: {total_dataset_size} customer records.")

        X_df, y = self.prepare_data(df_combined)

        preprocessor = ColumnTransformer(transformers=[
            ('num', StandardScaler(), NUMERIC_FEATURES),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_FEATURES)
        ])

        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X_df, y, test_size=0.2, random_state=42, stratify=y
        )

        X_train = preprocessor.fit_transform(X_train_raw)
        X_test = preprocessor.transform(X_test_raw)

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

        challenger_results = {}
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for name, clf in candidate_models.items():
            cv_metrics = cross_validate(
                clf, X_train, y_train, cv=skf, scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
            )
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)[:, 1]

            challenger_results[name] = {
                'model': clf,
                'accuracy': round(float(accuracy_score(y_test, y_pred)), 4),
                'precision': round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
                'recall': round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
                'f1_score': round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
                'roc_auc': round(float(roc_auc_score(y_test, y_prob)), 4),
                'pr_auc': round(float(average_precision_score(y_test, y_prob)), 4),
                'cv_scores': {
                    'mean_accuracy': round(float(np.mean(cv_metrics['test_accuracy'])), 4),
                    'mean_roc_auc': round(float(np.mean(cv_metrics['test_roc_auc'])), 4)
                }
            }

        best_challenger_name = max(challenger_results, key=lambda k: challenger_results[k]['roc_auc'])
        best_challenger = challenger_results[best_challenger_name]

        current_champion_auc = self.metrics.get('roc_auc', 0.0)
        challenger_auc = best_challenger['roc_auc']

        # Determine promotion (Promote if challenger outperforms champion or if forced / first run)
        promoted = (challenger_auc >= current_champion_auc) or force or (self.model is None)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        new_version_id = f"v2_{best_challenger_name}_{timestamp}"

        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "reason": reason,
            "dataset_size": total_dataset_size,
            "unprocessed_feedback_processed": self.unprocessed_feedback_count,
            "champion_version": self.version,
            "champion_roc_auc": current_champion_auc,
            "challenger_version": new_version_id,
            "challenger_model_name": best_challenger_name,
            "challenger_roc_auc": challenger_auc,
            "challenger_f1": best_challenger['f1_score'],
            "challenger_accuracy": best_challenger['accuracy'],
            "status": "PROMOTED" if promoted else "REJECTED"
        }

        # Save model version binary in versions folder
        os.makedirs(VERSION_DIR, exist_ok=True)
        version_path = os.path.join(VERSION_DIR, f"{new_version_id}.pkl")
        joblib.dump({
            'model': best_challenger['model'],
            'preprocessor': preprocessor,
            'version': new_version_id,
            'metrics': best_challenger
        }, version_path)

        if promoted:
            logger.info(f"PROMOTION SUCCESS: Challenger ({new_version_id}) AUC {challenger_auc} >= Champion AUC {current_champion_auc}. Promoting to production!")
            self.model = best_challenger['model']
            self.preprocessor = preprocessor
            self.version = new_version_id
            self.metrics = {
                'best_model_name': best_challenger_name,
                'accuracy': best_challenger['accuracy'],
                'precision': best_challenger['precision'],
                'recall': best_challenger['recall'],
                'f1_score': best_challenger['f1_score'],
                'roc_auc': best_challenger['roc_auc'],
                'pr_auc': best_challenger['pr_auc'],
                'cv_scores': best_challenger['cv_scores']
            }
            self._build_customer_db(df_combined)
            self._save_model()
        else:
            logger.info(f"PROMOTION REJECTED: Challenger AUC {challenger_auc} < Champion AUC {current_champion_auc}. Retaining current production Champion.")

        self.unprocessed_feedback_count = 0
        self._record_retraining_log(audit_entry)

        return audit_entry

    def _record_retraining_log(self, audit_entry: dict):
        """Appends audit entry to retraining history log file."""
        self.retraining_history.append(audit_entry)
        os.makedirs(MODEL_DIR, exist_ok=True)
        with open(self.history_file, 'w') as f:
            json.dump(self.retraining_history, f, indent=2)

    def get_retraining_history(self) -> List[dict]:
        """Loads and returns full retraining history log."""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    self.retraining_history = json.load(f)
            except Exception:
                pass
        return self.retraining_history

    def _rule_based_fallback(self, features: dict) -> dict:
        """Rule-based risk score fallback for new customers."""
        score = 10.0
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
        """Populates in-memory customer lookup database."""
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
                'flagged_by_company': int(row['flagged_by_company']) if 'flagged_by_company' in row else 0
            }

    def get_customer_features(self, customer_id: str) -> Optional[dict]:
        """Retrieves customer features from database lookup."""
        return self.customer_db.get(customer_id, None)

    def update_customer_history(self, customer_id: str, new_event: dict) -> dict:
        """Updates customer profile when a new transaction/return occurs."""
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
        """Saves active production model and metadata."""
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
            'unprocessed_feedback_count': self.unprocessed_feedback_count,
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
                'total_customers_in_db': len(self.customer_db),
                'unprocessed_feedback_count': self.unprocessed_feedback_count,
                'retrain_threshold': self.retrain_threshold
            }, f, indent=2)

    def load_model(self) -> bool:
        """Loads saved production model, metadata, and retraining history."""
        model_path = os.path.join(MODEL_DIR, 'customer_risk_model.pkl')
        self.get_retraining_history()
        if os.path.exists(model_path):
            try:
                data = joblib.load(model_path)
                self.model = data['model']
                self.preprocessor = data['preprocessor']
                self.version = data['version']
                self.metrics = data['metrics']
                self.cv_results = data.get('cv_results', {})
                self.customer_db = data.get('customer_db', {})
                self.unprocessed_feedback_count = data.get('unprocessed_feedback_count', 0)
                logger.info(f"Loaded Production Model ({self.version}) with {len(self.customer_db)} customers.")
                return True
            except Exception as e:
                logger.error(f"Failed to load model from {model_path}: {e}")
                return False
        return False
