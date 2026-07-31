import os
import json
import time
import joblib
import logging
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
VERSION_DIR = os.path.join(MODEL_DIR, 'versions')
LOG_DIR = os.path.join(BASE_DIR, 'logs')
ARTIFACT_DIR = os.path.join(BASE_DIR, 'artifacts')

LEAKAGE_COLS = [
    'Days_to_Return', 'Return_Date', 'Return_Status', 'Return_Reason',
    'total_returns', 'Order_ID', 'Product_ID', 'User_ID', 'Customer_ID'
]

NUMERIC_FEATURES = [
    'Product_Price', 'Order_Quantity', 'User_Age', 'Discount_Applied',
    'Order_Month', 'Order_Day', 'Order_DayOfWeek',
    'Total_Order_Value', 'Effective_Order_Value', 'Discount_Percentage',
    'total_orders', 'return_ratio', 'avg_return_window', 'vague_reason_count',
    'category_return_rate', 'category_avg_price', 'category_order_count',
    'location_return_rate', 'location_order_count',
    'payment_return_rate', 'payment_order_count',
    'shipping_return_rate', 'shipping_order_count',
    'price_percentile', 'quantity_percentile', 'discount_percentile',
    'price_x_quantity', 'discount_ratio', 'price_deviation_from_category'
]

CAT_FEATURES = [
    'Product_Category', 'User_Gender', 'Payment_Method', 'Shipping_Method'
]

BOOLEAN_FEATURES = [
    'Is_Weekend', 'Is_Clothing_Category', 'High_Discount_Flag',
    'Is_Express_Shipping', 'mismatch_flag_history'
]


def load_data():
    path = os.path.join(DATA_DIR, 'order_features.csv')
    logger.info(f"Loading data from {path}")
    df = pd.read_csv(path)
    logger.info(f"Raw data shape: {df.shape}")
    return df


def preprocess(df):
    logger.info("Starting preprocessing...")

    df = df.drop(columns=[c for c in LEAKAGE_COLS if c in df.columns], errors='ignore')
    logger.info(f"After removing leakage cols: {df.shape}")

    df = df.dropna(subset=['Is_Returned'])
    logger.info(f"After dropping null targets: {df.shape}")

    available_num = [c for c in NUMERIC_FEATURES if c in df.columns]
    available_cat = [c for c in CAT_FEATURES if c in df.columns]
    available_bool = [c for c in BOOLEAN_FEATURES if c in df.columns]

    for col in available_num:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    for col in available_cat:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])

    for col in available_bool:
        df[col] = df[col].astype(int)

    logger.info(f"Numeric features: {len(available_num)}")
    logger.info(f"Categorical features: {len(available_cat)}")
    logger.info(f"Boolean features: {len(available_bool)}")
    logger.info(f"Total features: {len(available_num) + len(available_cat) + len(available_bool)}")

    return df, available_num, available_cat, available_bool


def prepare_features(df, num_cols, cat_cols, bool_cols):
    X_num = df[num_cols].values
    X_bool = df[bool_cols].values

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), list(range(len(num_cols)))),
        ('bool', 'passthrough', list(range(len(num_cols), len(num_cols) + len(bool_cols))))
    ])

    X_num_bool = preprocessor.fit_transform(np.hstack([X_num, X_bool]))

    if cat_cols:
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        X_cat = encoder.fit_transform(df[cat_cols])
        X = np.hstack([X_num_bool, X_cat])
    else:
        X = X_num_bool
        encoder = None

    y = df['Is_Returned'].values

    return X, y, preprocessor, encoder


def train_models(X_train, y_train, X_val, y_val):
    logger.info("Training models...")

    smote = SMOTE(random_state=42, sampling_strategy=0.5)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    logger.info(f"After SMOTE: {X_train_res.shape[0]} samples (was {X_train.shape[0]})")
    logger.info(f"Class distribution after SMOTE: {np.bincount(y_train_res.astype(int))}")

    models = {
        'LogisticRegression': LogisticRegression(
            max_iter=1000, class_weight='balanced', random_state=42, C=0.5
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_split=5,
            class_weight='balanced', random_state=42, n_jobs=-1
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            subsample=0.8, random_state=42
        ),
    }

    try:
        from xgboost import XGBClassifier
        scale_pos = len(y_train_res[y_train_res == 0]) / len(y_train_res[y_train_res == 1])
        models['XGBoost'] = XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos, random_state=42,
            eval_metric='logloss', use_label_encoder=False
        )
    except ImportError:
        logger.warning("XGBoost not available, skipping...")

    results = {}
    for name, model in models.items():
        logger.info(f"Training {name}...")
        start = time.time()

        model.fit(X_train_res, y_train_res)
        train_time = time.time() - start

        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else y_pred

        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, zero_division=0)
        rec = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        auc_roc = roc_auc_score(y_val, y_prob)
        pr_auc = average_precision_score(y_val, y_prob)

        results[name] = {
            'model': model,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'auc_roc': auc_roc,
            'pr_auc': pr_auc,
            'train_time': train_time,
            'y_pred': y_pred,
            'y_prob': y_prob
        }

        logger.info(f"  {name}: Acc={acc:.4f} | F1={f1:.4f} | AUC-ROC={auc_roc:.4f} | PR-AUC={pr_auc:.4f} | Time={train_time:.1f}s")

    return results, X_train_res, y_train_res


def build_ensemble(results, X_train, y_train):
    logger.info("Building ensemble model...")

    top_models = sorted(results.items(), key=lambda x: x[1]['auc_roc'], reverse=True)[:3]
    estimators = [(name, res['model']) for name, res in top_models]

    ensemble = VotingClassifier(
        estimators=estimators, voting='soft', weights=[3, 2, 1]
    )
    ensemble.fit(X_train, y_train)

    return ensemble, [name for name, _ in estimators]


def evaluate_model(model, X_test, y_test, model_name="Model"):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc_roc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    logger.info(f"\n{'='*60}")
    logger.info(f"  {model_name} - Test Set Evaluation")
    logger.info(f"{'='*60}")
    logger.info(f"  Accuracy:  {acc:.4f}")
    logger.info(f"  Precision: {prec:.4f}")
    logger.info(f"  Recall:    {rec:.4f}")
    logger.info(f"  F1 Score:  {f1:.4f}")
    logger.info(f"  AUC-ROC:   {auc_roc:.4f}")
    logger.info(f"  PR-AUC:    {pr_auc:.4f}")
    logger.info(f"\n  Confusion Matrix:")
    logger.info(f"  {cm}")
    logger.info(f"\n  Classification Report:")
    logger.info(f"\n{classification_report(y_test, y_pred, target_names=['Not Returned', 'Returned'])}")

    return {
        'accuracy': acc, 'precision': prec, 'recall': rec,
        'f1': f1, 'auc_roc': auc_roc, 'pr_auc': pr_auc,
        'confusion_matrix': cm.tolist()
    }


def save_model_version(model, metrics, model_name, feature_info):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    version_name = f"{model_name}_{timestamp}"

    model_path = os.path.join(VERSION_DIR, f"{version_name}.pkl")
    metrics_path = os.path.join(VERSION_DIR, f"{version_name}_metrics.json")

    joblib.dump(model, model_path)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)

    latest_path = os.path.join(MODEL_DIR, 'return_risk_model_v2.pkl')
    joblib.dump(model, latest_path)

    meta_path = os.path.join(MODEL_DIR, 'return_risk_model_v2_metadata.json')
    meta = {
        'model_name': model_name,
        'version': version_name,
        'timestamp': timestamp,
        'features': feature_info,
        'metrics': metrics,
        'training_samples': int(metrics.get('train_samples', 0)),
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2, default=str)

    logger.info(f"Model saved: {model_path}")
    logger.info(f"Latest model saved: {latest_path}")
    return version_name


class ContinuousLearner:
    def __init__(self, model_path=None):
        self.model_path = model_path or os.path.join(MODEL_DIR, 'return_risk_model_v2.pkl')
        self.model = None
        self.preprocessor = None
        self.encoder = None
        self.feature_info = None
        self.performance_log = []
        self.load()

    def load(self):
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            meta_path = self.model_path.replace('.pkl', '_metadata.json')
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    self.feature_info = json.load(f)
            logger.info("ContinuousLearner: Model loaded successfully")
        else:
            logger.warning("ContinuousLearner: No model found")

    def predict(self, X):
        if self.model is None:
            raise ValueError("No model loaded. Train first.")
        return self.model.predict_proba(X)[:, 1]

    def incremental_train(self, X_new, y_new, X_val, y_val):
        if self.model is None:
            logger.error("No model to update. Train baseline first.")
            return None

        logger.info(f"Incremental update with {len(X_new)} new samples...")

        smote = SMOTE(random_state=42, sampling_strategy=0.5)
        X_res, y_res = smote.fit_resample(X_new, y_new)

        if hasattr(self.model, 'estimators_'):
            for est in self.model.estimators_:
                est.fit(X_res, y_res)
        else:
            self.model.fit(X_res, y_res)

        y_prob = self.model.predict_proba(X_val)[:, 1]
        y_pred = self.model.predict(X_val)
        new_metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'f1': f1_score(y_val, y_pred, zero_division=0),
            'auc_roc': roc_auc_score(y_val, y_prob),
            'pr_auc': average_precision_score(y_val, y_prob),
            'timestamp': datetime.now().isoformat(),
            'incremental_samples': len(X_new)
        }

        self.performance_log.append(new_metrics)

        if len(self.performance_log) > 1:
            prev = self.performance_log[-2]['auc_roc']
            curr = new_metrics['auc_roc']
            if curr < prev - 0.05:
                logger.warning(f"Performance drop detected! AUC: {prev:.4f} -> {curr:.4f}")
                logger.warning("Consider reverting to previous model version.")

        joblib.dump(self.model, self.model_path)
        logger.info(f"Incremental update complete. New AUC-ROC: {new_metrics['auc_roc']:.4f}")

        return new_metrics

    def check_drift(self, X_recent, y_recent, threshold=0.05):
        if self.model is None:
            return False

        y_prob = self.model.predict_proba(X_recent)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        current_auc = roc_auc_score(y_recent, y_prob)

        if self.feature_info and 'metrics' in self.feature_info:
            baseline_auc = self.feature_info['metrics'].get('auc_roc', 0)
            drift = baseline_auc - current_auc
            if drift > threshold:
                logger.warning(f"Drift detected! Baseline AUC: {baseline_auc:.4f}, Current: {current_auc:.4f}, Drift: {drift:.4f}")
                return True

        return False


def run_pipeline():
    logger.info("="*60)
    logger.info("  E-COMMERCE RETURN RISK - CONTINUOUS LEARNING PIPELINE")
    logger.info("="*60)

    df = load_data()
    df, num_cols, cat_cols, bool_cols = preprocess(df)

    X, y, preprocessor, encoder = prepare_features(df, num_cols, cat_cols, bool_cols)
    logger.info(f"Final feature matrix: {X.shape}")

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.18, random_state=42, stratify=y_train_full
    )

    logger.info(f"Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")
    logger.info(f"Train return rate: {y_train.mean():.3f}")
    logger.info(f"Val return rate: {y_val.mean():.3f}")
    logger.info(f"Test return rate: {y_test.mean():.3f}")

    results, X_train_res, y_train_res = train_models(X_train, y_train, X_val, y_val)

    ensemble_model, ensemble_members = build_ensemble(results, X_train_res, y_train_res)
    ensemble_metrics = evaluate_model(ensemble_model, X_test, y_test, "Ensemble")

    best_single = max(results.items(), key=lambda x: x[1]['auc_roc'])
    best_single_metrics = evaluate_model(
        best_single[1]['model'], X_test, y_test, f"Best Single ({best_single[0]})"
    )

    if ensemble_metrics['auc_roc'] >= best_single_metrics['auc_roc']:
        final_model = ensemble_model
        final_name = "Ensemble"
        final_metrics = ensemble_metrics
    else:
        final_model = best_single[1]['model']
        final_name = best_single[0]
        final_metrics = best_single_metrics

    final_metrics['train_samples'] = X_train.shape[0]
    final_metrics['val_samples'] = X_val.shape[0]
    final_metrics['test_samples'] = X_test.shape[0]
    final_metrics['ensemble_members'] = ensemble_members

    feature_info = {
        'numeric_features': num_cols,
        'categorical_features': cat_cols,
        'boolean_features': bool_cols,
        'total_features': X.shape[1]
    }

    version = save_model_version(final_model, final_metrics, final_name, feature_info)

    logger.info("\n" + "="*60)
    logger.info("  PIPELINE COMPLETE")
    logger.info("="*60)
    logger.info(f"  Best Model: {final_name}")
    logger.info(f"  Version: {version}")
    logger.info(f"  Test AUC-ROC: {final_metrics['auc_roc']:.4f}")
    logger.info(f"  Test F1: {final_metrics['f1']:.4f}")
    logger.info(f"  Test Accuracy: {final_metrics['accuracy']:.4f}")

    return final_model, final_metrics


if __name__ == "__main__":
    model, metrics = run_pipeline()
