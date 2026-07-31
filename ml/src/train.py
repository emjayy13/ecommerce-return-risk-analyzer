import os
import pandas as pd
import logging

from preprocess import clean_and_preprocess_data
from simulate_customers import simulate_customer_history
from aggregate_features import build_customer_features
from customer_risk_model import CustomerRiskModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_full_ml_pipeline():
    """
    Executes the end-to-end ML pipeline:
    1. Preprocesses raw order dataset.
    2. Simulates customer profiles with independent ground-truth risk labels (`flagged_by_company`).
    3. Aggregates customer-level historical features.
    4. Trains customer risk models with Stratified K-Fold Cross-Validation.
    5. Evaluates model performance (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC).
    6. Saves trained model artifact and metadata JSON.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_csv = os.path.join(base_dir, 'data', 'ecommerce_returns_synthetic_data.csv')
    clean_csv = os.path.join(base_dir, 'data', 'ecommerce_returns_clean.csv')
    simulated_csv = os.path.join(base_dir, 'data', 'ecommerce_returns_simulated_customers.csv')
    features_csv = os.path.join(base_dir, 'data', 'customer_features.csv')

    logger.info("=" * 70)
    logger.info("  STARTING CUSTOMER RETURN RISK ANALYZER - ML PIPELINE")
    logger.info("=" * 70)

    # Step 1: Preprocessing
    logger.info("\n--- STEP 1: Data Cleaning & Preprocessing ---")
    clean_and_preprocess_data(raw_csv, clean_csv)

    # Step 2: Customer Simulation
    logger.info("\n--- STEP 2: Customer Profile Simulation & Independent Target Generation ---")
    simulate_customer_history(clean_csv, simulated_csv)

    # Step 3: Feature Aggregation
    logger.info("\n--- STEP 3: Feature Aggregation per Customer ---")
    customer_df = build_customer_features(simulated_csv, features_csv)

    # Step 4: Model Training & Cross Validation
    logger.info("\n--- STEP 4: Model Training, Cross-Validation & Metric Evaluation ---")
    model = CustomerRiskModel()
    training_results = model.train(customer_df)

    logger.info("\n" + "=" * 70)
    logger.info("  PIPELINE EVALUATION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Selected Model Version: {model.version}")
    logger.info(f"Total Customer Records: {len(customer_df)}")
    
    for name, res in training_results.items():
        logger.info(
            f"Model: {name:18s} | Acc: {res['accuracy']:.4f} | Prec: {res['precision']:.4f} | "
            f"Rec: {res['recall']:.4f} | F1: {res['f1_score']:.4f} | ROC-AUC: {res['roc_auc']:.4f}"
        )
        logger.info(
            f"  5-Fold CV Mean Scores -> Acc: {res['cv_scores']['mean_accuracy']:.4f} | "
            f"Prec: {res['cv_scores']['mean_precision']:.4f} | Rec: {res['cv_scores']['mean_recall']:.4f} | "
            f"F1: {res['cv_scores']['mean_f1']:.4f} | ROC-AUC: {res['cv_scores']['mean_roc_auc']:.4f}"
        )

    logger.info("\nPipeline execution complete. Model successfully exported.")
    return model


if __name__ == "__main__":
    run_full_ml_pipeline()
