import os
import sys
import logging

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from generate_data import generate_synthetic_data
from preprocess import clean_and_preprocess_data
from simulate_customers import simulate_customer_history
from aggregate_features import build_customer_features
from customer_risk_model import CustomerRiskModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')


def run_full_ml_pipeline():
    """
    Standalone pipeline orchestrator. Generates data, preprocesses, trains, and saves model.

    Flow:
        generate_data.py     -> ecommerce_returns_synthetic_data.csv
        preprocess.py        -> ecommerce_returns_clean.csv
        simulate_customers.py -> ecommerce_returns_simulated_customers.csv
        aggregate_features.py -> customer_features.csv
        CustomerRiskModel    -> customer_risk_model.pkl
    """
    raw_csv = os.path.join(DATA_DIR, 'ecommerce_returns_synthetic_data.csv')
    clean_csv = os.path.join(DATA_DIR, 'ecommerce_returns_clean.csv')
    simulated_csv = os.path.join(DATA_DIR, 'ecommerce_returns_simulated_customers.csv')
    features_csv = os.path.join(DATA_DIR, 'customer_features.csv')

    os.makedirs(DATA_DIR, exist_ok=True)

    logger.info("=" * 70)
    logger.info("  CUSTOMER RETURN RISK ANALYZER - ML PIPELINE")
    logger.info("=" * 70)

    logger.info("\n--- STEP 1: Generate Synthetic Dataset ---")
    generate_synthetic_data(raw_csv)

    logger.info("\n--- STEP 2: Preprocess & Clean Data ---")
    clean_and_preprocess_data(raw_csv, clean_csv)

    logger.info("\n--- STEP 3: Simulate Customer Profiles ---")
    simulate_customer_history(clean_csv, simulated_csv)

    logger.info("\n--- STEP 4: Feature Aggregation ---")
    customer_df = build_customer_features(simulated_csv, features_csv)

    logger.info("\n--- STEP 5: Model Training & Cross-Validation ---")
    model = CustomerRiskModel()
    training_results = model.train(customer_df)

    logger.info("\n" + "=" * 70)
    logger.info("  PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Model Version : {model.version}")
    logger.info(f"Dataset Size  : {len(customer_df)} customers")
    logger.info(f"Model Saved   : ml/models/customer_risk_model.pkl")
    logger.info(f"Features Saved: ml/data/customer_features.csv")

    for name, res in training_results.items():
        logger.info(
            f"  {name:18s} | Acc: {res['accuracy']:.4f} | Prec: {res['precision']:.4f} | "
            f"Rec: {res['recall']:.4f} | F1: {res['f1_score']:.4f} | ROC-AUC: {res['roc_auc']:.4f}"
        )

    logger.info("\nNext step: python ml/src/predict.py")
    return model


if __name__ == "__main__":
    run_full_ml_pipeline()
