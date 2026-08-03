import os
import logging
import pandas as pd
from customer_risk_model import CustomerRiskModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_continuous_retraining(force: bool = True, reason: str = "Manual trigger via continuous_pipeline"):
    """
    Orchestrates continuous learning retraining:
    Merges base dataset + feedback records, runs K-Fold CV across candidate models,
    evaluates Challenger vs. active Champion, and updates production model if superior.
    """
    logger.info("=" * 70)
    logger.info("  CONTINUOUS LEARNING RETRAINING PIPELINE v2")
    logger.info("=" * 70)

    model = CustomerRiskModel()
    if not model.load_model():
        logger.info("No active model binary found. Executing initial baseline training...")
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        features_csv = os.path.join(base_dir, 'data', 'customer_features.csv')
        df_base = pd.read_csv(features_csv)
        model.train(df_base)

    audit_entry = model.retrain_pipeline(force=force, reason=reason)

    logger.info("\n" + "=" * 70)
    logger.info("  CONTINUOUS RETRAINING SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Reason: {audit_entry['reason']}")
    logger.info(f"Dataset Size: {audit_entry['dataset_size']} records")
    logger.info(f"Champion Version: {audit_entry['champion_version']} (AUC: {audit_entry['champion_roc_auc']})")
    logger.info(f"Challenger Version: {audit_entry['challenger_version']} (AUC: {audit_entry['challenger_roc_auc']})")
    logger.info(f"Status: {audit_entry['status']}")

    return model, audit_entry


if __name__ == "__main__":
    run_continuous_retraining()
