from pathlib import Path
import pandas as pd
from scipy.stats import false_discovery_control

from src.common.logging import get_logger

logger = get_logger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent

TRAIN_PATH = BASE_DIR / 'data' / 'processed' / 'splits' / 'train.parquet'
NEW_DATA_PATH = BASE_DIR / 'data' / 'processed' / 'splits' / 'new_interactions.parquet'

def calculate_stats(df):
    """Compute dataset statistics"""

    stats = {
        "num_interactions": len(df),
        "num_users": df["user_id"].nunique(),
        "num_items": df["item_id"].nunique(),
        "avg_interactions_per_user": len(df) / df["user_id"].nunique(),
    }

    return stats

def detect_drift(train_stats, new_stats, threshold=0.2):
    """Compute dataset statistics"""

    drift_detected = False

    for key in train_stats:

        train_value = train_stats[key]
        new_value = new_stats[key]

        change =  abs(new_value - train_value) / train_value

        logger.info(
            "Feature: %s | Train: %.2f | New: %.2f | Change: %.2f%%",
            key,
            train_value,
            new_value,
            change * 100,
        )

        if change > threshold:
            logger.warning("Drift detected in %s", key)
            drift_detected = True

        return drift_detected

def run_drift_monitoring():
    """Main drift monitoring pipeline"""

    logger.info("Drift monitoring started")

    train_df = pd.read_parquet(TRAIN_PATH)
    new_df = pd.read_parquet(NEW_DATA_PATH)

    train_stats = calculate_stats(train_df)
    new_stats = calculate_stats(new_df)

    drfit = detect_drift(train_stats, new_stats)

    if drfit:
        logger.warning("Drift detected! Model retraining recommended")
    else:
        logger.info("No drift detected")

    logger.info("Drift monitoring completed")
