from pathlib import Path
import joblib
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from src.ranking.feature_builder import build_ranking_features
from src.common.logging import get_logger

logger = get_logger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent

TRAIN_PATH = BASE_DIR / 'data' / 'processed' / 'splits' / 'train.parquet'
MODEL_DIR = BASE_DIR / 'models' / 'ranker'
MODEL_PATH = MODEL_DIR / 'ranker.joblib'

def train_ranker():

    logger.info("Loading train split for ranker")

    interactions = pd.read_parquet(TRAIN_PATH)

    logger.info("Building ranking features")

    features = build_ranking_features(interactions=interactions)

    required_cols = {"user_id", "item_id", "label"}
    if not required_cols.issubset(features.columns):
        raise ValueError("feature_builder must return user_id, item_id and label columns")

    X = features.drop(columns=["user_id", "item_id", "label"])
    y = features["label"]

    logger.info("Training ranker model")

    model = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    )

    model.fit(X, y)

    preds = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, preds)

    logger.info(f"Training AUC: %.4f",auc)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    logger.info("Ranker model saved at %s", MODEL_PATH)

if __name__ == "__main__":
    train_ranker()