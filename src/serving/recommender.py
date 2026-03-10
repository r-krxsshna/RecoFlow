from pathlib import Path

import joblib
import pandas as pd

from src.candidate_generation.generate_candidates import generate_candidate_for_users
from src.common.logging import get_logger

logger = get_logger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent

MODEL_DIR = BASE_DIR / 'models'
CANDIDATE_MODEL_DIR = MODEL_DIR / 'candidate_model'
RANKER_MODEL_DIR = MODEL_DIR / 'ranker' / 'ranker.joblib'

TRAIN_PATH = BASE_DIR / 'data' / 'processed' / 'splits' / 'train.parquet'

class Recommender:

    def __init__(self):

        logger.info('loading ranker model')

        self.ranker = joblib.load(RANKER_MODEL_DIR)

        logger.info('loading interactions')

        self.interactions = pd.read_parquet(TRAIN_PATH)

    def recommend(self, user_id: int, k: int = 10):

        logger.info(f"Generating candidates for user {user_id}")

        candidates = generate_candidate_for_users(
            user_ids=[user_id],
            train_interactions=self.interactions,
            k=50
        )

        if candidates.empty:
            logger.warning(f"No candidates generated for user {user_id}")
            return []

        user_count = (
            self.interactions
            .groupby('user_id')
            .size()
            .to_dict()
        )

        item_count = (
            self.interactions
            .groupby('item_id')
            .size()
            .to_dict()
        )

        candidates["user_interaction_count"] = candidates["user_id"].map(user_count)
        candidates["item_interaction_count"] = candidates["item_id"].map(item_count)

        X = candidates[
            ["user_interaction_count", "item_interaction_count"]
        ]

        logger.info("Ranking candidates")

        scores = self.ranker.predict_proba(X)[:, 1]

        candidates["scores"] = scores

        recs = (
            candidates
            .sort_values("scores", ascending=False)
            .head(k)
        )

        return recs["item_id"].tolist()