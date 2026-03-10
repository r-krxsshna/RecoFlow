from pathlib import Path
from scipy.sparse import csr_matrix,issparse

import joblib
import pandas as pd

from src.common.logging import get_logger

logger = get_logger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent

def generate_candidate_for_users(
        user_ids,
        train_interactions,
        k: int = 50
):
    logger.info("Loading candidate model")

    model_dir = BASE_DIR / "models" / "candidate_model"

    model = joblib.load(model_dir / "als_model.joblib")
    user_map = joblib.load(model_dir / "user_mapping.joblib")
    item_map = joblib.load(model_dir / "item_mapping.joblib")

    index_to_item = {v: k for k, v in item_map.items()}

    if not issparse(train_interactions):
        train_interactions = csr_matrix(train_interactions.values)

    candidates = []

    for user_id in user_ids:

        if user_id not in user_map:
            continue

        uidx = user_map[user_id]

        user_row = train_interactions[uidx]

        item_indices, scores = model.recommend(
            userid = uidx,
            user_items = user_row,
            N=k
        )

        for i,s  in zip(item_indices, scores):
            candidates.append(
                {
                    'user_id': user_id,
                    'item_id': index_to_item[i],
                    "score": float(s)
                }
            )

    result = pd.DataFrame(candidates)

    logger.info("Generated %d candidate rows", len(result))

    return result