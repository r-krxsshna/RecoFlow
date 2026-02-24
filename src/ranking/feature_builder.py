import pandas as pd
import numpy as np

from src.common.logging import get_logger

logger = get_logger(__name__)

def build_ranking_features(interactions: pd.DataFrame, num_negative_samples: int = 1) -> pd.DataFrame:
    """
    Build training data for the ranking model.

    Output columns:
        user_id, item_id, label, user_interaction_count, item_interaction_count
    """

    logger.info('Building ranking features...')

    required = {"user_id","item_id"}
    if not required.issubset(interactions.columns):
        raise ValueError('Interactions must contain user_id and item_id')

    user_counts = interactions.groupby("user_id").size()
    item_counts = interactions.groupby("item_id").size()

    all_items = interactions["item_id"].unique()

    rows = []

    # ---------------------------------------
    # Positive samples
    # ---------------------------------------
    logger.info("Generating positive samples")

    for row in interactions.itertuples(index=False):
        rows.append(
            {
                "user_id": row.user_id,
                "item_id": row.item_id,
                "label": 1,
                "user_interaction_count": user_counts[row.user_id],
                "item_interaction_count": item_counts[row.item_id],
            }
        )

        # ---------------------------------------
        # Negative sampling
        # ---------------------------------------
        logger.info("Generating negative samples")

        user_item_sets = (
            interactions
            .groupby("user_id")["item_id"]
            .apply(set)
            .to_dict()
        )

        rng = np.random.default_rng(42)

        for user_id, seen_items in user_item_sets.items():

            candidate_items = np.setdiff1d(all_items, list(seen_items))

            if len(candidate_items) == 0:
                continue

            sampled_items = rng.choice(
                candidate_items,
                size=min(num_negative_samples * len(seen_items),len(candidate_items)),
                replace=False
            )

            for item_id in sampled_items:
                rows.append(
                    {
                        "user_id": user_id,
                        "item_id": item_id,
                        "label": 0,
                        "user_interaction_count": user_counts[user_id],
                        "item_interaction_count": item_counts.get(item_id, 0),
                    }
                )

        features = pd.DataFrame(rows)

        logger.info("Ranking feature dataset size: %d", len(features))

        return features