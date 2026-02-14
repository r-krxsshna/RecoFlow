from pathlib import Path
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
import joblib
from implicit.als import AlternatingLeastSquares

from src.common.logging import get_logger

logger = get_logger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent

def build_interaction_matrix(df: pd.DataFrame):
    """
    Builds user-item sparse matrix and id mappings
    """

    logger.info("Building interaction matrix")

    user_ids = df["user_id"].unique()
    item_ids = df["item_id"].unique()

    user_to_index = {u: i for i, u in enumerate(user_ids)}
    item_to_index = {i: j for j, i in enumerate(item_ids)}

    rows = df["user_id"].map(user_to_index)
    cols = df["item_id"].map(item_to_index)
    data = np.ones(len(df))

    matrix = coo_matrix(
        (data, (rows, cols)),
        shape=(len(user_ids), len(item_ids))
    ).tocsr()

    logger.info(
        "Interaction matrix built with shape %s", matrix.shape
    )

    return matrix, user_to_index, item_to_index

def train_candidate_model(
        train_df: pd.DataFrame,
        factors: int = 64,
        iterations: int = 20,
        regularization: float = 0.01,
):
    matrix, user_map, item_map = build_interaction_matrix(train_df)

    logger.info("Training ALS candidate model")

    model = AlternatingLeastSquares(
        factors=factors,
        regularization=regularization,
        iterations=iterations,
    )

    model.fit(matrix.T)

    model_dir = BASE_DIR / "models" / "candidate_model"

    model_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_dir / "als_model.joblib")
    logger.info(f"Saved ALS candidate model")

    joblib.dump(user_map, model_dir / "user_mapping.joblib")
    logger.info(f"Saved user mapping")

    joblib.dump(item_map, model_dir / "item_mapping.joblib")
    logger.info(f"Saved item mapping")

    logger.info("Candidate model saved to %s", model_dir)

    return model
