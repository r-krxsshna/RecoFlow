from pathlib import Path
import pandas as pd

from src.common.logging import get_logger

BASE_DIR = Path(__file__).resolve().parent.parent.parent

logger = get_logger(__name__)

def temporal_user_split(interactions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    logger.info("Starting temporal user-based split")

    required = {"user_id","item_id","timestamp"}
    if not required.issubset(interactions.columns):
        raise ValueError("Missing required columns for splitting")

    interactions = interactions.sort_values(["user_id","timestamp"])

    train_parts = []
    validation_parts = []
    test_parts = []

    for user_id,user_df in interactions.groupby("user_id"):
        if len(user_df) < 3 :
            train_parts.append(user_df)
            continue

        train = user_df.iloc[:-2]
        validation = user_df.iloc[-2:-1]
        test = user_df.iloc[-1:]

        train_parts.append(train)
        validation_parts.append(validation)
        test_parts.append(test)

    train_df = pd.concat(train_parts).reset_index(drop=True)
    validation_df = pd.concat(validation_parts).reset_index(drop=True)
    test_df = pd.concat(test_parts).reset_index(drop=True)

    logger.info(
        "Finished temporal user-based split: train=%d, val=%d, test=%d",
        len(train_df), len(validation_df), len(test_df)
    )

    return train_df, validation_df, test_df

def save_splits(train_df: pd.DataFrame,validation_df: pd.DataFrame,test_df: pd.DataFrame) -> None:
    logger.info("Saving splits...")

    output_dir = BASE_DIR / "data" / "processed" / "splits"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.parquet"
    validation_path = output_dir / "validation.parquet"
    test_path = output_dir / "test.parquet"

    train_df.to_parquet(train_path, index=False)
    validation_df.to_parquet(validation_path, index=False)
    test_df.to_parquet(test_path, index=False)

    logger.info("Saved splits to %s", output_dir)

