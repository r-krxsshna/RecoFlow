from pathlib import Path
import pandas as pd

from src.ingestion.load_interactions import load_interactions
from src.validation.interaction_checks import validate_interactions
from src.common.logging import get_logger

logger = get_logger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent

RAW_PATH = BASE_DIR / "data" / "raw" / "u.data"
PROCESSED_PATH = BASE_DIR / "data" / "processed" / "interactions.parquet"

def validate_and_save_interactions(df: pd.DataFrame, path: Path) -> None:
    logger.info("Starting validation and persistence step")

    try:
        validate_interactions(df)
        logger.info("Interaction validation successful")
    except Exception:
        logger.exception("Interaction validation failed")
        raise

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path,index=False)

    logger.info(f"Saved processed interactions to {PROCESSED_PATH}")

def ingest() -> pd.DataFrame:
    logger.info("Starting ingestion pipeline")

    df = load_interactions(RAW_PATH)

    df["user_id"] = df["user_id"].astype(int)
    df["item_id"] = df["item_id"].astype(int)
    df["rating"] = df["rating"].astype(int)
    df["timestamp"] = df["timestamp"].astype(int)

    logger.info("Running validation and saving processed data")

    try:
        validate_and_save_interactions(df, PROCESSED_PATH)
    except Exception:
        logger.exception("Validation or saving failed in ingestion step")
        raise

    return df