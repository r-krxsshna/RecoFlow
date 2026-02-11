from pathlib import Path
import pandas as pd
from src.common.logging import  get_logger

logger = get_logger(__name__)

def load_interactions(path: Path) -> pd.DataFrame:
    logger.info(f"Loading raw interactions from {path}")
    if not path.exists():
        logger.error(f"Raw data not found at {path}")
        raise FileNotFoundError(f"Raw interactions not found at {path}.")

    df = pd.read_csv(
        path,
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"]
    )

    logger.info(f"Loaded {len(df)} interactions from raw data")

    return df
