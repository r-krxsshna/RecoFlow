import pandas as pd
from src.common.logging import get_logger

logger = get_logger(__name__)

REQUIRED_COLUMNS = ["user_id","item_id","rating","timestamp"]

class DataValidationError(Exception):
    pass

def validate_schema(df: pd.DataFrame) -> None:
    logger.info("Checking interaction schema")
    missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_columns:
        raise DataValidationError(f"Missing columns: {missing_columns}")

    logger.info("Schema validation passed")

def validate_nulls(df: pd.DataFrame) -> None:
    logger.info("Checking null values")
    if df[REQUIRED_COLUMNS].isnull().any().any():
        raise DataValidationError(f"Null values found in required columns")

    logger.info("Null check passed")


def validate_rating_range(df: pd.DataFrame, min_rating = 1.0, max_rating = 5.0) -> None:
    logger.info("Checking rating range")
    invalid = df[
        (df["rating"] < min_rating) | (df["rating"] > max_rating)
    ]

    if not invalid.empty:
        raise DataValidationError(
            f"Rating range must be between {min_rating} and {max_rating}"
        )

    logger.info("Rating range validation passed")

def validate_duplicates(df: pd.DataFrame) -> None:
    logger.info("Checking duplicate interactions")
    duplicates = df.duplicated(subset=["user_id", "item_id"])
    if duplicates.any():
        raise DataValidationError(
            "Duplicate user-item interactions found"
        )

    logger.info("Duplicate check passed")

def validate_interactions(df: pd.DataFrame):
    logger.info("Starting interaction validations")

    validate_schema(df)
    validate_nulls(df)
    validate_rating_range(df)
    validate_duplicates(df)

    logger.info("All interaction validations passed")
