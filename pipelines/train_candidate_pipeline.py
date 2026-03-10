from src.ingestion.ingest import ingest
from src.common.logging import get_logger
from src.splitting.user_split import temporal_user_split,save_splits
from src.candidate_generation.train_cf import train_candidate_model

logger = get_logger(__name__)

def main():
    logger.info("Candidate training pipeline started")

    try:
        df = ingest()

        train_df, validation_df, test_df = temporal_user_split(df)
        save_splits(train_df, validation_df, test_df)

        train_candidate_model(train_df = train_df)



        logger.info("Ingestion completed. %d interactions loaded", len(df))
    except Exception:
        logger.exception("Candidate pipeline failed")
        raise

    logger.info("Candidate training pipeline finished successfully")

if __name__ == "__main__":
    main()