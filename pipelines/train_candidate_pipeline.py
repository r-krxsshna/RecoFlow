from src.ingestion.ingest import ingest
from src.common.logging import get_logger

logger = get_logger(__name__)

def main():
    logger.info("Candidate training pipeline started")

    try:
        df = ingest()
        logger.info("Ingestion completed. %d interactions loaded", len(df))
    except Exception:
        logger.exception("Candidate pipeline failed")
        raise

    logger.info("Candidate training pipeline finished successfully")

if __name__ == "__main__":
    main()