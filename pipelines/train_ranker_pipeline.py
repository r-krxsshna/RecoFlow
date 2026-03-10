from src.common.logging import get_logger
from src.ranking.train_ranker import train_ranker

logger = get_logger(__name__)

def main():
    logger.info("Ranker training pipeline started")

    try:
        train_ranker()

        logger.info("Ranker model training completed successfully")

    except Exception:
        logger.exception("Ranker training pipeline failed")
        raise

    logger.info("Ranker training pipeline finished")

if __name__ == "__main__":
    main()