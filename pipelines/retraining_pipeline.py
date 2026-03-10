from src.monitoring.drift import run_drift_monitoring
from src.common.logging import get_logger

from pipelines.train_candidate_pipeline import main as train_candidate
from pipelines.train_ranker_pipeline import main as train_ranker

logger = get_logger(__name__)

def main():

    logger.info("Retraining pipeline started")

    try:

        drift_detected = run_drift_monitoring()

        if drift_detected:

            logger.warning("Drift detected. Starting retraining pipelines")

            train_candidate()

            train_ranker()

            logger.info("Retraining completed successfully")

        else:

            logger.info("No drift detected. Skipping retraining")

    except Exception:

        logger.exception("Retraining pipeline failed")
        raise

    logger.info("Retraining pipeline finished")

if __name__ == "__main__":
    main()




