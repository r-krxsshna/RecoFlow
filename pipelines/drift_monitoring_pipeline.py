from src.monitoring.drift import run_drift_monitoring
from src.common.logging import get_logger

logger = get_logger(__name__)

def main():
    logger.info("Drift monitoring pipeline started")

    try:
        run_drift_monitoring()

    except Exception:
        logger.exception("Drift monitoring failed")
        raise

    logger.info("Drift monitoring pipeline finished")

if __name__ == "__main__":
    main()
