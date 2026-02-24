from pathlib import Path
import pandas as pd

from src.candidate_generation.generate_candidates import generate_candidate_for_users

BASE_DIR = Path(__file__).resolve().parent.parent.parent

PROCESSED_DIR = BASE_DIR / "data" / "processed" /  "splits"

TRAIN_INTERACTIONS_PATH = PROCESSED_DIR / "train.parquet"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "user_candidates.parquet"


def main(): # For development purpose only

    train_df = pd.read_parquet(TRAIN_INTERACTIONS_PATH)

    user_ids = train_df["user_id"].unique().tolist()

    candidates = generate_candidate_for_users(
        user_ids=user_ids,
        train_interactions=train_df,
        k=50
    )

    candidates.to_parquet(OUTPUT_PATH, index=False)

    print(f"Saved candidates to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
