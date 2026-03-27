"""Step 1.1 — Load and filter Marina Bay Sands TripAdvisor reviews."""

import pandas as pd

from src.config import MBS_RAW_FILE, MBS_FILTERED_REVIEWS, PROCESSED_DATA_DIR


def load_raw_reviews() -> pd.DataFrame:
    """Read the raw Excel file and return the full DataFrame."""
    return pd.read_excel(MBS_RAW_FILE, engine="openpyxl")


def filter_high_rating(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to 4-5 star reviews and drop rows with missing/empty content."""
    df = df[df["ratings"].isin([4, 5])].copy()
    df = df.dropna(subset=["content"])
    df = df[df["content"].str.strip().astype(bool)]
    return df


def standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Reset index, add review_id, rename columns for downstream consistency."""
    df = df.reset_index(drop=True)
    df.insert(0, "review_id", range(len(df)))
    df = df.rename(columns={
        "ratings": "rating",
        "date_of_stay": "date",
        "content": "review_text",
        "title": "review_title",
    })
    return df[["review_id", "rating", "date", "review_title", "review_text"]]


def load_and_filter_reviews(save: bool = True) -> pd.DataFrame:
    """Full pipeline: load raw → filter → standardise → optionally save.

    Returns the filtered DataFrame with columns:
        review_id, rating, date, review_title, review_text
    """
    df_raw = load_raw_reviews()
    print(f"Raw dataset: {len(df_raw)} reviews")

    df = filter_high_rating(df_raw)
    print(f"After 4-5 star filter + null drop: {len(df)} reviews")

    df = standardise_columns(df)

    if save:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(MBS_FILTERED_REVIEWS, index=False)
        print(f"Saved to {MBS_FILTERED_REVIEWS}")

    return df
