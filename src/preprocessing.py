"""Step 1.2 — Sentence segmentation and cleaning."""

import pandas as pd
import spacy
from tqdm import tqdm

from src.config import (
    MBS_FILTERED_REVIEWS,
    MBS_SENTENCES,
    MIN_SENTENCE_TOKENS,
    MAX_SENTENCE_TOKENS,
    MIN_ASCII_RATIO,
)


def _ascii_ratio(text: str) -> float:
    """Fraction of characters that are ASCII."""
    if not text:
        return 0.0
    return sum(1 for c in text if ord(c) < 128) / len(text)


def segment_sentences(df_reviews: pd.DataFrame) -> pd.DataFrame:
    """Segment reviews into sentences using spaCy.

    Returns a DataFrame with columns:
        review_id, rating, sentence_text, n_tokens
    """
    nlp = spacy.load("en_core_web_sm")
    nlp.max_length = 200_000

    rows = []
    for _, review in tqdm(df_reviews.iterrows(), total=len(df_reviews), desc="Segmenting sentences"):
        doc = nlp(review["review_text"])
        for sent in doc.sents:
            text = sent.text.strip()
            if text:
                rows.append({
                    "review_id": review["review_id"],
                    "rating": review["rating"],
                    "sentence_text": text,
                    "n_tokens": len(sent),
                })

    df = pd.DataFrame(rows)
    print(f"Segmented {len(df_reviews)} reviews into {len(df)} sentences "
          f"({len(df) / len(df_reviews):.1f} avg per review)")
    return df


def filter_language(df: pd.DataFrame) -> pd.DataFrame:
    """Remove non-English sentences using ASCII ratio heuristic."""
    tqdm.pandas(desc="Language filtering")
    ratios = df["sentence_text"].progress_apply(_ascii_ratio)
    mask = ratios >= MIN_ASCII_RATIO
    removed = (~mask).sum()
    print(f"Language filter: removed {removed} non-English sentences "
          f"(threshold: {MIN_ASCII_RATIO})")
    return df[mask].copy()


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact-duplicate sentences."""
    before = len(df)
    df = df.drop_duplicates(subset=["sentence_text"], keep="first")
    print(f"Dedup: removed {before - len(df)} duplicate sentences")
    return df


def filter_length(df: pd.DataFrame) -> pd.DataFrame:
    """Remove sentences outside the token-length window."""
    before = len(df)
    df = df[
        (df["n_tokens"] >= MIN_SENTENCE_TOKENS)
        & (df["n_tokens"] <= MAX_SENTENCE_TOKENS)
    ].copy()
    print(f"Length filter ({MIN_SENTENCE_TOKENS}-{MAX_SENTENCE_TOKENS} tokens): "
          f"removed {before - len(df)} sentences")
    return df


def preprocess_sentences(save: bool = True) -> pd.DataFrame:
    """Full pipeline: load filtered reviews → segment → clean → save.

    Returns the sentence DataFrame with columns:
        sentence_id, review_id, rating, sentence_text, n_tokens
    """
    df_reviews = pd.read_csv(MBS_FILTERED_REVIEWS)
    print(f"Loaded {len(df_reviews)} filtered reviews")

    df = segment_sentences(df_reviews)
    df = filter_language(df)
    df = deduplicate(df)
    df = filter_length(df)

    # Assign sentence IDs
    df = df.reset_index(drop=True)
    df.insert(0, "sentence_id", range(len(df)))
    df = df[["sentence_id", "review_id", "rating", "sentence_text", "n_tokens"]]

    print(f"Final: {len(df)} sentences from {df['review_id'].nunique()} reviews")

    if save:
        df.to_csv(MBS_SENTENCES, index=False)
        print(f"Saved to {MBS_SENTENCES}")

    return df
