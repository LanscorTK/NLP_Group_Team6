"""Phase 3 — Annotation utilities.

Functions for sampling sentences, computing inter-annotator agreement,
resolving labels, and creating train/test splits.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split

from src.config import (
    MBS_SENTENCES,
    MBS_CALIBRATION_SHEET,
    MBS_ANNOTATED_FULL,
    MBS_ANNOTATED_TRAIN,
    MBS_ANNOTATED_TEST,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    MODAL_VERBS,
    IMPERATIVE_SIGNALS,
    CONDITIONAL_SIGNALS,
)

ANNOTATORS = ["chase", "clara", "leah", "xiayi"]


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_for_annotation(
    df_sents: pd.DataFrame | None = None,
    n: int = 100,
    seed: int = RANDOM_SEED,
    enrich: bool = True,
    enrich_ratio: float = 0.5,
) -> pd.DataFrame:
    """Sample sentences for annotation.

    If enrich=True, enrich_ratio of the sample comes from sentences
    containing suggestion signals (modal verbs, imperatives, conditionals)
    to increase the expected suggestion rate. The remaining are random.
    This enrichment strategy must be documented in the report.
    """
    if df_sents is None:
        df_sents = pd.read_csv(MBS_SENTENCES)

    if not enrich:
        return df_sents.sample(n=n, random_state=seed).reset_index(drop=True)

    # Identify sentences with suggestion signals
    all_signals = MODAL_VERBS + IMPERATIVE_SIGNALS + CONDITIONAL_SIGNALS
    pattern = "|".join(rf"\b{s}\b" for s in all_signals)
    has_signal = df_sents["sentence_text"].str.lower().str.contains(pattern, regex=True)

    signal_pool = df_sents[has_signal]
    random_pool = df_sents[~has_signal]

    n_signal = min(int(n * enrich_ratio), len(signal_pool))
    n_random = n - n_signal

    rng = np.random.RandomState(seed)
    signal_sample = signal_pool.sample(n=n_signal, random_state=rng)
    random_sample = random_pool.sample(n=n_random, random_state=rng)

    result = pd.concat([signal_sample, random_sample]).sample(frac=1, random_state=seed)
    return result.reset_index(drop=True)


def export_calibration_sheet(
    df_sample: pd.DataFrame,
    output_path=MBS_CALIBRATION_SHEET,
) -> None:
    """Export a CSV for the IAA calibration round.

    Columns: sentence_id, sentence_text, label_chase, label_clara, label_leah, label_xiayi
    Label columns are empty — annotators fill them in manually.
    """
    out = df_sample[["sentence_id", "sentence_text"]].copy()
    for name in ANNOTATORS:
        out[f"label_{name}"] = ""

    out.to_csv(output_path, index=False)
    print(f"Exported calibration sheet ({len(out)} sentences) to {output_path}")
    print(f"Columns: {list(out.columns)}")


def sample_additional(
    df_sents: pd.DataFrame | None = None,
    already_sampled_ids: list | set = None,
    n: int = 300,
    seed: int = RANDOM_SEED + 1,
    enrich: bool = True,
) -> pd.DataFrame:
    """Sample additional sentences, excluding already-annotated ones."""
    if df_sents is None:
        df_sents = pd.read_csv(MBS_SENTENCES)

    if already_sampled_ids is not None:
        df_sents = df_sents[~df_sents["sentence_id"].isin(already_sampled_ids)]

    return sample_for_annotation(df_sents, n=n, seed=seed, enrich=enrich)


def export_split_batches(
    df_additional: pd.DataFrame,
    output_dir=PROCESSED_DATA_DIR,
) -> None:
    """Split additional sentences into 4 batches (one per annotator) and export."""
    n = len(df_additional)
    batch_size = n // len(ANNOTATORS)

    for i, name in enumerate(ANNOTATORS):
        start = i * batch_size
        end = start + batch_size if i < len(ANNOTATORS) - 1 else n
        batch = df_additional.iloc[start:end][["sentence_id", "sentence_text"]].copy()
        batch["label"] = ""
        batch["annotator"] = name

        path = output_dir / f"annotation_batch_{name}.csv"
        batch.to_csv(path, index=False)
        print(f"  Batch for {name}: {len(batch)} sentences -> {path}")


# ---------------------------------------------------------------------------
# IAA computation
# ---------------------------------------------------------------------------

def compute_pairwise_kappa(labels_df: pd.DataFrame) -> pd.DataFrame:
    """Compute Cohen's kappa for each pair of annotators.

    labels_df must have columns: label_chase, label_clara, label_leah, label_xiayi
    with integer values (0 or 1).
    """
    label_cols = [f"label_{name}" for name in ANNOTATORS]
    rows = []

    for i in range(len(ANNOTATORS)):
        for j in range(i + 1, len(ANNOTATORS)):
            a, b = label_cols[i], label_cols[j]
            kappa = cohen_kappa_score(labels_df[a], labels_df[b])
            rows.append({
                "annotator_1": ANNOTATORS[i],
                "annotator_2": ANNOTATORS[j],
                "kappa": round(kappa, 3),
            })

    return pd.DataFrame(rows)


def compute_fleiss_kappa(labels_df: pd.DataFrame) -> float:
    """Compute Fleiss' kappa for all annotators.

    labels_df must have columns: label_chase, label_clara, label_leah, label_xiayi
    """
    label_cols = [f"label_{name}" for name in ANNOTATORS]
    n_items = len(labels_df)
    n_raters = len(label_cols)
    n_categories = 2  # binary

    # Build the count matrix: for each item, count how many raters chose each category
    counts = np.zeros((n_items, n_categories), dtype=int)
    for col in label_cols:
        vals = labels_df[col].astype(int).values
        for cat in range(n_categories):
            counts[:, cat] += (vals == cat).astype(int)

    # Fleiss' kappa formula
    p_j = counts.sum(axis=0) / (n_items * n_raters)
    Pe = (p_j ** 2).sum()

    Pi = ((counts ** 2).sum(axis=1) - n_raters) / (n_raters * (n_raters - 1))
    P_bar = Pi.mean()

    kappa = (P_bar - Pe) / (1 - Pe) if (1 - Pe) != 0 else 1.0
    return round(kappa, 3)


# ---------------------------------------------------------------------------
# Label resolution & splitting
# ---------------------------------------------------------------------------

def resolve_labels(labels_df: pd.DataFrame, strategy: str = "majority") -> pd.DataFrame:
    """Resolve multi-annotator labels into a single gold label.

    strategy='majority': majority vote (for ties in 4-rater case, flag for discussion)
    Returns DataFrame with: sentence_id, sentence_text, label, agreement_count, needs_discussion
    """
    label_cols = [f"label_{name}" for name in ANNOTATORS]

    result = labels_df[["sentence_id", "sentence_text"]].copy()
    labels_matrix = labels_df[label_cols].astype(int)

    votes_for_1 = labels_matrix.sum(axis=1)
    n_raters = len(label_cols)

    result["label"] = (votes_for_1 > n_raters / 2).astype(int)
    result["agreement_count"] = labels_matrix.apply(
        lambda row: max(row.sum(), n_raters - row.sum()), axis=1
    )
    result["needs_discussion"] = votes_for_1 == n_raters / 2  # exactly 2v2 tie

    n_discuss = result["needs_discussion"].sum()
    if n_discuss > 0:
        print(f"WARNING: {n_discuss} sentences have a 2-vs-2 tie and need manual resolution")

    return result


def merge_annotations(
    calibration_path=MBS_CALIBRATION_SHEET,
    batch_dir=PROCESSED_DATA_DIR,
    save_full=True,
) -> pd.DataFrame:
    """Merge calibration gold labels + split annotation batches into one DataFrame.

    Uses the `gold_label` column from the calibration CSV directly (preserves
    the 11 consensus-resolved ties). Loads all 4 annotator batch CSVs and
    concatenates with calibration labels.

    Returns DataFrame with columns: sentence_id, sentence_text, label
    """
    # --- Calibration (100 sentences with resolved gold labels) ---
    cal = pd.read_csv(calibration_path)
    cal_merged = cal[["sentence_id", "sentence_text", "gold_label"]].rename(
        columns={"gold_label": "label"}
    )
    cal_merged["label"] = cal_merged["label"].astype(int)
    print(f"  Calibration: {len(cal_merged)} sentences "
          f"({cal_merged['label'].sum()} suggestions)")

    # --- Batch annotations (4 annotators × ~75 sentences each) ---
    batches = []
    for name in ANNOTATORS:
        path = batch_dir / f"annotation_batch_{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Batch file not found: {path}")
        batch = pd.read_csv(path)
        batch = batch[["sentence_id", "sentence_text", "label"]].copy()
        batch["label"] = batch["label"].astype(int)
        print(f"  Batch {name}: {len(batch)} sentences ({batch['label'].sum()} suggestions)")
        batches.append(batch)

    batch_merged = pd.concat(batches, ignore_index=True)

    # --- Combine ---
    df_all = pd.concat([cal_merged, batch_merged], ignore_index=True)

    # Sanity checks
    n_dups = df_all["sentence_id"].duplicated().sum()
    if n_dups > 0:
        print(f"  WARNING: {n_dups} duplicate sentence_id(s) found — dropping duplicates")
        df_all = df_all.drop_duplicates(subset="sentence_id", keep="first")

    n_total = len(df_all)
    n_sugg = df_all["label"].sum()
    print(f"\n  Total annotated: {n_total} sentences "
          f"({n_sugg} suggestions, {n_sugg / n_total:.1%})")

    if save_full:
        df_all.to_csv(MBS_ANNOTATED_FULL, index=False)
        print(f"  Saved full merged file to {MBS_ANNOTATED_FULL}")

    return df_all


def create_train_test_split(
    df_annotated: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = RANDOM_SEED,
    save: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create stratified train/test split from annotated data."""
    train, test = train_test_split(
        df_annotated,
        test_size=test_size,
        stratify=df_annotated["label"],
        random_state=seed,
    )

    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    print(f"Train: {len(train)} sentences ({(train['label'] == 1).mean():.1%} suggestions)")
    print(f"Test:  {len(test)} sentences ({(test['label'] == 1).mean():.1%} suggestions)")

    if save:
        train.to_csv(MBS_ANNOTATED_TRAIN, index=False)
        test.to_csv(MBS_ANNOTATED_TEST, index=False)
        print(f"Saved to {MBS_ANNOTATED_TRAIN} and {MBS_ANNOTATED_TEST}")

    return train, test
