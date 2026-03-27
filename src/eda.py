"""Phase 2 — Exploratory Data Analysis.

Functions for dataset statistics, modal verb analysis, and domain gap comparison
between MBS hotel reviews and SemEval-2019 software forum data.
"""

from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

from src.config import (
    FIGURES_DIR,
    MBS_FILTERED_REVIEWS,
    MBS_SENTENCES,
    SEMEVAL_TRAIN,
    SEMEVAL_TEST_LABELED,
    MODAL_VERBS,
    IMPERATIVE_SIGNALS,
    CONDITIONAL_SIGNALS,
)

sns.set_theme(style="white", font_scale=1.3)
plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.titlepad": 12,
    "axes.labelpad": 8,
    "xtick.major.pad": 5,
    "ytick.major.pad": 5,
})

# Colourblind-friendly palette
_CLR_PRIMARY = "#2c7bb6"    # steel blue
_CLR_SECONDARY = "#27ae60"  # emerald green
_CLR_ACCENT = "#e67e22"     # warm amber
_CLR_MEDIAN = "#c0392b"     # muted red


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_semeval_train() -> pd.DataFrame:
    """Load SemEval-2019 Task 9 training data."""
    df = pd.read_csv(SEMEVAL_TRAIN, header=None, names=["id", "text", "label"])
    df["label"] = df["label"].astype(int)
    return df


def load_semeval_test() -> pd.DataFrame:
    """Load SemEval-2019 Task 9 labelled evaluation data."""
    df = pd.read_csv(SEMEVAL_TEST_LABELED, header=None, names=["id", "text", "label"])
    df["label"] = df["label"].astype(int)
    return df


# ---------------------------------------------------------------------------
# Step 2.1 — MBS dataset statistics
# ---------------------------------------------------------------------------

def plot_rating_distribution(df_reviews: pd.DataFrame, save: bool = True):
    """Bar chart of 4-star vs 5-star review counts."""
    fig, ax = plt.subplots(figsize=(7, 5))
    counts = df_reviews["rating"].value_counts().sort_index()
    bars = counts.plot.bar(ax=ax, color=[_CLR_PRIMARY, _CLR_SECONDARY], edgecolor="white",
                           width=0.55)
    ax.set_xlabel("Star Rating")
    ax.set_ylabel("Number of Reviews")
    ax.set_title("Review Distribution by Star Rating")
    ax.bar_label(ax.containers[0], fmt=lambda x: f"{x:,.0f}", fontsize=12, fontweight="bold",
                 padding=4)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    sns.despine(ax=ax)
    fig.tight_layout()
    if save:
        _save_fig(fig, "rating_distribution.png")
    return fig


def plot_review_length_distribution(df_reviews: pd.DataFrame, save: bool = True):
    """Histogram of word count per review."""
    fig, ax = plt.subplots(figsize=(9, 5))
    word_counts = df_reviews["review_text"].str.split().str.len()
    ax.hist(word_counts, bins=50, color=_CLR_PRIMARY, edgecolor="white", alpha=0.85)
    ax.axvline(word_counts.median(), color=_CLR_MEDIAN, linestyle="--", linewidth=2,
               label=f"Median: {word_counts.median():.0f} words")
    ax.set_xlabel("Words per Review")
    ax.set_ylabel("Frequency")
    ax.set_title("Review Length Distribution")
    ax.legend(frameon=False, fontsize=11)
    sns.despine(ax=ax)
    fig.tight_layout()
    if save:
        _save_fig(fig, "review_length_distribution.png")
    return fig


def plot_sentence_length_distribution(df_sents: pd.DataFrame, save: bool = True):
    """Histogram of token count per sentence."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(df_sents["n_tokens"], bins=50, color=_CLR_SECONDARY, edgecolor="white", alpha=0.85)
    ax.axvline(df_sents["n_tokens"].median(), color=_CLR_MEDIAN, linestyle="--", linewidth=2,
               label=f"Median: {df_sents['n_tokens'].median():.0f} tokens")
    ax.set_xlabel("Tokens per Sentence")
    ax.set_ylabel("Frequency")
    ax.set_title("Sentence Length Distribution (MBS)")
    ax.legend(frameon=False, fontsize=11)
    sns.despine(ax=ax)
    fig.tight_layout()
    if save:
        _save_fig(fig, "sentence_length_distribution.png")
    return fig


def plot_temporal_distribution(df_reviews: pd.DataFrame, save: bool = True):
    """Time series of reviews per month."""
    fig, ax = plt.subplots(figsize=(11, 5))
    df = df_reviews.dropna(subset=["date"]).copy()
    df["month"] = pd.to_datetime(df["date"]).dt.to_period("M")
    monthly = df.groupby("month").size()
    monthly.index = monthly.index.to_timestamp()
    ax.plot(monthly.index, monthly.values, color=_CLR_PRIMARY, linewidth=2)
    ax.fill_between(monthly.index, monthly.values, alpha=0.15, color=_CLR_PRIMARY)
    ax.set_xlabel("Date")
    ax.set_ylabel("Reviews per Month")
    ax.set_title("Temporal Distribution of Reviews")
    sns.despine(ax=ax)
    fig.tight_layout()
    if save:
        _save_fig(fig, "temporal_distribution.png")
    return fig


def compute_top_ngrams(
    df_sents: pd.DataFrame, n: int = 1, top_k: int = 30, text_col: str = "sentence_text"
) -> pd.DataFrame:
    """Compute top-k n-grams from sentences."""
    vec = CountVectorizer(ngram_range=(n, n), stop_words="english", max_features=top_k)
    X = vec.fit_transform(df_sents[text_col])
    freqs = X.sum(axis=0).A1
    terms = vec.get_feature_names_out()
    result = pd.DataFrame({"term": terms, "count": freqs})
    return result.sort_values("count", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 2.2 — Modal verb & suggestion signal analysis
# ---------------------------------------------------------------------------

def count_suggestion_signals(
    df: pd.DataFrame, text_col: str = "sentence_text"
) -> pd.DataFrame:
    """Count occurrences of suggestion signals in a sentence DataFrame.

    Returns a DataFrame with columns: signal, category, count, pct.
    """
    texts_lower = df[text_col].str.lower()
    total = len(df)
    rows = []

    for word in MODAL_VERBS:
        c = texts_lower.str.contains(rf"\b{word}\b", regex=True).sum()
        rows.append({"signal": word, "category": "modal", "count": c, "pct": c / total * 100})

    for word in IMPERATIVE_SIGNALS:
        c = texts_lower.str.contains(rf"\b{word}\b", regex=True).sum()
        rows.append({"signal": word, "category": "imperative", "count": c, "pct": c / total * 100})

    for phrase in CONDITIONAL_SIGNALS:
        c = texts_lower.str.contains(phrase, regex=False).sum()
        rows.append({"signal": phrase, "category": "conditional", "count": c, "pct": c / total * 100})

    return pd.DataFrame(rows)


def plot_signal_comparison(
    mbs_signals: pd.DataFrame, semeval_signals: pd.DataFrame, save: bool = True
):
    """Side-by-side bar chart comparing suggestion signal frequencies."""
    fig, ax = plt.subplots(figsize=(12, 7))

    merged = mbs_signals[["signal", "pct"]].rename(columns={"pct": "MBS"}).merge(
        semeval_signals[["signal", "pct"]].rename(columns={"pct": "SemEval"}),
        on="signal",
    )
    merged = merged.sort_values("MBS", ascending=True)

    y = range(len(merged))
    h = 0.35
    ax.barh([i + h / 2 for i in y], merged["MBS"], h, label="MBS (hotel)",
            color=_CLR_PRIMARY, edgecolor="white")
    ax.barh([i - h / 2 for i in y], merged["SemEval"], h, label="SemEval (forum)",
            color=_CLR_ACCENT, edgecolor="white")
    ax.set_yticks(list(y))
    ax.set_yticklabels(merged["signal"])
    ax.set_xlabel("% of Sentences Containing Signal")
    ax.set_title("Suggestion Signal Frequency: MBS vs. SemEval")
    ax.legend(frameon=False, fontsize=11)
    ax.xaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)
    fig.tight_layout()
    if save:
        _save_fig(fig, "signal_comparison.png")
    return fig


# ---------------------------------------------------------------------------
# Step 2.3 — Domain gap analysis
# ---------------------------------------------------------------------------

def compare_sentence_lengths(
    df_mbs: pd.DataFrame, df_semeval: pd.DataFrame, save: bool = True
):
    """Overlaid histograms of sentence lengths for both domains."""
    fig, ax = plt.subplots(figsize=(9, 5))

    mbs_lens = df_mbs["n_tokens"] if "n_tokens" in df_mbs.columns else df_mbs["text"].str.split().str.len()
    sem_lens = df_semeval["text"].str.split().str.len()

    ax.hist(mbs_lens, bins=50, alpha=0.65, label=f"MBS (n={len(df_mbs):,})",
            color=_CLR_PRIMARY, density=True, edgecolor="white")
    ax.hist(sem_lens, bins=50, alpha=0.65, label=f"SemEval (n={len(df_semeval):,})",
            color=_CLR_ACCENT, density=True, edgecolor="white")
    ax.set_xlabel("Tokens per Sentence")
    ax.set_ylabel("Density")
    ax.set_title("Sentence Length Distribution: MBS vs. SemEval")
    ax.legend(frameon=False, fontsize=11)
    sns.despine(ax=ax)
    fig.tight_layout()
    if save:
        _save_fig(fig, "domain_sentence_lengths.png")
    return fig


def compute_vocab_overlap(
    df_mbs: pd.DataFrame, df_semeval: pd.DataFrame,
    mbs_text_col: str = "sentence_text", sem_text_col: str = "text",
) -> dict:
    """Compute vocabulary overlap between two corpora."""
    mbs_words = set()
    for text in tqdm(df_mbs[mbs_text_col].dropna(), desc="Building MBS vocab"):
        mbs_words.update(text.lower().split())

    sem_words = set()
    for text in tqdm(df_semeval[sem_text_col].dropna(), desc="Building SemEval vocab"):
        sem_words.update(text.lower().split())

    overlap = mbs_words & sem_words
    return {
        "mbs_vocab_size": len(mbs_words),
        "semeval_vocab_size": len(sem_words),
        "overlap_size": len(overlap),
        "jaccard": len(overlap) / len(mbs_words | sem_words),
        "mbs_coverage": len(overlap) / len(mbs_words),
        "semeval_coverage": len(overlap) / len(sem_words),
    }


def compute_domain_specific_terms(
    df_mbs: pd.DataFrame, df_semeval: pd.DataFrame,
    mbs_text_col: str = "sentence_text", sem_text_col: str = "text",
    top_k: int = 20,
) -> pd.DataFrame:
    """Find top terms unique to each domain."""
    def _word_counts(series, desc="Counting words"):
        counter = Counter()
        for text in tqdm(series.dropna(), desc=desc):
            counter.update(text.lower().split())
        return counter

    mbs_counts = _word_counts(df_mbs[mbs_text_col], desc="MBS term counts")
    sem_counts = _word_counts(df_semeval[sem_text_col], desc="SemEval term counts")

    # Terms with high frequency in one domain but absent/rare in the other
    mbs_only = {w: c for w, c in mbs_counts.items() if sem_counts.get(w, 0) < 5}
    sem_only = {w: c for w, c in sem_counts.items() if mbs_counts.get(w, 0) < 5}

    mbs_top = sorted(mbs_only.items(), key=lambda x: -x[1])[:top_k]
    sem_top = sorted(sem_only.items(), key=lambda x: -x[1])[:top_k]

    rows = []
    for (mw, mc), (sw, sc) in zip(mbs_top, sem_top):
        rows.append({"mbs_term": mw, "mbs_count": mc, "semeval_term": sw, "semeval_count": sc})
    return pd.DataFrame(rows)


def domain_gap_summary_table(
    df_mbs: pd.DataFrame, df_semeval: pd.DataFrame,
    mbs_text_col: str = "sentence_text", sem_text_col: str = "text",
) -> pd.DataFrame:
    """Summary table comparing the two domains for the report."""
    sem_lens = df_semeval[sem_text_col].str.split().str.len()
    mbs_lens = df_mbs[mbs_text_col].str.split().str.len()

    suggestion_rate_sem = (df_semeval["label"] == 1).mean() * 100 if "label" in df_semeval.columns else None

    vocab = compute_vocab_overlap(df_mbs, df_semeval, mbs_text_col, sem_text_col)

    rows = [
        {"Metric": "Number of sentences", "MBS": f"{len(df_mbs):,}", "SemEval": f"{len(df_semeval):,}"},
        {"Metric": "Median sentence length (words)", "MBS": f"{mbs_lens.median():.0f}", "SemEval": f"{sem_lens.median():.0f}"},
        {"Metric": "Mean sentence length (words)", "MBS": f"{mbs_lens.mean():.1f}", "SemEval": f"{sem_lens.mean():.1f}"},
        {"Metric": "Vocabulary size", "MBS": f"{vocab['mbs_vocab_size']:,}", "SemEval": f"{vocab['semeval_vocab_size']:,}"},
        {"Metric": "Vocabulary overlap (Jaccard)", "MBS": f"{vocab['jaccard']:.3f}", "SemEval": "—"},
        {"Metric": "Suggestion base rate (%)", "MBS": "TBD (after annotation)", "SemEval": f"{suggestion_rate_sem:.1f}" if suggestion_rate_sem else "N/A"},
        {"Metric": "Domain", "MBS": "Hotel reviews (TripAdvisor)", "SemEval": "Software forums (Windows Phone)"},
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Run all EDA
# ---------------------------------------------------------------------------

def run_eda():
    """Run all EDA steps and save figures."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df_reviews = pd.read_csv(MBS_FILTERED_REVIEWS)
    df_sents = pd.read_csv(MBS_SENTENCES)
    df_semeval = load_semeval_train()

    print("--- Step 2.1: MBS dataset statistics ---")
    plot_rating_distribution(df_reviews)
    plot_review_length_distribution(df_reviews)
    plot_sentence_length_distribution(df_sents)
    plot_temporal_distribution(df_reviews)

    print("Top 30 unigrams:")
    uni = compute_top_ngrams(df_sents, n=1, top_k=30)
    print(uni.to_string(index=False))

    print("\nTop 30 bigrams:")
    bi = compute_top_ngrams(df_sents, n=2, top_k=30)
    print(bi.to_string(index=False))

    print("\n--- Step 2.2: Suggestion signal analysis ---")
    mbs_signals = count_suggestion_signals(df_sents, text_col="sentence_text")
    sem_signals = count_suggestion_signals(df_semeval, text_col="text")
    print("MBS signals:")
    print(mbs_signals.to_string(index=False))
    print("\nSemEval signals:")
    print(sem_signals.to_string(index=False))
    plot_signal_comparison(mbs_signals, sem_signals)

    print("\n--- Step 2.3: Domain gap analysis ---")
    compare_sentence_lengths(df_sents, df_semeval)
    vocab = compute_vocab_overlap(df_sents, df_semeval)
    print(f"Vocab overlap: {vocab}")

    domain_terms = compute_domain_specific_terms(df_sents, df_semeval)
    print("\nDomain-specific terms:")
    print(domain_terms.to_string(index=False))

    summary = domain_gap_summary_table(df_sents, df_semeval)
    print("\nDomain gap summary:")
    print(summary.to_string(index=False))

    plt.close("all")
    print(f"\nFigures saved to {FIGURES_DIR}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_fig(fig, filename: str):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / filename, dpi=300, bbox_inches="tight")
