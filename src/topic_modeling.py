"""Phase 5 — Topic modeling on extracted suggestions.

Uses BERTopic for clustering predicted suggestion sentences, with keyword-based
aspect grouping as a supplementary analysis.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud

from src.config import (
    FIGURES_DIR,
    HOTEL_ASPECTS,
    MBS_PREDICTIONS,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_predicted_suggestions(predictions_path=None):
    """Load predicted suggestions from mbs_predictions.csv.

    Returns DataFrame filtered to predicted_label == 1.
    """
    if predictions_path is None:
        predictions_path = MBS_PREDICTIONS

    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Predictions file not found: {predictions_path}\n"
            "  Run notebook 06 (BERT stage-1 predictions on 70k) first."
        )

    df = pd.read_csv(predictions_path)
    df_sugg = df[df["predicted_label"] == 1].copy().reset_index(drop=True)
    print(f"  Loaded {len(df_sugg)} predicted suggestions out of {len(df)} total sentences "
          f"({len(df_sugg)/len(df):.1%})")
    return df_sugg


# ---------------------------------------------------------------------------
# BERTopic clustering
# ---------------------------------------------------------------------------

def run_bertopic(texts, min_topic_size=5):
    """Run BERTopic clustering on a list of texts.

    Args:
        texts: list of suggestion sentences
        min_topic_size: minimum cluster size for HDBSCAN

    Returns:
        (topic_model, topics, probs)
    """
    from bertopic import BERTopic
    from umap import UMAP
    from hdbscan import HDBSCAN

    # Adjust for small datasets
    n_docs = len(texts)
    if n_docs < 50:
        min_topic_size = max(3, min_topic_size)
        print(f"  Small dataset ({n_docs} docs) — using min_topic_size={min_topic_size}")

    umap_model = UMAP(
        n_components=5,
        n_neighbors=min(15, n_docs - 1),
        min_dist=0.0,
        metric="cosine",
        random_state=RANDOM_SEED,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_topic_size,
        min_samples=max(1, min_topic_size - 2),
        metric="euclidean",
        prediction_data=True,
    )

    topic_model = BERTopic(
        embedding_model="all-MiniLM-L6-v2",
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        nr_topics="auto",
        verbose=True,
    )

    topics, probs = topic_model.fit_transform(texts)

    n_topics = len(set(topics)) - (1 if -1 in topics else 0)
    n_outliers = sum(1 for t in topics if t == -1)
    print(f"  Found {n_topics} topics ({n_outliers} outlier sentences)")

    return topic_model, topics, probs


def get_topic_summary(topic_model, docs, topics):
    """Build a summary DataFrame of topics with top words and representative sentences.

    Returns DataFrame with: topic_id, count, top_words, representative_sentences.
    """
    topic_info = topic_model.get_topic_info()
    rows = []
    for _, row in topic_info.iterrows():
        topic_id = row["Topic"]
        if topic_id == -1:
            continue  # skip outlier topic
        count = row["Count"]
        # Top words
        top_words = [w for w, _ in topic_model.get_topic(topic_id)[:8]]
        # Representative docs
        topic_docs = [d for d, t in zip(docs, topics) if t == topic_id]
        rep_docs = topic_docs[:5]
        rows.append({
            "topic_id": topic_id,
            "count": count,
            "top_words": ", ".join(top_words),
            "representative_sentences": rep_docs,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_topic_barchart(topic_model, save_path=None):
    """Bar chart of topic frequencies (excluding outliers)."""
    topic_info = topic_model.get_topic_info()
    topic_info = topic_info[topic_info["Topic"] != -1].head(15)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        [f"Topic {t}" for t in topic_info["Topic"]],
        topic_info["Count"],
        color="steelblue",
    )
    ax.set_xlabel("Number of Sentences")
    ax.set_title("Suggestion Topics — Frequency")
    ax.invert_yaxis()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"  Saved: {save_path}")
    plt.close(fig)
    return fig


def plot_topic_wordclouds(topic_model, n_topics=6, save_path=None):
    """Word clouds for the top N topics."""
    topic_info = topic_model.get_topic_info()
    topic_ids = [t for t in topic_info["Topic"] if t != -1][:n_topics]

    n_cols = min(3, len(topic_ids))
    n_rows = (len(topic_ids) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, topic_id in enumerate(topic_ids):
        words = dict(topic_model.get_topic(topic_id)[:20])
        # Ensure all weights are positive for wordcloud
        words = {k: max(v, 0.001) for k, v in words.items()}
        wc = WordCloud(
            width=400, height=300, background_color="white",
            colormap="viridis", max_words=20,
        ).generate_from_frequencies(words)
        axes[i].imshow(wc, interpolation="bilinear")
        axes[i].set_title(f"Topic {topic_id}", fontsize=12)
        axes[i].axis("off")

    # Hide unused axes
    for j in range(len(topic_ids), len(axes)):
        axes[j].axis("off")

    plt.suptitle("Top Suggestion Topics — Word Clouds", fontsize=14)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"  Saved: {save_path}")
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Aspect-based grouping (supplementary)
# ---------------------------------------------------------------------------

def aspect_grouping(texts, aspects=None):
    """Classify suggestion sentences into hotel aspects using keyword matching.

    Args:
        texts: list of suggestion sentences
        aspects: dict mapping aspect name to keyword list (defaults to HOTEL_ASPECTS)

    Returns DataFrame with: sentence_text, aspect (or "other" if no match).
    """
    if aspects is None:
        aspects = HOTEL_ASPECTS

    rows = []
    for text in texts:
        text_lower = text.lower()
        matched_aspect = "other"
        best_count = 0
        for aspect_name, keywords in aspects.items():
            count = sum(1 for kw in keywords if kw in text_lower)
            if count > best_count:
                best_count = count
                matched_aspect = aspect_name
        rows.append({"sentence_text": text, "aspect": matched_aspect})

    return pd.DataFrame(rows)


def get_aspect_summary(df_aspects):
    """Summarise aspect distribution with example sentences."""
    rows = []
    for aspect, group in df_aspects.groupby("aspect"):
        rows.append({
            "aspect": aspect,
            "count": len(group),
            "percentage": f"{len(group)/len(df_aspects):.1%}",
            "examples": group["sentence_text"].head(3).tolist(),
        })
    df = pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)
    return df


def plot_aspect_distribution(df_aspects, save_path=None):
    """Bar chart of aspect distribution."""
    counts = df_aspects["aspect"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 5))
    counts.plot(kind="bar", ax=ax, color="teal")
    ax.set_ylabel("Number of Suggestions")
    ax.set_title("Suggestions by Hotel Aspect")
    ax.set_xlabel("")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"  Saved: {save_path}")
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_topic_modeling(predictions_path=None):
    """Run full Phase 5 pipeline.

    Returns dict with topic_model, summary_df, aspect_df.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load suggestions
    print("\n--- Loading predicted suggestions ---")
    df_sugg = load_predicted_suggestions(predictions_path)
    texts = df_sugg["sentence_text"].tolist()

    # 2. BERTopic
    print("\n--- Running BERTopic ---")
    topic_model, topics, probs = run_bertopic(texts)

    # 3. Summary
    print("\n--- Topic Summary ---")
    summary_df = get_topic_summary(topic_model, texts, topics)
    for _, row in summary_df.iterrows():
        print(f"\n  Topic {row['topic_id']} ({row['count']} sentences)")
        print(f"    Words: {row['top_words']}")
        for sent in row["representative_sentences"][:3]:
            print(f"    - \"{sent[:100]}{'...' if len(sent) > 100 else ''}\"")

    # 4. Visualisations
    print("\n--- Generating visualisations ---")
    plot_topic_barchart(topic_model, save_path=FIGURES_DIR / "topic_barchart.png")
    plot_topic_wordclouds(topic_model, save_path=FIGURES_DIR / "topic_wordclouds.png")

    # 5. Aspect grouping
    print("\n--- Aspect-based grouping ---")
    df_aspects = aspect_grouping(texts)
    aspect_summary = get_aspect_summary(df_aspects)
    print(aspect_summary[["aspect", "count", "percentage"]].to_string(index=False))
    plot_aspect_distribution(df_aspects, save_path=FIGURES_DIR / "aspect_distribution.png")

    # 6. Save summaries
    summary_path = PROCESSED_DATA_DIR / "topic_summary.csv"
    summary_df.drop(columns=["representative_sentences"]).to_csv(summary_path, index=False)
    print(f"\n  Topic summary saved: {summary_path}")

    aspect_path = PROCESSED_DATA_DIR / "aspect_summary.csv"
    aspect_summary.drop(columns=["examples"]).to_csv(aspect_path, index=False)
    print(f"  Aspect summary saved: {aspect_path}")

    return {
        "topic_model": topic_model,
        "summary_df": summary_df,
        "aspect_df": df_aspects,
        "aspect_summary": aspect_summary,
    }
