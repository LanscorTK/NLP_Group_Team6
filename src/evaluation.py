"""Shared evaluation utilities for all models.

Provides SemEval data loading, metrics computation, confusion matrix plotting,
and cross-model comparison table generation.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    accuracy_score,
)

from src.config import FIGURES_DIR, SEMEVAL_TRAIN, SEMEVAL_TEST_LABELED

sns.set_theme(style="white", font_scale=1.3)
plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.titlepad": 12,
    "axes.labelpad": 8,
})

_CLR_PRIMARY = "#2c7bb6"
_CLR_ACCENT = "#e67e22"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_semeval_data(path=None):
    """Load a SemEval CSV file (id, text, label).

    Handles double-quoted text fields with internal commas.
    Returns DataFrame with columns: id, text, label.
    """
    if path is None:
        path = SEMEVAL_TRAIN
    df = pd.read_csv(path, header=None, names=["id", "text", "label"])
    df["text"] = df["text"].astype(str).str.strip().str.strip('"')
    df["label"] = df["label"].astype(int)
    return df


def load_semeval_train():
    """Load SemEval training data."""
    return load_semeval_data(SEMEVAL_TRAIN)


def load_semeval_test():
    """Load SemEval labeled test data."""
    return load_semeval_data(SEMEVAL_TEST_LABELED)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred):
    """Compute classification metrics.

    Returns dict with: precision, recall, f1 (all macro),
    precision_1, recall_1, f1_1 (suggestion class only), accuracy.
    """
    return {
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_1": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_1": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
    }


def compute_pr_auc(y_true, y_proba):
    """Compute Precision-Recall AUC (Average Precision) for the suggestion class.

    Args:
        y_true: ground truth labels (0 or 1)
        y_proba: predicted probability of class 1 (suggestion)

    Returns float PR-AUC score.
    """
    return average_precision_score(y_true, y_proba)


def print_report(y_true, y_pred, title=""):
    """Print a sklearn classification report with an optional title."""
    if title:
        print(f"\n{'=' * 50}")
        print(f"  {title}")
        print(f"{'=' * 50}")
    print(classification_report(
        y_true, y_pred,
        target_names=["Non-Suggestion", "Suggestion"],
        digits=4,
        zero_division=0,
    ))


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", save_path=None):
    """Plot a 2x2 confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Non-Sugg", "Sugg"],
        yticklabels=["Non-Sugg", "Sugg"],
        ax=ax, annot_kws={"size": 16, "weight": "bold"},
        linewidths=0.5, linecolor="white",
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)
    return fig


def plot_pr_curve(y_true, y_proba_dict, title="Precision-Recall Curve", save_path=None):
    """Plot Precision-Recall curves for multiple models on one figure.

    Args:
        y_true: ground truth labels (0 or 1)
        y_proba_dict: dict mapping model name to predicted probabilities of class 1
                      e.g. {"TF-IDF + LR": proba_array, "BERT": proba_array}
    """
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    for model_name, y_proba in y_proba_dict.items():
        precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_proba)
        ap = average_precision_score(y_true, y_proba)
        ax.plot(recall_vals, precision_vals, linewidth=2,
                label=f"{model_name} (AP={ap:.3f})")

    # Baseline: prevalence line (random classifier)
    prevalence = np.mean(y_true)
    ax.axhline(y=prevalence, color="grey", linestyle="--", alpha=0.5,
               label=f"Baseline (prevalence={prevalence:.2f})")

    ax.set_xlabel("Recall (Suggestion)")
    ax.set_ylabel("Precision (Suggestion)")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower left", frameon=False, fontsize=10)
    sns.despine(ax=ax)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)
    return fig


def build_comparison_table(results):
    """Build a comparison DataFrame from a list of result dicts.

    Each dict should have: model, dataset, precision, recall, f1
    (plus optional precision_1, recall_1, f1_1, accuracy).
    """
    df = pd.DataFrame(results)
    # Lead with suggestion-class metrics (primary), then macro (secondary)
    display_cols = ["model", "dataset", "f1_1", "pr_auc", "precision_1", "recall_1",
                    "f1", "precision", "recall", "accuracy"]
    available = [c for c in display_cols if c in df.columns]
    df = df[available]
    # Round numeric columns
    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].round(4)
    return df


def plot_comparison_chart(comparison_df, metric="f1", save_path=None):
    """Grouped bar chart: metric by model and dataset."""
    if metric not in comparison_df.columns:
        print(f"  Metric '{metric}' not in comparison table, skipping chart.")
        return None
    pivot = comparison_df.pivot(index="model", columns="dataset", values=metric)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    pivot.plot(kind="bar", ax=ax, rot=15, color=[_CLR_PRIMARY, _CLR_ACCENT],
               edgecolor="white", width=0.7)
    metric_label = "Suggestion-Class F1" if metric == "f1_1" else metric.upper()
    ax.set_ylabel(metric_label)
    ax.set_title(f"Model Comparison — {metric_label} by Dataset")
    ax.set_xlabel("")
    ax.set_ylim(0, 1)
    ax.legend(title="Dataset", frameon=False, fontsize=11)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)
    return fig
