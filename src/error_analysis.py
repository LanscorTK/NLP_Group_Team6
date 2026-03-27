"""Phase 6 — Error analysis across all models on the MBS test set.

Categorises false positives and false negatives by suggestion type,
produces summary tables and figures for the report.
"""

import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import (
    FIGURES_DIR,
    MBS_ANNOTATED_TEST,
    MODELS_DIR,
)
from src.evaluation import compute_metrics, print_report

sns.set_theme(style="white", font_scale=1.3)
plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.titlepad": 12,
    "axes.labelpad": 8,
})

_CLR_FP = "#c0392b"
_CLR_FN = "#2c7bb6"

# ---------------------------------------------------------------------------
# Suggestion-type heuristics (keyword-based)
# ---------------------------------------------------------------------------

_CATEGORY_PATTERNS = {
    "explicit": [
        r"\bshould\b", r"\bneed to\b", r"\bought to\b",
        r"\bmust\b", r"\bhave to\b",
    ],
    "implicit": [
        r"\bi wish\b", r"\bit would be nice\b", r"\bhopefully\b",
        r"\bif only\b", r"\bwould like\b", r"\bwould love\b",
        r"\bwould be better\b", r"\bcould improve\b", r"\bcould be better\b",
        r"\bcould use\b",
    ],
    "comparative": [
        r"\bother hotels?\b", r"\bcompared to\b", r"\bbetter than\b",
        r"\bunlike\b", r"\belsewhere\b",
    ],
    "complaint": [
        r"\bdisappoint\w*\b", r"\bunfortunate\w*\b", r"\bterrible\b",
        r"\bawful\b", r"\bhorrible\b", r"\bworse\b", r"\bworst\b",
        r"\bpoor\b", r"\bbad\b", r"\brude\b",
    ],
    "positive_modal": [
        r"\bwould definitely\b", r"\bwould certainly\b",
        r"\bwould recommend\b", r"\bwould come back\b",
        r"\bwould stay\b", r"\bwould return\b", r"\bwould visit\b",
    ],
    "imperative": [
        r"\bplease\b", r"\brecommend\b", r"\bsuggest\b",
        r"\bconsider\b", r"\btry\b", r"\bmake sure\b",
        r"\bgo for\b", r"\bdon'?t forget\b", r"\bensure\b",
    ],
}

_COMPILED = {
    cat: [re.compile(p, re.IGNORECASE) for p in patterns]
    for cat, patterns in _CATEGORY_PATTERNS.items()
}


def _classify_sentence_category(text: str) -> str:
    """Assign a suggestion-type category to a sentence using keyword heuristics.

    Priority order matters: positive_modal checked before explicit to avoid
    misclassifying "would definitely return" as an explicit suggestion.
    """
    text_lower = text.lower()

    # Check positive_modal first (to avoid false hits on "would" patterns)
    if any(p.search(text_lower) for p in _COMPILED["positive_modal"]):
        return "positive_modal"
    if any(p.search(text_lower) for p in _COMPILED["implicit"]):
        return "implicit"
    if any(p.search(text_lower) for p in _COMPILED["explicit"]):
        return "explicit"
    if any(p.search(text_lower) for p in _COMPILED["imperative"]):
        return "imperative"
    if any(p.search(text_lower) for p in _COMPILED["comparative"]):
        return "comparative"
    if any(p.search(text_lower) for p in _COMPILED["complaint"]):
        return "complaint"
    return "other"


# ---------------------------------------------------------------------------
# Core analysis functions
# ---------------------------------------------------------------------------

def categorize_errors(y_true, y_pred, texts) -> pd.DataFrame:
    """Classify each prediction as TP/TN/FP/FN and assign suggestion category.

    Returns DataFrame with columns:
        sentence_text, y_true, y_pred, error_type, suggestion_category
    """
    records = []
    for text, yt, yp in zip(texts, y_true, y_pred):
        if yt == 1 and yp == 1:
            error_type = "TP"
        elif yt == 0 and yp == 0:
            error_type = "TN"
        elif yt == 0 and yp == 1:
            error_type = "FP"
        else:
            error_type = "FN"

        category = _classify_sentence_category(text)
        records.append({
            "sentence_text": text,
            "y_true": yt,
            "y_pred": yp,
            "error_type": error_type,
            "suggestion_category": category,
        })
    return pd.DataFrame(records)


def error_summary(df_errors: pd.DataFrame) -> pd.DataFrame:
    """Aggregate errors by suggestion_category and error_type.

    Returns a pivot table with categories as rows, error types as columns,
    and counts as values.
    """
    ct = pd.crosstab(
        df_errors["suggestion_category"],
        df_errors["error_type"],
        margins=True,
    )
    # Reorder columns for readability
    col_order = [c for c in ["TP", "FP", "FN", "TN", "All"] if c in ct.columns]
    return ct[col_order]


def get_error_examples(df_errors: pd.DataFrame, error_type: str, n: int = 5) -> pd.DataFrame:
    """Return up to n example sentences for a given error type."""
    subset = df_errors[df_errors["error_type"] == error_type]
    if len(subset) <= n:
        return subset[["sentence_text", "suggestion_category"]].reset_index(drop=True)
    return subset.sample(n=n, random_state=42)[["sentence_text", "suggestion_category"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_error_breakdown(df_errors: pd.DataFrame, model_name: str, save_path=None):
    """Stacked bar chart showing FP/FN counts by suggestion category."""
    errors_only = df_errors[df_errors["error_type"].isin(["FP", "FN"])]
    if errors_only.empty:
        print(f"  No errors to plot for {model_name}")
        return None

    ct = pd.crosstab(errors_only["suggestion_category"], errors_only["error_type"])
    for col in ["FP", "FN"]:
        if col not in ct.columns:
            ct[col] = 0

    ct = ct[["FP", "FN"]].sort_values(by=["FP", "FN"], ascending=False)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ct.plot(kind="bar", stacked=True, ax=ax, color=[_CLR_FP, _CLR_FN], edgecolor="white")
    ax.set_xlabel("Suggestion Category")
    ax.set_ylabel("Count")
    ax.set_title(f"Error Breakdown by Category — {model_name}")
    ax.legend(title="Error Type", frameon=False, fontsize=11)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)
    return fig


def plot_model_error_comparison(all_model_errors: dict, save_path=None):
    """Grouped bar chart comparing total FP and FN counts across models."""
    records = []
    for model_name, df_err in all_model_errors.items():
        n_fp = (df_err["error_type"] == "FP").sum()
        n_fn = (df_err["error_type"] == "FN").sum()
        records.append({"model": model_name, "FP": n_fp, "FN": n_fn})

    df = pd.DataFrame(records).set_index("model")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    df.plot(kind="bar", ax=ax, color=[_CLR_FP, _CLR_FN], edgecolor="white", width=0.7)
    ax.set_ylabel("Count")
    ax.set_xlabel("")
    ax.set_title("False Positives vs False Negatives by Model")
    ax.legend(title="Error Type", frameon=False, fontsize=11)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_full_error_analysis():
    """Run error analysis for all models on the MBS test set.

    Loads MBS test data, runs each model, categorizes errors, and produces
    summary tables and figures.
    """
    if not MBS_ANNOTATED_TEST.exists():
        print("  MBS annotated test set not found — cannot run error analysis.")
        print(f"  Expected: {MBS_ANNOTATED_TEST}")
        return None

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df_test = pd.read_csv(MBS_ANNOTATED_TEST)
    texts = df_test["sentence_text"].tolist()
    y_true = df_test["label"].tolist()
    print(f"  MBS test set: {len(df_test)} sentences ({sum(y_true)} suggestions)")

    # --- Collect predictions from each model ---
    model_predictions = {}

    # 1. Regex baseline
    from src.baselines import regex_classify
    model_predictions["Regex"] = regex_classify(texts)

    # 2. TF-IDF + LR (train on SemEval, predict on MBS)
    from src.baselines import train_tfidf_lr, predict_tfidf_lr
    from src.evaluation import load_semeval_train
    df_train = load_semeval_train()
    vec, lr = train_tfidf_lr(df_train["text"].tolist(), df_train["label"].tolist())
    model_predictions["TF-IDF + LR"] = predict_tfidf_lr(vec, lr, texts)

    # 3. BERT Stage-1 (SemEval only)
    stage1_dir = MODELS_DIR / "bert_stage1"
    if (stage1_dir / "config.json").exists():
        from src.bert_model import predict as bert_predict, get_device
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        print("\n  Loading BERT Stage-1...")
        tok_s1 = AutoTokenizer.from_pretrained(stage1_dir)
        mod_s1 = AutoModelForSequenceClassification.from_pretrained(stage1_dir)
        device = get_device()
        model_predictions["BERT (SemEval)"] = bert_predict(mod_s1, tok_s1, texts, device)
    else:
        print("  BERT Stage-1 checkpoint not found — skipping.")

    # 4. BERT Stage-2 (SemEval + MBS)
    stage2_dir = MODELS_DIR / "bert_stage2"
    if (stage2_dir / "config.json").exists():
        from src.bert_model import predict as bert_predict, get_device
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        print("\n  Loading BERT Stage-2...")
        tok_s2 = AutoTokenizer.from_pretrained(stage2_dir)
        mod_s2 = AutoModelForSequenceClassification.from_pretrained(stage2_dir)
        device = get_device()
        model_predictions["BERT (SemEval+MBS)"] = bert_predict(mod_s2, tok_s2, texts, device)
    else:
        print("  BERT Stage-2 checkpoint not found — skipping.")

    # --- Analyse each model ---
    all_model_errors = {}

    for model_name, y_pred in model_predictions.items():
        print(f"\n{'=' * 60}")
        print(f"  Error Analysis: {model_name}")
        print(f"{'=' * 60}")

        # Metrics summary
        metrics = compute_metrics(y_true, y_pred)
        print(f"  F1 (suggestion): {metrics['f1_1']:.4f}  |  "
              f"Precision: {metrics['precision_1']:.4f}  |  "
              f"Recall: {metrics['recall_1']:.4f}")
        print_report(y_true, y_pred, model_name)

        # Categorize
        df_err = categorize_errors(y_true, y_pred, texts)
        all_model_errors[model_name] = df_err

        # Summary table
        summary = error_summary(df_err)
        print(f"\n  Error breakdown by category:")
        print(summary.to_string())

        # FP examples
        fp_examples = get_error_examples(df_err, "FP")
        if not fp_examples.empty:
            print(f"\n  False Positive examples ({model_name}):")
            for _, row in fp_examples.iterrows():
                print(f"    [{row['suggestion_category']}] {row['sentence_text'][:100]}")

        # FN examples
        fn_examples = get_error_examples(df_err, "FN")
        if not fn_examples.empty:
            print(f"\n  False Negative examples ({model_name}):")
            for _, row in fn_examples.iterrows():
                print(f"    [{row['suggestion_category']}] {row['sentence_text'][:100]}")

        # Per-model error breakdown figure
        safe_name = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("+", "")
        plot_error_breakdown(
            df_err, model_name,
            save_path=FIGURES_DIR / f"error_breakdown_{safe_name}.png",
        )

    # --- Cross-model comparison figure ---
    if len(all_model_errors) > 1:
        plot_model_error_comparison(
            all_model_errors,
            save_path=FIGURES_DIR / "error_comparison_models.png",
        )

    return all_model_errors
