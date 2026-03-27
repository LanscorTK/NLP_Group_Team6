"""Phase 4 — Baseline classifiers: Regex + TF-IDF + Logistic Regression.

Regex uses curated patterns from config.REGEX_PATTERNS.
TF-IDF+LR is trained on SemEval data and evaluated on both SemEval and MBS test sets.
"""

import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from src.config import (
    FIGURES_DIR,
    MBS_ANNOTATED_TEST,
    MBS_SENTENCES,
    RANDOM_SEED,
    REGEX_PATTERNS,
)
from src.evaluation import (
    build_comparison_table,
    compute_metrics,
    compute_pr_auc,
    load_semeval_test,
    load_semeval_train,
    plot_confusion_matrix,
    plot_pr_curve,
    print_report,
)


# ---------------------------------------------------------------------------
# Regex classifier
# ---------------------------------------------------------------------------

# Compile patterns once at module level
_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in REGEX_PATTERNS]


def regex_classify(texts):
    """Classify texts using regex pattern matching.

    Returns list of predicted labels (0 or 1).
    A sentence is labelled 1 (suggestion) if ANY pattern matches.
    """
    predictions = []
    for text in texts:
        text_lower = text.lower()
        matched = any(p.search(text_lower) for p in _COMPILED_PATTERNS)
        predictions.append(1 if matched else 0)
    return predictions


def get_regex_matches(text):
    """Return which patterns matched for a given text (for error analysis)."""
    text_lower = text.lower()
    return [
        REGEX_PATTERNS[i]
        for i, p in enumerate(_COMPILED_PATTERNS)
        if p.search(text_lower)
    ]


# ---------------------------------------------------------------------------
# TF-IDF + Logistic Regression
# ---------------------------------------------------------------------------

def train_tfidf_lr(train_texts, train_labels):
    """Train a TF-IDF + Logistic Regression pipeline.

    Args:
        train_texts: list of strings
        train_labels: list of int (0 or 1)

    Returns:
        (vectorizer, model) — fitted TfidfVectorizer and LogisticRegression
    """
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=10000,
        min_df=2,
        sublinear_tf=True,
    )
    X_train = vectorizer.fit_transform(train_texts)

    model = LogisticRegression(
        class_weight="balanced",
        C=1.0,
        max_iter=1000,
        random_state=RANDOM_SEED,
    )
    model.fit(X_train, train_labels)

    return vectorizer, model


def predict_tfidf_lr(vectorizer, model, texts):
    """Predict labels using a fitted TF-IDF+LR pipeline."""
    X = vectorizer.transform(texts)
    return model.predict(X).tolist()


def predict_proba_tfidf_lr(vectorizer, model, texts):
    """Predict probability of class 1 (suggestion) using a fitted TF-IDF+LR pipeline."""
    X = vectorizer.transform(texts)
    return model.predict_proba(X)[:, 1]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_baselines(include_mbs=None):
    """Run all baseline models and return results list.

    Args:
        include_mbs: If True, evaluate on MBS annotated test set.
                     If None, auto-detect (True if mbs_annotated_test.csv exists).
    """
    if include_mbs is None:
        include_mbs = MBS_ANNOTATED_TEST.exists()

    results = []

    # --- Load SemEval data ---
    print("\n--- Loading SemEval data ---")
    df_train = load_semeval_train()
    df_test = load_semeval_test()
    print(f"  SemEval train: {len(df_train)} sentences ({df_train['label'].sum()} suggestions)")
    print(f"  SemEval test:  {len(df_test)} sentences ({df_test['label'].sum()} suggestions)")

    # --- Regex on SemEval test ---
    print("\n--- Regex baseline → SemEval test ---")
    y_true_se = df_test["label"].tolist()
    y_pred_regex_se = regex_classify(df_test["text"].tolist())
    metrics = compute_metrics(y_true_se, y_pred_regex_se)
    results.append({"model": "Regex", "dataset": "SemEval", **metrics})
    print_report(y_true_se, y_pred_regex_se, "Regex — SemEval Test")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(y_true_se, y_pred_regex_se, "Regex — SemEval Test",
                          save_path=FIGURES_DIR / "cm_regex_semeval.png")

    # --- TF-IDF+LR on SemEval ---
    print("\n--- TF-IDF + LR → train on SemEval, evaluate on SemEval test ---")
    vectorizer, lr_model = train_tfidf_lr(
        df_train["text"].tolist(), df_train["label"].tolist()
    )
    y_pred_lr_se = predict_tfidf_lr(vectorizer, lr_model, df_test["text"].tolist())
    y_proba_lr_se = predict_proba_tfidf_lr(vectorizer, lr_model, df_test["text"].tolist())
    metrics = compute_metrics(y_true_se, y_pred_lr_se)
    metrics["pr_auc"] = compute_pr_auc(y_true_se, y_proba_lr_se)
    results.append({"model": "TF-IDF + LR", "dataset": "SemEval", **metrics})
    print_report(y_true_se, y_pred_lr_se, "TF-IDF + LR — SemEval Test")
    print(f"  PR-AUC: {metrics['pr_auc']:.4f}")
    plot_confusion_matrix(y_true_se, y_pred_lr_se, "TF-IDF + LR — SemEval Test",
                          save_path=FIGURES_DIR / "cm_tfidf_lr_semeval.png")

    # PR curve for SemEval (TF-IDF+LR only — regex has no probabilities)
    plot_pr_curve(y_true_se, {"TF-IDF + LR": y_proba_lr_se},
                  title="Precision-Recall Curve — SemEval Test (Baselines)",
                  save_path=FIGURES_DIR / "pr_curve_baselines_semeval.png")

    # --- 🔲 NEEDS_LABELS: MBS evaluation ---
    if include_mbs:
        print("\n--- Loading MBS annotated test set ---")
        df_mbs_test = pd.read_csv(MBS_ANNOTATED_TEST)
        y_true_mbs = df_mbs_test["label"].tolist()
        texts_mbs = df_mbs_test["sentence_text"].tolist()
        print(f"  MBS test: {len(df_mbs_test)} sentences ({df_mbs_test['label'].sum()} suggestions)")

        # Regex on MBS
        print("\n--- Regex baseline → MBS test ---")
        y_pred_regex_mbs = regex_classify(texts_mbs)
        metrics = compute_metrics(y_true_mbs, y_pred_regex_mbs)
        results.append({"model": "Regex", "dataset": "MBS", **metrics})
        print_report(y_true_mbs, y_pred_regex_mbs, "Regex — MBS Test")
        plot_confusion_matrix(y_true_mbs, y_pred_regex_mbs, "Regex — MBS Test",
                              save_path=FIGURES_DIR / "cm_regex_mbs.png")

        # TF-IDF+LR on MBS
        print("\n--- TF-IDF + LR → MBS test (cross-domain) ---")
        y_pred_lr_mbs = predict_tfidf_lr(vectorizer, lr_model, texts_mbs)
        y_proba_lr_mbs = predict_proba_tfidf_lr(vectorizer, lr_model, texts_mbs)
        metrics = compute_metrics(y_true_mbs, y_pred_lr_mbs)
        metrics["pr_auc"] = compute_pr_auc(y_true_mbs, y_proba_lr_mbs)
        results.append({"model": "TF-IDF + LR", "dataset": "MBS", **metrics})
        print_report(y_true_mbs, y_pred_lr_mbs, "TF-IDF + LR — MBS Test")
        print(f"  PR-AUC: {metrics['pr_auc']:.4f}")
        plot_confusion_matrix(y_true_mbs, y_pred_lr_mbs, "TF-IDF + LR — MBS Test",
                              save_path=FIGURES_DIR / "cm_tfidf_lr_mbs.png")
    else:
        print("\n  ⚠ MBS annotated test set not found — skipping MBS evaluation.")
        print("    Run Phase 3 annotation first, then re-run baselines with include_mbs=True.")

    # --- Print comparison ---
    if results:
        print("\n--- Baseline Comparison ---")
        table = build_comparison_table(results)
        print(table.to_string(index=False))

    return results
