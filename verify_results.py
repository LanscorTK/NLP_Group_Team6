"""Verify pipeline outputs match expected results for reproducibility.

Usage:
    python verify_results.py          # after running python run_pipeline.py

Checks data row counts, prediction statistics, and figure existence
against the values recorded in docs/results_cheatsheet.md.
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "outputs" / "figures"

passed = 0
failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}  --  {detail}")


def check_row_count(path, expected, label=None):
    label = label or path.name
    if not path.exists():
        check(label, False, f"file not found: {path}")
        return None
    df = pd.read_csv(path)
    check(f"{label}: {len(df):,} rows", len(df) == expected,
          f"expected {expected:,}, got {len(df):,}")
    return df


# =====================================================================
print("=" * 60)
print("REPRODUCIBILITY VERIFICATION")
print("=" * 60)

# --- Phase 1: Data ---
print("\n--- Phase 1: Data Loading & Preprocessing ---")
check_row_count(PROCESSED / "mbs_filtered_reviews.csv", 8571)
check_row_count(PROCESSED / "mbs_sentences.csv", 70085)

# --- Phase 3: Annotations ---
print("\n--- Phase 3: Annotations ---")
df_full = check_row_count(PROCESSED / "mbs_annotated_full.csv", 400)
if df_full is not None:
    n_sugg = int(df_full["label"].sum())
    check(f"Suggestions in full set: {n_sugg}", n_sugg == 73,
          f"expected 73, got {n_sugg}")

check_row_count(PROCESSED / "mbs_annotated_train.csv", 320)
check_row_count(PROCESSED / "mbs_annotated_test.csv", 80)
check_row_count(PROCESSED / "mbs_calibration_100.csv", 100)

# --- Phase 4: Predictions ---
print("\n--- Phase 4: Model Predictions ---")
pred_path = PROCESSED / "mbs_predictions.csv"
if pred_path.exists():
    df_pred = pd.read_csv(pred_path)
    check(f"Predictions file: {len(df_pred):,} rows", len(df_pred) == 70085,
          f"expected 70,085, got {len(df_pred):,}")
    n_pred_sugg = int(df_pred["predicted_label"].sum())
    # Allow ±200 tolerance for BERT stochasticity across hardware
    check(f"Predicted suggestions: {n_pred_sugg:,}",
          abs(n_pred_sugg - 4179) <= 200,
          f"expected ~4,179 (±200), got {n_pred_sugg:,}")
else:
    check("Predictions file exists", False, f"not found: {pred_path}")

# --- Phase 4: Model metrics (recompute from test set + models) ---
print("\n--- Phase 4: Model Metrics (recomputed) ---")
test_path = PROCESSED / "mbs_annotated_test.csv"
if test_path.exists():
    from sklearn.metrics import f1_score
    df_test = pd.read_csv(test_path)
    y_true = df_test["label"].values

    # Regex baseline
    from src.baselines import regex_classify
    texts = df_test["sentence_text"].tolist()
    y_regex = regex_classify(texts)
    f1_regex = round(f1_score(y_true, y_regex, pos_label=1), 3)
    check(f"Regex MBS F1: {f1_regex:.3f}", abs(f1_regex - 0.471) < 0.01,
          f"expected ~0.471, got {f1_regex:.3f}")

    # TF-IDF + LR
    from src.baselines import train_tfidf_lr, predict_tfidf_lr
    from src.evaluation import load_semeval_train
    df_se = load_semeval_train()
    vec, model_lr = train_tfidf_lr(df_se["text"].tolist(), df_se["label"].tolist())
    y_lr = predict_tfidf_lr(vec, model_lr, texts)
    f1_lr = round(f1_score(y_true, y_lr, pos_label=1), 3)
    check(f"TF-IDF+LR MBS F1: {f1_lr:.3f}", abs(f1_lr - 0.190) < 0.01,
          f"expected ~0.190, got {f1_lr:.3f}")

    # BERT Stage 1
    s1_dir = ROOT / "outputs" / "models" / "bert_stage1"
    s2_dir = ROOT / "outputs" / "models" / "bert_stage2"

    if (s1_dir / "config.json").exists():
        from src.bert_model import predict, get_device
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        device = get_device()

        tok_s1 = AutoTokenizer.from_pretrained(s1_dir)
        mdl_s1 = AutoModelForSequenceClassification.from_pretrained(s1_dir)
        y_s1 = predict(mdl_s1, tok_s1, texts, device)
        f1_s1 = round(f1_score(y_true, y_s1, pos_label=1), 3)
        check(f"BERT S1 MBS F1: {f1_s1:.3f}", abs(f1_s1 - 0.286) < 0.03,
              f"expected ~0.286 (±0.02), got {f1_s1:.3f}")
    else:
        check("BERT Stage 1 checkpoint", False, f"not found: {s1_dir}")

    if (s2_dir / "config.json").exists():
        tok_s2 = AutoTokenizer.from_pretrained(s2_dir)
        mdl_s2 = AutoModelForSequenceClassification.from_pretrained(s2_dir)
        y_s2 = predict(mdl_s2, tok_s2, texts, device)
        f1_s2 = round(f1_score(y_true, y_s2, pos_label=1), 3)
        check(f"BERT S2 MBS F1: {f1_s2:.3f}", abs(f1_s2 - 0.581) < 0.03,
              f"expected ~0.581 (±0.02), got {f1_s2:.3f}")
    else:
        check("BERT Stage 2 checkpoint", False, f"not found: {s2_dir}")
else:
    print("  SKIP  (test set not found — cannot recompute metrics)")

# --- Phase 5: Topic Modeling ---
print("\n--- Phase 5: Topic Modeling ---")
topic_path = PROCESSED / "topic_summary.csv"
if topic_path.exists():
    df_topics = pd.read_csv(topic_path)
    n_topics = len(df_topics)
    check(f"Topics found: {n_topics}", abs(n_topics - 56) <= 10,
          f"expected ~56 (±10), got {n_topics}")
else:
    check("Topic summary file exists", False, f"not found: {topic_path}")

aspect_path = PROCESSED / "aspect_summary.csv"
if aspect_path.exists():
    df_aspects = pd.read_csv(aspect_path)
    check(f"Aspects: {len(df_aspects)}", len(df_aspects) == 7,
          f"expected 7, got {len(df_aspects)}")
else:
    check("Aspect summary file exists", False, f"not found: {aspect_path}")

# --- Figures ---
print("\n--- Figures ---")
EXPECTED_FIGURES = [
    # EDA
    "rating_distribution.png", "review_length_distribution.png",
    "sentence_length_distribution.png", "temporal_distribution.png",
    "signal_comparison.png", "domain_sentence_lengths.png",
    # Evaluation
    "cm_regex_semeval.png", "cm_regex_mbs.png",
    "cm_tfidf_lr_semeval.png", "cm_tfidf_lr_mbs.png",
    "cm_bert_s1_semeval.png", "cm_bert_s1_mbs.png", "cm_bert_s2_mbs.png",
    "pr_curve_baselines_semeval.png", "pr_curve_bert_s1_semeval.png",
    "pr_curve_bert_mbs.png", "model_comparison.png",
    # Error analysis
    "error_breakdown_regex.png", "error_breakdown_tf-idf__lr.png",
    "error_breakdown_bert_semeval.png", "error_breakdown_bert_semevalmbs.png",
    "error_comparison_models.png",
    # Topic modeling
    "topic_barchart.png", "topic_wordclouds.png", "aspect_distribution.png",
]

missing = [f for f in EXPECTED_FIGURES if not (FIGURES / f).exists()]
empty = [f for f in EXPECTED_FIGURES
         if (FIGURES / f).exists() and (FIGURES / f).stat().st_size == 0]

check(f"All {len(EXPECTED_FIGURES)} figures exist",
      len(missing) == 0,
      f"missing: {missing}")
check("No empty figure files",
      len(empty) == 0,
      f"empty: {empty}")

# =====================================================================
print("\n" + "=" * 60)
print(f"RESULTS:  {passed} passed,  {failed} failed")
print("=" * 60)

sys.exit(0 if failed == 0 else 1)
