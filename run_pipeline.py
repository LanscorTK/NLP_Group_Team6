"""Run the full suggestion-mining pipeline end-to-end.

Usage:
    python run_pipeline.py              # run all phases
    python run_pipeline.py --phase 1    # run only Phase 1 (data loading + preprocessing)

This script calls the same src/ functions used by the notebooks,
ensuring reproducible results without needing Jupyter.
"""

import argparse
import sys

from src.data_loading import load_and_filter_reviews
from src.preprocessing import preprocess_sentences


def phase1():
    """Phase 1: Data loading & preprocessing."""
    print("=" * 60)
    print("PHASE 1: Data Loading & Preprocessing")
    print("=" * 60)

    print("\n--- Step 1.1: Load and filter reviews ---")
    load_and_filter_reviews(save=True)

    print("\n--- Step 1.2: Sentence segmentation & cleaning ---")
    preprocess_sentences(save=True)


def phase2():
    """Phase 2: Exploratory Data Analysis."""
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend for script mode
    from src.eda import run_eda
    print("\n" + "=" * 60)
    print("PHASE 2: Exploratory Data Analysis")
    print("=" * 60)
    run_eda()


def phase3():
    """Phase 3: Merge annotations and create train/test split."""
    from src.annotation import merge_annotations, create_train_test_split
    print("\n" + "=" * 60)
    print("PHASE 3: Annotation Merge & Train/Test Split")
    print("=" * 60)
    df_all = merge_annotations()
    create_train_test_split(df_all, test_size=0.2, save=True)


def phase4():
    """Phase 4: Model Development."""
    import matplotlib
    matplotlib.use("Agg")
    from src.baselines import run_baselines
    from src.bert_model import train_and_evaluate

    print("\n" + "=" * 60)
    print("PHASE 4: Model Development")
    print("=" * 60)

    # Auto-detect whether MBS labels exist
    from src.config import MBS_ANNOTATED_TEST
    has_labels = MBS_ANNOTATED_TEST.exists()
    if not has_labels:
        print("\n  ⚠ MBS annotated data not found — running SemEval-only evaluation.")
        print("    Complete Phase 3 annotation to unlock MBS evaluation + BERT Stage 2.\n")

    print("\n--- Baselines ---")
    baseline_results = run_baselines(include_mbs=has_labels)

    print("\n--- BERT ---")
    bert_results = train_and_evaluate(include_stage2=has_labels)

    # Combined comparison
    all_results = baseline_results + bert_results
    if all_results:
        from src.evaluation import build_comparison_table, plot_comparison_chart
        from src.config import FIGURES_DIR
        print("\n" + "=" * 60)
        print("  FULL MODEL COMPARISON")
        print("=" * 60)
        table = build_comparison_table(all_results)
        print(table.to_string(index=False))
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        plot_comparison_chart(table, metric="f1_1", save_path=FIGURES_DIR / "model_comparison.png")

    # --- Re-predict on full MBS dataset with best available model ---
    from src.config import MBS_SENTENCES, MBS_PREDICTIONS, MODELS_DIR
    from src.bert_model import predict
    import pandas as pd

    stage2_dir = MODELS_DIR / "bert_stage2"
    stage1_dir = MODELS_DIR / "bert_stage1"
    if (stage2_dir / "config.json").exists():
        best_dir, best_name = stage2_dir, "Stage 2"
    elif (stage1_dir / "config.json").exists():
        best_dir, best_name = stage1_dir, "Stage 1"
    else:
        best_dir = None

    if best_dir and MBS_SENTENCES.exists():
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        print(f"\n--- Re-predicting all MBS sentences with BERT {best_name} ---")
        tokenizer = AutoTokenizer.from_pretrained(best_dir)
        model = AutoModelForSequenceClassification.from_pretrained(best_dir)
        df_sents = pd.read_csv(MBS_SENTENCES)
        preds = predict(model, tokenizer, df_sents["sentence_text"].tolist())
        df_sents["predicted_label"] = preds
        df_sents.to_csv(MBS_PREDICTIONS, index=False)
        n_sugg = sum(preds)
        print(f"  Predictions ({best_name}): {n_sugg}/{len(preds)} suggestions "
              f"({n_sugg / len(preds):.1%})")
        print(f"  Saved to {MBS_PREDICTIONS}")


def phase5():
    """Phase 5: Insight Generation (Topic Modeling)."""
    import matplotlib
    matplotlib.use("Agg")
    from src.topic_modeling import run_topic_modeling
    from src.config import MBS_PREDICTIONS

    print("\n" + "=" * 60)
    print("PHASE 5: Insight Generation")
    print("=" * 60)

    if not MBS_PREDICTIONS.exists():
        print("\n  ⚠ Predictions file not found — run Phase 4 first to generate")
        print(f"    BERT predictions on all MBS sentences.")
        print(f"    Expected: {MBS_PREDICTIONS}")
        return

    run_topic_modeling()


def phase6():
    """Phase 6: Error Analysis."""
    import matplotlib
    matplotlib.use("Agg")
    from src.error_analysis import run_full_error_analysis
    print("\n" + "=" * 60)
    print("PHASE 6: Error Analysis")
    print("=" * 60)
    run_full_error_analysis()


PHASES = {
    1: phase1,
    2: phase2,
    3: phase3,
    4: phase4,
    5: phase5,
    6: phase6,
}


def main():
    parser = argparse.ArgumentParser(description="Run the suggestion-mining pipeline.")
    parser.add_argument(
        "--phase", type=int, default=None,
        help="Run a specific phase (1-5). Omit to run all available phases.",
    )
    args = parser.parse_args()

    if args.phase is not None:
        if args.phase not in PHASES:
            print(f"Phase {args.phase} is not yet implemented. Available: {sorted(PHASES.keys())}")
            sys.exit(1)
        PHASES[args.phase]()
    else:
        for phase_num in sorted(PHASES.keys()):
            PHASES[phase_num]()

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
