"""Phase 4 — BERT classifier: two-stage fine-tuning for suggestion mining.

Stage 1: Fine-tune bert-base-uncased on SemEval-2019 training data (software forums).
Stage 2: Further fine-tune on MBS annotated hotel review data (domain adaptation).

Supports MPS (Apple Silicon), CUDA, and CPU backends.
"""

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from tqdm import tqdm

from src.config import (
    BERT_BATCH_SIZE,
    BERT_EPOCHS_STAGE1,
    BERT_EPOCHS_STAGE2,
    BERT_LR_STAGE1,
    BERT_LR_STAGE2,
    BERT_MAX_LENGTH,
    BERT_MODEL_NAME,
    BERT_WARMUP_RATIO,
    BERT_WEIGHT_DECAY,
    FIGURES_DIR,
    MBS_ANNOTATED_TEST,
    MBS_ANNOTATED_TRAIN,
    MODELS_DIR,
    RANDOM_SEED,
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
# Device detection
# ---------------------------------------------------------------------------

def get_device():
    """Detect the best available device for PyTorch."""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"  Device: Apple Silicon MPS")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"  Device: CUDA ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print(f"  Device: CPU")
    return device


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_dataset(texts, labels, tokenizer, max_length=BERT_MAX_LENGTH):
    """Tokenize texts and return a HuggingFace Dataset."""
    encodings = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors=None,
    )
    dataset = Dataset.from_dict({
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "labels": labels,
    })
    dataset.set_format("torch")
    return dataset


def _compute_trainer_metrics(eval_pred):
    """Metric computation function for HuggingFace Trainer."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "f1": float(compute_metrics(labels, preds)["f1"]),
        "f1_suggestion": float(compute_metrics(labels, preds)["f1_1"]),
        "accuracy": float(compute_metrics(labels, preds)["accuracy"]),
    }


# ---------------------------------------------------------------------------
# Stage 1: Fine-tune on SemEval
# ---------------------------------------------------------------------------

def train_stage1(save_dir=None):
    """Fine-tune BERT on SemEval training data.

    Returns (model, tokenizer, eval_results_dict).
    If a checkpoint already exists at save_dir, loads it instead of re-training.
    """
    if save_dir is None:
        save_dir = MODELS_DIR / "bert_stage1"

    # Check for existing checkpoint
    if (save_dir / "config.json").exists():
        print(f"\n  Stage-1 checkpoint found at {save_dir} — loading instead of re-training.")
        tokenizer = AutoTokenizer.from_pretrained(save_dir)
        model = AutoModelForSequenceClassification.from_pretrained(save_dir)
        return model, tokenizer, None

    print("\n--- BERT Stage 1: Fine-tune on SemEval ---")
    device = get_device()

    # Load data
    df_train_full = load_semeval_train()
    print(f"  SemEval train: {len(df_train_full)} sentences")

    # Create train/val split (90/10)
    texts = df_train_full["text"].tolist()
    labels = df_train_full["label"].tolist()
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.1, stratify=labels, random_state=RANDOM_SEED,
    )
    print(f"  Train split: {len(train_texts)} | Val split: {len(val_texts)}")

    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    train_dataset = prepare_dataset(train_texts, train_labels, tokenizer)
    val_dataset = prepare_dataset(val_texts, val_labels, tokenizer)

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        BERT_MODEL_NAME, num_labels=2,
    )

    # Training arguments
    save_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(save_dir / "checkpoints"),
        num_train_epochs=BERT_EPOCHS_STAGE1,
        per_device_train_batch_size=BERT_BATCH_SIZE,
        per_device_eval_batch_size=BERT_BATCH_SIZE * 2,
        learning_rate=BERT_LR_STAGE1,
        weight_decay=BERT_WEIGHT_DECAY,
        warmup_ratio=BERT_WARMUP_RATIO,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        seed=RANDOM_SEED,
        logging_steps=50,
        report_to="none",
        # MPS is auto-detected in transformers >= 4.41
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=_compute_trainer_metrics,
    )

    # Train
    print(f"  Training for {BERT_EPOCHS_STAGE1} epochs...")
    train_output = trainer.train()

    # Save best model
    trainer.save_model(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))
    print(f"  Model saved to {save_dir}")

    # Eval on validation set — use train() metrics if evaluate() fails
    # (some Transformers versions reset callback state after train())
    try:
        val_results = trainer.evaluate()
    except RuntimeError:
        val_results = {k: v for k, v in train_output.metrics.items()
                       if k.startswith("eval") or k.startswith("train")}
    print(f"  Val F1: {val_results.get('eval_f1', 'N/A'):.4f}")

    return model, tokenizer, val_results


# ---------------------------------------------------------------------------
# Stage 2: Fine-tune on MBS annotated data
# ---------------------------------------------------------------------------

def train_stage2(stage1_dir=None, save_dir=None):
    """Fine-tune stage-1 BERT further on MBS annotated data.

    🔲 NEEDS_LABELS — requires mbs_annotated_train.csv and mbs_annotated_test.csv.

    Returns (model, tokenizer, eval_results_dict) or None if labels missing.
    """
    if stage1_dir is None:
        stage1_dir = MODELS_DIR / "bert_stage1"
    if save_dir is None:
        save_dir = MODELS_DIR / "bert_stage2"

    # Check for labels
    if not MBS_ANNOTATED_TRAIN.exists():
        print("\n  ⚠ MBS annotated training data not found — skipping Stage 2.")
        print(f"    Expected: {MBS_ANNOTATED_TRAIN}")
        print("    Complete Phase 3 annotation first.")
        return None, None, None

    # Check for existing checkpoint
    if (save_dir / "config.json").exists():
        print(f"\n  Stage-2 checkpoint found at {save_dir} — loading instead of re-training.")
        tokenizer = AutoTokenizer.from_pretrained(save_dir)
        model = AutoModelForSequenceClassification.from_pretrained(save_dir)
        return model, tokenizer, None

    print("\n--- BERT Stage 2: Fine-tune on MBS annotated data ---")
    device = get_device()

    # Load stage-1 model
    if not (stage1_dir / "config.json").exists():
        print(f"  ✗ Stage-1 checkpoint not found at {stage1_dir}. Run train_stage1() first.")
        return None, None, None

    tokenizer = AutoTokenizer.from_pretrained(stage1_dir)
    model = AutoModelForSequenceClassification.from_pretrained(stage1_dir)

    # Load MBS annotated data
    df_train = pd.read_csv(MBS_ANNOTATED_TRAIN)
    print(f"  MBS train: {len(df_train)} sentences ({df_train['label'].sum()} suggestions)")

    # Use a small validation split from training data for early stopping
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df_train["sentence_text"].tolist(),
        df_train["label"].tolist(),
        test_size=0.15,
        stratify=df_train["label"].tolist(),
        random_state=RANDOM_SEED,
    )
    print(f"  Train split: {len(train_texts)} | Val split: {len(val_texts)}")

    train_dataset = prepare_dataset(train_texts, train_labels, tokenizer)
    val_dataset = prepare_dataset(val_texts, val_labels, tokenizer)

    # Training arguments — lower LR, fewer epochs to avoid overfitting
    save_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(save_dir / "checkpoints"),
        num_train_epochs=BERT_EPOCHS_STAGE2,
        per_device_train_batch_size=BERT_BATCH_SIZE,
        per_device_eval_batch_size=BERT_BATCH_SIZE * 2,
        learning_rate=BERT_LR_STAGE2,
        weight_decay=BERT_WEIGHT_DECAY,
        warmup_ratio=BERT_WARMUP_RATIO,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        seed=RANDOM_SEED,
        logging_steps=10,
        report_to="none",
        # MPS is auto-detected in transformers >= 4.41
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=_compute_trainer_metrics,
    )

    print(f"  Training for {BERT_EPOCHS_STAGE2} epochs (LR={BERT_LR_STAGE2})...")
    train_output = trainer.train()

    trainer.save_model(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))
    print(f"  Model saved to {save_dir}")

    try:
        val_results = trainer.evaluate()
    except RuntimeError:
        val_results = {k: v for k, v in train_output.metrics.items()
                       if k.startswith("eval") or k.startswith("train")}
    print(f"  Val F1: {val_results.get('eval_f1', 'N/A'):.4f}")

    return model, tokenizer, val_results


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict(model, tokenizer, texts, device=None, batch_size=32):
    """Run batch inference on a list of texts.

    Returns list of predicted labels (0 or 1).
    """
    if device is None:
        device = get_device()
    model.to(device)
    model.eval()

    predictions = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Predicting"):
        batch_texts = texts[i : i + batch_size]
        encodings = tokenizer(
            batch_texts,
            truncation=True,
            padding="max_length",
            max_length=BERT_MAX_LENGTH,
            return_tensors="pt",
        )
        encodings = {k: v.to(device) for k, v in encodings.items()}

        with torch.no_grad():
            outputs = model(**encodings)
        preds = torch.argmax(outputs.logits, dim=-1).cpu().tolist()
        predictions.extend(preds)

    return predictions


def predict_proba(model, tokenizer, texts, device=None, batch_size=32):
    """Run batch inference and return probability of class 1 (suggestion).

    Returns numpy array of probabilities.
    """
    if device is None:
        device = get_device()
    model.to(device)
    model.eval()

    all_probs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Predicting probas"):
        batch_texts = texts[i : i + batch_size]
        encodings = tokenizer(
            batch_texts,
            truncation=True,
            padding="max_length",
            max_length=BERT_MAX_LENGTH,
            return_tensors="pt",
        )
        encodings = {k: v.to(device) for k, v in encodings.items()}

        with torch.no_grad():
            outputs = model(**encodings)
        probs = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()
        all_probs.extend(probs)

    return np.array(all_probs)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def train_and_evaluate(include_stage2=None):
    """Run full BERT pipeline and return results list.

    Args:
        include_stage2: If True, train stage-2 on MBS data.
                        If None, auto-detect (True if mbs_annotated_train.csv exists).
    """
    if include_stage2 is None:
        include_stage2 = MBS_ANNOTATED_TRAIN.exists()

    results = []
    device = get_device()

    # --- Stage 1: SemEval ---
    model_s1, tokenizer_s1, _ = train_stage1()

    # Evaluate stage-1 on SemEval test
    print("\n--- BERT Stage 1 → SemEval test ---")
    df_se_test = load_semeval_test()
    y_true_se = df_se_test["label"].tolist()
    se_texts = df_se_test["text"].tolist()
    y_pred_s1_se = predict(model_s1, tokenizer_s1, se_texts, device)
    y_proba_s1_se = predict_proba(model_s1, tokenizer_s1, se_texts, device)
    metrics = compute_metrics(y_true_se, y_pred_s1_se)
    metrics["pr_auc"] = compute_pr_auc(y_true_se, y_proba_s1_se)
    results.append({"model": "BERT (SemEval)", "dataset": "SemEval", **metrics})
    print_report(y_true_se, y_pred_s1_se, "BERT Stage 1 — SemEval Test")
    print(f"  PR-AUC: {metrics['pr_auc']:.4f}")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(y_true_se, y_pred_s1_se, "BERT Stage 1 — SemEval Test",
                          save_path=FIGURES_DIR / "cm_bert_s1_semeval.png")
    plot_pr_curve(y_true_se, {"BERT (SemEval)": y_proba_s1_se},
                  title="Precision-Recall Curve — BERT Stage 1 — SemEval Test",
                  save_path=FIGURES_DIR / "pr_curve_bert_s1_semeval.png")

    # --- 🔲 NEEDS_LABELS: MBS evaluation ---
    if include_stage2:
        if not MBS_ANNOTATED_TEST.exists():
            print("\n  ⚠ MBS annotated test set not found — skipping MBS evaluation.")
        else:
            df_mbs_test = pd.read_csv(MBS_ANNOTATED_TEST)
            y_true_mbs = df_mbs_test["label"].tolist()
            texts_mbs = df_mbs_test["sentence_text"].tolist()

            # Stage-1 on MBS (cross-domain)
            print("\n--- BERT Stage 1 → MBS test (cross-domain) ---")
            y_pred_s1_mbs = predict(model_s1, tokenizer_s1, texts_mbs, device)
            y_proba_s1_mbs = predict_proba(model_s1, tokenizer_s1, texts_mbs, device)
            metrics = compute_metrics(y_true_mbs, y_pred_s1_mbs)
            metrics["pr_auc"] = compute_pr_auc(y_true_mbs, y_proba_s1_mbs)
            results.append({"model": "BERT (SemEval)", "dataset": "MBS", **metrics})
            print_report(y_true_mbs, y_pred_s1_mbs, "BERT Stage 1 — MBS Test (cross-domain)")
            print(f"  PR-AUC: {metrics['pr_auc']:.4f}")
            plot_confusion_matrix(y_true_mbs, y_pred_s1_mbs,
                                  "BERT Stage 1 — MBS Test",
                                  save_path=FIGURES_DIR / "cm_bert_s1_mbs.png")

            # Stage-2: fine-tune on MBS
            pr_curves_mbs = {"BERT (SemEval)": y_proba_s1_mbs}
            model_s2, tokenizer_s2, _ = train_stage2()
            if model_s2 is not None:
                print("\n--- BERT Stage 2 → MBS test ---")
                y_pred_s2_mbs = predict(model_s2, tokenizer_s2, texts_mbs, device)
                y_proba_s2_mbs = predict_proba(model_s2, tokenizer_s2, texts_mbs, device)
                metrics = compute_metrics(y_true_mbs, y_pred_s2_mbs)
                metrics["pr_auc"] = compute_pr_auc(y_true_mbs, y_proba_s2_mbs)
                results.append({"model": "BERT (SemEval+MBS)", "dataset": "MBS", **metrics})
                print_report(y_true_mbs, y_pred_s2_mbs, "BERT Stage 2 — MBS Test")
                print(f"  PR-AUC: {metrics['pr_auc']:.4f}")
                plot_confusion_matrix(y_true_mbs, y_pred_s2_mbs,
                                      "BERT Stage 2 — MBS Test",
                                      save_path=FIGURES_DIR / "cm_bert_s2_mbs.png")
                pr_curves_mbs["BERT (SemEval+MBS)"] = y_proba_s2_mbs

            plot_pr_curve(y_true_mbs, pr_curves_mbs,
                          title="Precision-Recall Curve — MBS Test",
                          save_path=FIGURES_DIR / "pr_curve_bert_mbs.png")

            # Stage-2 on SemEval test (catastrophic forgetting check)
            if model_s2 is not None:
                print("\n--- BERT Stage 2 → SemEval test (forgetting check) ---")
                y_pred_s2_se = predict(model_s2, tokenizer_s2, se_texts, device)
                y_proba_s2_se = predict_proba(model_s2, tokenizer_s2, se_texts, device)
                metrics_s2_se = compute_metrics(y_true_se, y_pred_s2_se)
                metrics_s2_se["pr_auc"] = compute_pr_auc(y_true_se, y_proba_s2_se)
                results.append({"model": "BERT (SemEval+MBS)", "dataset": "SemEval", **metrics_s2_se})
                print_report(y_true_se, y_pred_s2_se, "BERT Stage 2 — SemEval Test (forgetting check)")
                print(f"  PR-AUC: {metrics_s2_se['pr_auc']:.4f}")
    else:
        print("\n  ⚠ MBS annotated data not found — skipping Stage 2 and MBS evaluation.")
        print("    Complete Phase 3 annotation first.")

    # --- Print comparison ---
    if results:
        print("\n--- BERT Comparison ---")
        table = build_comparison_table(results)
        print(table.to_string(index=False))

    return results
