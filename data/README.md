# Data Guide (NLP_Team6)

This project uses:
1) **Marina Bay Sands TripAdvisor reviews** (primary, unlabeled) for inference and insight generation.
2) **SemEval-2019 Task 9 (Suggestion Mining)** data (reference labeled dataset) for benchmarking and to inform annotation/modeling.

---

## 1) Primary Dataset (Marina Bay Sands TripAdvisor Reviews)

**Source (Kaggle):**
- Marina Bay Sands Hotel Reviews on TripAdvisor  
  https://www.kaggle.com/datasets/lucashkliu/marina-bay-sands-hotel-reviews-on-tripadvisor

**Where to place the raw file:**
- Download the dataset from Kaggle and place it under:

```
data/raw/
```

**Expected filename (recommended):**
- Keep the original filename if possible (e.g. `tripadvisor_mbs_review_from201501_v2.xlsx`)

**Important: raw data is NOT tracked by Git**
- `data/raw/` should be ignored in `.gitignore` to avoid pushing large files.

---

## 2) Reference Labeled Dataset (SemEval-2019 Task 9)

We keep the SemEval files under:

```
external/semeval2019task9/
```

This folder contains the official training/dev/test CSV files and baseline/evaluation scripts.

---

## 3) Processed / Derived Data (Tracked for Reproducibility)

Outputs produced by our preprocessing pipeline should be saved to:

```
data/processed/
```

### Key processed artifacts (planned)
- `reviews_high_rating.csv`  
  Review-level data filtered to 4–5 stars (optional intermediate).
- `sentences_high_rating.csv`  
  **Main modeling table** (sentence-level). Suggested schema:

  - `review_id` (or row id)
  - `rating`
  - `date` (if available)
  - `sentence_id`
  - `sentence_text`

- `labeled_sentences.csv`  
  Manually annotated subset for training/evaluation:
  - `sentence_text`
  - `label` (Suggestion / Non-Suggestion)
  - optional: `annotator`, `notes`

We recommend tracking `data/processed/sentences_high_rating.csv` (and the labeled subset) in Git to ensure the project is reproducible for all group members.

---

## 4) Notes / Common Pitfalls

- **Stopwords**: Do NOT remove stopwords globally; modal verbs (e.g., *should*, *would*, *could*, *if*) are important cues for suggestions.
- **Sentence segmentation**: The unit of analysis is a sentence. Review text must be segmented before annotation/modeling.
- **Language filtering**: If needed, keep only English sentences for consistency with SemEval and model assumptions.
