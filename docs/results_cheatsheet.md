# Results Cheat Sheet

Quick-reference for all metrics, data points, and figure paths needed for the report and presentation. All numbers verified against actual data files on 2026-03-19.

---

## 1. Dataset Statistics

| Metric | Value |
|---|---|
| Raw TripAdvisor reviews | 10,232 |
| After 4-5 star filter + null drop | 8,571 |
| Rating split | 2,414 four-star / 6,157 five-star |
| Total sentences (after segmentation + filtering) | 70,085 |
| Median sentence length | 15 tokens |
| Token range | 4–100 |
| Unique reviews in sentence set | 8,559 |

### Preprocessing pipeline breakdown

| Step | Sentences | Removed | Notes |
|---|---|---|---|
| Sentence segmentation (spaCy) | 72,391 | — | 8.4 sentences/review avg |
| Language filter (ASCII ratio < 0.5) | 72,390 | 1 | Only 1 non-English sentence |
| Deduplication (exact match) | 71,253 | 1,137 | — |
| Length filter (4-100 tokens) | **70,085** | 1,168 | 1,115 too short + 53 too long |

## 2. Annotation Statistics

| Metric | Value |
|---|---|
| Total annotated sentences | 400 |
| Suggestions | 73 (18.2%) |
| Non-suggestions | 327 (81.8%) |
| Training set | 320 sentences (18.1% suggestions) |
| Test set | 80 sentences (18.8% suggestions) |
| Calibration set | 100 sentences (17% suggestions) |
| Annotators | 4 (Chase, Clara, Leah, Xiayi) |

### IAA (Inter-Annotator Agreement)

| Metric | Value |
|---|---|
| **Pre-discussion Fleiss' kappa** | **0.379** (fair agreement) |
| Pairwise Cohen's kappa range | 0.212 (Chase-Leah) to 0.672 (Chase-Clara) |
| Unanimous agreement (4/4) | 60 sentences (53 non-sugg + 7 sugg) |
| 2v2 ties | 11 sentences (resolved by team consensus) |
| Majority vote (3v1) | 29 sentences |
| Primary disagreement source | Over-labelling complaints/descriptions as suggestions |
| Resolution | 89 by majority vote, 11 by team consensus |
| Guideline revision | v1.0 → v1.1 (added §4.8 complaints, clarified §4.6) |

### Per-annotator suggestion rates (calibration round)

| Annotator | Suggestions / 100 |
|---|---|
| Chase | 10% |
| Clara | 14% |
| Leah | 43% |
| Xiayi | 20% |

### Per-annotator suggestion rates (split batches, 75 each)

| Annotator | Suggestions / 75 |
|---|---|
| Chase | 9 (12%) |
| Clara | 9 (12%) |
| Leah | 22 (29%) |
| Xiayi | 16 (21%) |

## 3. Model Results

| Model | Dataset | F1 (sugg) | PR-AUC | Precision | Recall | Accuracy |
|---|---|---|---|---|---|---|
| Regex | SemEval | 0.386 | — | 0.405 | 0.368 | 0.878 |
| Regex | MBS | 0.471 | — | 0.421 | 0.533 | 0.775 |
| TF-IDF+LR | SemEval | 0.596 | 0.601 | 0.486 | 0.770 | 0.891 |
| TF-IDF+LR | MBS | 0.190 | 0.333 | 0.333 | 0.133 | 0.788 |
| BERT (SemEval) | SemEval | 0.716 | 0.805 | 0.624 | 0.839 | 0.930 |
| BERT (SemEval) | MBS | 0.286 | 0.508 | 0.500 | 0.200 | 0.813 |
| **BERT (SemEval+MBS)** | **MBS** | **0.581** | **0.531** | **0.563** | **0.600** | **0.838** |
| BERT (SemEval+MBS) | SemEval | 0.656 | 0.796 | 0.513 | 0.908 | 0.900 |

### Key narrative points
- TF-IDF+LR collapses cross-domain: F1 0.596 → 0.190 (vocabulary mismatch)
- BERT Stage 1 also degrades: F1 0.716 → 0.286 (high precision, very low recall on MBS)
- **Stage 2 domain adaptation nearly doubles MBS F1: 0.286 → 0.581**
- Regex outperforms learned models on MBS without adaptation (F1 = 0.471) due to pattern-based recall
- **Catastrophic forgetting (Stage 2 on SemEval):** F1 drops 0.716 → 0.656. Precision drops sharply (0.624 → 0.513) but recall increases (0.839 → 0.908). The model becomes more trigger-happy — it over-predicts suggestions on software text after hotel-domain fine-tuning. PR-AUC drops slightly (0.805 → 0.796). This is moderate forgetting, not catastrophic.

### Full-dataset predictions (BERT Stage 2 on all 70,085 sentences)
- Predicted suggestions: **4,179 (6.0%)**
- Saved to: `data/processed/mbs_predictions.csv`

## 4. Error Analysis (MBS test set, 80 sentences, 15 suggestions)

| Model | FP | FN | Key FP pattern | Key FN pattern |
|---|---|---|---|---|
| Regex | 11 | 7 | Imperative endorsements ("I recommend this hotel") | Implicit suggestions in "other" category |
| TF-IDF+LR | 4 | 13 | Low recall — misses almost everything | Nearly all categories missed |
| BERT (SemEval) | 3 | 12 | Imperative endorsements | Low recall, misses implicit/other |
| **BERT (SemEval+MBS)** | **7** | **6** | Imperative endorsements, 1 positive modal | Implicit suggestions, "other" category |

## 5. Topic Modeling (BERTopic on 4,179 predicted suggestions)

| Metric | Value |
|---|---|
| Topics found | 56 |
| Outlier sentences | 1,101 (26%) |
| Embedding model | all-MiniLM-L6-v2 |
| Min topic size | 5 |

### Aspect distribution

| Aspect | Count | Percentage |
|---|---|---|
| Other | 1,299 | 31.1% |
| Room | 990 | 23.7% |
| Facilities | 638 | 15.3% |
| Food & beverage | 490 | 11.7% |
| Value | 309 | 7.4% |
| Service | 234 | 5.6% |
| Location | 219 | 5.2% |

### Notable specific topics
Minibar sensor warnings, passport for casino, elevator waits, dress code after 7pm, booking in advance, sunscreen at rooftop pool, waterproof phone cases, USB charging ports, coffee machines in rooms, smoking areas, valet parking, light/laser show timing, Spago restaurant tips.

## 6. Domain Gap Statistics

| Metric | MBS | SemEval |
|---|---|---|
| Sentences | 70,085 | 8,500 |
| Median sentence length | 13 words | 15 words |
| Mean sentence length | 15.4 words | 17.8 words |
| Vocabulary size | 36,722 | 18,347 |
| Vocabulary overlap (Jaccard) | 0.104 | — |
| Suggestion base rate | 18.2% (annotated) | 24.5% |
| Domain | Hotel reviews (TripAdvisor) | Software forums (Windows Phone) |

## 7. Figure Manifest (25 files in `outputs/figures/`)

### EDA (6 figures)
| File | Description |
|---|---|
| `rating_distribution.png` | Bar chart: 4-star vs 5-star review counts |
| `review_length_distribution.png` | Histogram: words per review |
| `sentence_length_distribution.png` | Histogram: tokens per sentence |
| `temporal_distribution.png` | Time series: reviews per month |
| `signal_comparison.png` | Bar chart: modal verb/suggestion signal frequencies, MBS vs SemEval |
| `domain_sentence_lengths.png` | Overlaid histograms: sentence lengths, MBS vs SemEval |

### Model Evaluation (11 figures)
| File | Description |
|---|---|
| `cm_regex_semeval.png` | Confusion matrix: Regex on SemEval test |
| `cm_regex_mbs.png` | Confusion matrix: Regex on MBS test |
| `cm_tfidf_lr_semeval.png` | Confusion matrix: TF-IDF+LR on SemEval test |
| `cm_tfidf_lr_mbs.png` | Confusion matrix: TF-IDF+LR on MBS test |
| `cm_bert_s1_semeval.png` | Confusion matrix: BERT Stage 1 on SemEval test |
| `cm_bert_s1_mbs.png` | Confusion matrix: BERT Stage 1 on MBS test |
| `cm_bert_s2_mbs.png` | Confusion matrix: BERT Stage 2 on MBS test |
| `pr_curve_baselines_semeval.png` | PR curves: Regex + TF-IDF+LR on SemEval |
| `pr_curve_bert_s1_semeval.png` | PR curve: BERT Stage 1 on SemEval |
| `pr_curve_bert_mbs.png` | PR curves: BERT Stage 1 + Stage 2 on MBS |
| `model_comparison.png` | Bar chart: all 7 model/dataset combinations |

### Error Analysis (5 figures)
| File | Description |
|---|---|
| `error_breakdown_regex.png` | Stacked bar: FP/FN by suggestion category (Regex) |
| `error_breakdown_tf-idf__lr.png` | Stacked bar: FP/FN by suggestion category (TF-IDF+LR) |
| `error_breakdown_bert_semeval.png` | Stacked bar: FP/FN by suggestion category (BERT Stage 1) |
| `error_breakdown_bert_semevalmbs.png` | Stacked bar: FP/FN by suggestion category (BERT Stage 2) |
| `error_comparison_models.png` | Grouped bar: total FP vs FN across all models |

### Topic Modeling (3 figures)
| File | Description |
|---|---|
| `topic_barchart.png` | Top topics by sentence count |
| `topic_wordclouds.png` | Word clouds for top topics |
| `aspect_distribution.png` | Bar chart: suggestions by hotel aspect |

## 8. Data File Paths

| File | Rows | Description |
|---|---|---|
| `data/processed/mbs_filtered_reviews.csv` | 8,571 | Filtered reviews (4-5 stars) |
| `data/processed/mbs_sentences.csv` | 70,085 | Segmented sentences |
| `data/processed/mbs_calibration_100.csv` | 100 | Calibration with 4 annotator labels + gold |
| `data/processed/mbs_annotated_full.csv` | 400 | All annotated sentences |
| `data/processed/mbs_annotated_train.csv` | 320 | Training split |
| `data/processed/mbs_annotated_test.csv` | 80 | Test split |
| `data/processed/mbs_predictions.csv` | 70,085 | BERT Stage 2 predictions on all sentences |
| `data/processed/topic_summary.csv` | 56 | BERTopic topic summaries |
| `data/processed/aspect_summary.csv` | 7 | Aspect-based grouping summary |
