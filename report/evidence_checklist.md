# Evidence Checklist — Artifacts Required for the Final Report

Each item below must be sourced from a real project artifact. Items marked [HAVE] are available now. Items marked [MISSING] need to be produced or retrieved before writing.

---

## A. Dataset Statistics

| # | Item | Source | Status |
|---|------|--------|--------|
| A1 | Raw review count (10,232) | data_loading.py output | [HAVE] |
| A2 | Filtered review count after 4-5 star filter (8,571) | mbs_filtered_reviews.csv | [HAVE] |
| A3 | Rating split: 2,414 four-star + 6,157 five-star | mbs_filtered_reviews.csv | [HAVE] |
| A4 | Total sentence count after segmentation (70,085) | mbs_sentences.csv | [HAVE] |
| A5 | Preprocessing removal counts (language filter, dedup, length) | preprocessing.py output logs | [HAVE] |
| A6 | Median sentence length (15 tokens), mean sentences per review (8.4) | EDA notebook | [HAVE] |
| A7 | SemEval training set size and label distribution | V1.4_Training.csv | [HAVE] |
| A8 | SemEval test set size and label distribution | SubtaskA_EvaluationData_labeled.csv | [HAVE] |
| A9 | Domain gap statistics: vocabulary overlap, modal verb frequency comparison | EDA notebook / eda.py | [HAVE] |

## B. Figures (all excluded from word count)

| # | Figure | File | Status |
|---|--------|------|--------|
| B1 | Rating distribution bar chart | outputs/figures/rating_distribution.png | [HAVE] |
| B2 | Review length distribution | outputs/figures/review_length_distribution.png | [HAVE] |
| B3 | Sentence length distribution | outputs/figures/sentence_length_distribution.png | [HAVE] |
| B4 | Temporal distribution of reviews | outputs/figures/temporal_distribution.png | [HAVE] |
| B5 | Suggestion signal comparison (MBS vs SemEval) | outputs/figures/signal_comparison.png | [HAVE] |
| B6 | Domain sentence length comparison | outputs/figures/domain_sentence_lengths.png | [HAVE] |
| B7 | Confusion matrix: regex on SemEval | outputs/figures/cm_regex_semeval.png | [HAVE] |
| B8 | Confusion matrix: TF-IDF+LR on SemEval | outputs/figures/cm_tfidf_lr_semeval.png | [HAVE] |
| B9 | PR curve: baselines on SemEval | outputs/figures/pr_curve_baselines_semeval.png | [HAVE] |
| B10 | Topic bar chart (BERTopic) | outputs/figures/topic_barchart.png | [HAVE] |
| B11 | Topic word clouds | outputs/figures/topic_wordclouds.png | [HAVE] |
| B12 | Aspect distribution | outputs/figures/aspect_distribution.png | [HAVE] |
| B13 | Confusion matrices: all models on MBS test | outputs/figures/cm_*_mbs.png | [HAVE] |
| B14 | PR curves on MBS test set | outputs/figures/pr_curve_bert_mbs.png | [HAVE] |
| B15 | BERT training curves (loss per epoch, stage 1 + stage 2) | — | [MISSING — not generated] |
| B16 | Pipeline/architecture diagram | figures/pipeline_diagram.png | [HAVE] |
| B17 | Model comparison chart | outputs/figures/model_comparison.png | [HAVE] |
| B18 | Error breakdown by category (all models) | outputs/figures/error_breakdown_*.png | [HAVE] |
| B19 | Error comparison across models | outputs/figures/error_comparison_models.png | [HAVE] |

## C. Annotation & IAA

| # | Item | Source | Status |
|---|------|--------|--------|
| C1 | Annotation guidelines document | docs/annotation_guidelines.md | [HAVE] |
| C2 | Calibration sample details (100 sentences, enrichment strategy) | mbs_calibration_100.csv + annotation.py | [HAVE] |
| C3 | Fleiss' kappa from calibration round (0.379) | mbs_calibration_100.csv | [HAVE — written in report] |
| C4 | Pairwise Cohen's kappa (0.212–0.672, mean 0.426) | mbs_calibration_100.csv | [HAVE — written in report] |
| C5 | Number of ties (11 at 2-vs-2) and resolution process | mbs_calibration_100.csv | [HAVE — written in report] |
| C6 | Final annotated dataset size (400) and label distribution (73/327) | mbs_annotated_full.csv | [HAVE — written in report] |
| C7 | Train/test split sizes (320/80, stratified) | mbs_annotated_train/test.csv | [HAVE — written in report] |
| C8 | Enrichment strategy documentation | annotation_guidelines.md §7 | [HAVE] |

## D. Model Configuration & Results

| # | Item | Source | Status |
|---|------|--------|--------|
| D1 | Regex pattern list (exact patterns used) | src/baselines.py | [HAVE] |
| D2 | Regex F1 on SemEval test (0.386) | cm_regex_semeval.png | [HAVE — written in report] |
| D3 | TF-IDF+LR config (ngrams, max_features, min_df, class_weight) | src/baselines.py | [HAVE] |
| D4 | TF-IDF+LR F1 on SemEval test (0.595) | cm_tfidf_lr_semeval.png | [HAVE — written in report] |
| D5 | TF-IDF+LR PR-AUC on SemEval | baselines.py output | [HAVE] |
| D6 | BERT Stage-1 config (lr, batch, epochs, device, seed) | src/bert_model.py | [HAVE] |
| D7 | BERT Stage-1 val F1 macro (0.886), suggestion F1 (0.828) | bert_model.py output | [HAVE] |
| D8 | BERT Stage-1 checkpoint saved | outputs/models/bert_stage1/ | [HAVE] |
| D9 | Regex F1 on MBS test set (0.471) | cm_regex_mbs.png | [HAVE — written in report] |
| D10 | TF-IDF+LR F1 on MBS test set (0.190) | cm_tfidf_lr_mbs.png | [HAVE — written in report] |
| D11 | BERT Stage-1 F1 on MBS test set (0.286) | cm_bert_s1_mbs.png | [HAVE — written in report] |
| D12 | BERT Stage-2 config (lr=1e-5, 3 epochs) | src/bert_model.py | [HAVE] |
| D13 | BERT Stage-2 F1 on MBS test set (0.581) | cm_bert_s2_mbs.png | [HAVE — written in report] |
| D14 | Cross-domain delta table | Computed from CMs | [HAVE — written in report] |
| D15 | BERTopic config (min_topic_size, embedding model, nr_topics) | src/topic_modeling.py | [HAVE] |
| D16 | BERTopic results: 56 topics, 4,179 suggestions | topic_summary.csv | [HAVE] |
| D17 | Aspect-based grouping distribution | aspect_summary.csv | [HAVE] |

## E. Error Analysis

| # | Item | Source | Status |
|---|------|--------|--------|
| E1 | False positive examples from MBS test set | error_analysis.py output | [HAVE — written in report] |
| E2 | False negative examples from MBS test set | error_analysis.py output | [HAVE — written in report] |
| E3 | Error type breakdown (imperative/other/positive_modal) | error_breakdown_bert_semevalmbs.png | [HAVE — written in report] |
| E4 | Manual validation results (100+100 predictions) | — | [NOT DONE — noted in Limitations] |
| E5 | Cross-domain error comparison (FP vs FN by model) | error_comparison_models.png | [HAVE — written in report] |

## F. References & Citations

| # | Item | Status |
|---|------|--------|
| F1 | Negi et al. (2019) — SemEval-2019 Task 9 overview | [VERIFIED] |
| F2 | Devlin et al. (2019) — BERT paper | [VERIFIED] |
| F3 | Grootendorst (2022) — BERTopic paper | [VERIFIED] |
| F4 | spaCy citation | [VERIFIED — Honnibal et al. (2020)] |
| F5 | scikit-learn citation | [VERIFIED — Pedregosa et al. (2011)] |
| F6 | HuggingFace Transformers citation | [VERIFIED — Wolf et al. (2020)] |
| F7 | TripAdvisor data source acknowledgement | [HAVE — mentioned in Data section] |
| F8 | AI tool acknowledgement | [HAVE — written in Appendix] |
| F9 | Additional suggestion mining papers | [VERIFIED — 4 papers] |
| F10 | sentence-transformers citation | [VERIFIED — Reimers & Gurevych (2019)] |

---

## Summary

| Category | Have | Missing | Total |
|----------|------|---------|-------|
| Dataset Statistics (A) | 9 | 0 | 9 |
| Figures (B) | 18 | 1 | 19 |
| Annotation & IAA (C) | 8 | 0 | 8 |
| Model Results (D) | 17 | 0 | 17 |
| Error Analysis (E) | 4 | 1 | 5 |
| References (F) | 10 | 0 | 10 |
| **Total** | **66** | **2** | **68** |

**Remaining gaps:**
- B15: BERT training curves (loss/F1 per epoch) — not generated by pipeline. Would require modifying `bert_model.py` to log per-epoch metrics. Low priority.
- E4: Manual validation — not performed. Documented as a limitation in Section 6.3.
