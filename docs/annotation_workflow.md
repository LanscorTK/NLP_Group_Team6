# Annotation Workflow — Step-by-Step Team Guide

**Project:** NLP_Team6 — Suggestion Mining in High-Rated Hotel Reviews
**Date:** 2026-03-14
**Team:** Chase, Clara, Leah, Xiayi

---

## Overview

This document describes the full annotation process from start to finish. Follow each step in order. The process has two main rounds:

1. **Calibration round** (100 shared sentences) — all 4 members label the same sentences independently to measure agreement
2. **Split annotation** (200-400 additional sentences) — each member labels a separate batch

**Total target:** 300-500 labelled sentences

---

## Step 1: Read the Annotation Guidelines

**Who:** All 4 members
**Time:** 15-20 minutes

1. Open `docs/annotation_guidelines.md`
2. Read the entire document carefully, especially the edge cases in Section 4
3. Pay special attention to:
   - The core decision rule (Section 2)
   - Complaint vs. suggestion distinction (Section 4.4)
   - Positive modal verbs that are NOT suggestions (Section 4.5)
4. If anything is unclear, discuss with the team **before** the calibration round begins

---

## Step 2: Generate the Calibration Sample

**Who:** Chase (or whoever runs the code)
**Time:** 5 minutes

1. Open `notebooks/00_full_pipeline.ipynb` (Phase 3: Annotation section)
2. Run the cells under "Step 3.2: Sample for IAA calibration" (or call `src/annotation.py` directly)
3. This produces `data/processed/mbs_calibration_100.csv` with 100 sentences
4. The sample is enriched: ~50% of sentences contain suggestion signals (modal verbs, imperatives), ~50% are random. This is documented in the guidelines.
5. Share this CSV with all team members (e.g., upload to Google Drive, or use Google Sheets)

### CSV format:

| sentence_id | sentence_text | label_chase | label_clara | label_leah | label_xiayi |
|---|---|---|---|---|---|
| 12345 | "The room was beautiful..." | | | | |
| 67890 | "They should add more..." | | | | |

---

## Step 3: Calibration Labelling (Independent)

**Who:** All 4 members, working independently
**Time:** 30-60 minutes per person

### Rules:
- **DO NOT discuss labels with other members** — independence is critical for valid IAA
- **DO NOT look at other members' labels**
- Work through all 100 sentences in one sitting if possible
- For each sentence, enter `1` (suggestion) or `0` (non-suggestion) in your column
- When in doubt, refer to the guidelines; if still unsure, label `0`

### Process:
1. Open the shared CSV/spreadsheet
2. Read each sentence carefully
3. Apply the decision rule: "Does this sentence propose a concrete action?"
4. Enter your label (0 or 1) in your column
5. Move to the next sentence — do not go back and change labels

---

## Step 4: Compute Inter-Annotator Agreement

**Who:** Chase (or whoever runs the code)
**Time:** 10 minutes

1. Download the completed calibration CSV with all 4 members' labels
2. Save it as `data/processed/mbs_calibration_100.csv` (overwriting the empty version)
3. Run the IAA computation cells in `notebooks/00_full_pipeline.ipynb`
4. The notebook will compute:
   - **Fleiss' kappa** (overall agreement across all 4 raters)
   - **Pairwise Cohen's kappa** (agreement between each pair)

### Interpreting kappa:

| Kappa | Interpretation | Action |
|---|---|---|
| 0.81-1.00 | Almost perfect | Proceed to split annotation |
| 0.61-0.80 | Substantial | Proceed to split annotation |
| 0.41-0.60 | Moderate | Discuss disagreements, consider guideline revision |
| 0.21-0.40 | Fair | Revise guidelines, re-label disputed sentences |
| < 0.21 | Poor | Major guideline revision needed |

**Target: kappa >= 0.6.** If lower, proceed to Step 5 before continuing.

---

## Step 5: Disagreement Discussion (if needed)

**Who:** All 4 members
**Time:** 30-60 minutes

1. The notebook will display the sentences where annotators disagreed most (e.g., 2-vs-2 ties, 3-vs-1 splits)
2. For each disagreed sentence, discuss:
   - What did each person think? Why?
   - Which edge case category does this fall into?
   - Does the guideline need to be clearer?
3. Update `docs/annotation_guidelines.md` with any new rules or clarifications
4. Re-label the disputed sentences if needed
5. Re-compute kappa to verify improvement

### Common disagreement patterns:
- **Complaint vs. implicit suggestion:** "The walls were thin" — agree on the rule that complaints without a proposed action = 0
- **Positive modals:** "I would definitely return" — agree this is praise (0), not a suggestion
- **Advice to guests:** "Make sure to book early" — agree whether guest-directed advice counts (our guidelines say yes = 1)

---

## Step 6: Generate Additional Annotation Batches

**Who:** Chase (or whoever runs the code)
**Time:** 5 minutes

1. Run the cells in `notebooks/00_full_pipeline.ipynb` under "Step 3.3: Sample additional sentences"
2. This produces 4 CSV files, one per member:
   - `data/processed/annotation_batch_chase.csv`
   - `data/processed/annotation_batch_clara.csv`
   - `data/processed/annotation_batch_leah.csv`
   - `data/processed/annotation_batch_xiayi.csv`
3. Each batch contains 50-100 unique sentences (no overlap between batches or with the calibration set)
4. Distribute each file to the corresponding team member

---

## Step 7: Split Annotation (Independent)

**Who:** Each member labels only their own batch
**Time:** 30-60 minutes per person

1. Open your assigned batch CSV
2. For each sentence, enter `1` (suggestion) or `0` (non-suggestion) in the `label` column
3. Apply the same guidelines and decision rules as the calibration round
4. When finished, save and return the completed CSV

---

## Step 8: Merge and Create Train/Test Split

**Who:** Chase (or whoever runs the code)
**Time:** 10 minutes

1. Collect all completed batch CSVs from team members
2. Place them in `data/processed/`
3. Run the merge cells in `notebooks/00_full_pipeline.ipynb`
4. The notebook will:
   - Merge calibration labels (majority vote) with the 4 batch files
   - Report label distribution (% suggestions vs. non-suggestions)
   - Create a stratified 80/20 train/test split
   - Save:
     - `data/processed/mbs_annotated_train.csv`
     - `data/processed/mbs_annotated_test.csv`

### Expected output:
- Total annotated: 300-500 sentences
- Suggestion rate: ~20-35% (due to enrichment)
- Train set: 240-400 sentences
- Test set: 60-100 sentences

---

## Step 9: Record for the Report

Document the following in the Methods section of the final report:

1. **Annotation guidelines:** Summarise the definition and key edge cases
2. **Sample enrichment:** 50% signal-enriched sampling strategy and rationale
3. **Calibration process:** Fleiss' kappa score, number of disputed sentences, how disagreements were resolved
4. **Final dataset:** Total annotated sentences, label distribution, train/test split sizes
5. **Annotator:** Who labelled what (calibration = all 4; splits = individual assignment)

---

## Timeline

| Step | Who | When | Duration |
|---|---|---|---|
| 1. Read guidelines | All | Day 1 | 20 min |
| 2. Generate calibration sample | Chase | Day 1 | 5 min |
| 3. Calibration labelling | All (independent) | Days 1-2 | 30-60 min each |
| 4. Compute IAA | Chase | Day 2 | 10 min |
| 5. Disagreement discussion | All | Day 2 | 30-60 min |
| 6. Generate split batches | Chase | Day 2 | 5 min |
| 7. Split annotation | Each member | Days 2-3 | 30-60 min each |
| 8. Merge and split | Chase | Day 3 | 10 min |
| 9. Document for report | Leah/Clara | Day 3 | 30 min |
