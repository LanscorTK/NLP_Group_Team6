# Project Status & Requirements Analysis

**Project:** Suggestion Mining in High-Rated Hotel Reviews
**Module:** MSIN0221 Natural Language Processing (UCL School of Management)
**Team:** Team 6 — Chase Sun, Clara Yang, Leah Luo, Xiayi Zhu
**Date:** 2026-03-14 (last updated: 2026-03-20, report draft complete)
**Deadline:** 2026-03-31 (Final Report + Presentation), 10:00 AM UK time

---

## 1. What the Brief Requires

The assignment is an end-to-end NLP group project (60% of module grade) with three deliverables:

| Deliverable | Weight | Due | Status |
|---|---|---|---|
| Project Proposal (max 3 pages) | 25% | 02/02/2026 | DONE (submitted) |
| Final Report (max 4,000 words) | 50% | 31/03/2026 | DRAFT COMPLETE (3,560/4,000 words, proofreader approved) |
| Recorded Presentation (5 min) | 25% | 31/03/2026 | NOT STARTED |

### Key grading criteria for the Final Report (50%):
- **Structure & Format (5%):** Concise summary, coherent structure, good visualisations, proper referencing
- **Methodology & Analysis (30%):** Clear dataset description, reproducible methodology, in-depth analysis, robust validation, innovative approach
- **Quality of Narrative (10%):** Critical thinking, clear research question, limitations acknowledged, conclusions consistent with evidence
- **Code / Equations / Referencing (5%):** Well-structured annotated code, proper citations (including AI tools), mathematical notation

### Key grading criteria for the Presentation (25%):
- Clear and engaging within 5 min (+/- 30 sec)
- Good visualisations, clear motivation, convincing NLP demonstration
- Logical conclusions with actionable future-work recommendations

### Important constraints:
- Report must be structured like a scientific paper (Introduction, Methods, Results, Error Analysis & Discussion) — recommended as Jupyter Notebook PDF or LaTeX (ACL/ICLR template)
- **Must be reproducible** — code and data must be provided alongside the report
- AI use category is **Assistive** — AI tools may be used to support development but must be acknowledged
- First page must include a **member contribution breakdown**
- The brief emphasises **motivation, approach, and process understanding** over absolute model performance

---

## 2. What the Project Is Doing

**Research question:** *To what extent can NLP models detect actionable improvement suggestions within high-star hotel reviews?*

The project applies **suggestion mining** (a sentence-level binary classification task) to luxury hotel reviews, specifically targeting "hidden suggestions" buried in 4-5 star reviews that traditional sentiment analysis would miss.

### Central analytical theme: Domain Transfer

The SemEval-2019 training data comes from **software developer forums** (Windows Phone feature requests), which is structurally very different from hotel review language. Rather than treating this as a limitation, the project frames the **SemEval-to-hotel domain gap** as a central analytical finding: how well do suggestion-detection models transfer across domains, and what adaptations are needed?

### Planned 7-phase pipeline:

1. **Data Collection & Preprocessing** — Load Marina Bay Sands TripAdvisor reviews (10k+ reviews) from `data/raw/`, filter to 4-5 stars, segment into sentences with spaCy, retain stopwords (should/would/if)
2. **Exploratory Data Analysis** — Sentence length distributions, modal verb frequencies, domain gap analysis (SemEval vs. MBS data), temporal patterns
3. **Annotation** — Binary labelling (Suggestion / Non-Suggestion) at sentence level, with inter-annotator agreement (IAA) measurement and hotel-domain-adapted guidelines
4. **Model Development** — Three-tier comparison:
   - **Tier 1 — Regex baseline:** Rule-based pattern matching on modal verbs and imperative forms
   - **Tier 2 — TF-IDF + Logistic Regression:** Simple ML baseline trained on SemEval data (demonstrates value of learned features over rules)
   - **Tier 3 — BERT classifier:** Two-stage fine-tuning (SemEval → MBS annotated data) to demonstrate transfer learning and domain adaptation
5. **Insight Generation** — BERTopic clustering on extracted suggestions (replaces LDA to handle short-text sparsity), optionally supplemented by aspect-based grouping (room, food, service, facilities, location)
6. **Evaluation & Validation** — Precision/Recall/F1 on test set, confusion matrix, manual validation of 100 random predictions each for suggestions and non-suggestions
7. **Error Analysis** — Breakdown by suggestion type: explicit ("They should add...") vs. implicit ("It would be nice if...") vs. comparative ("Other hotels have..."); cross-domain performance comparison (SemEval test vs. MBS data)

---

## 3. What Has Been Done

### Architecture (established 2026-03-14):
- **Hybrid code architecture:** Core logic in `src/` modules, thin notebook wrappers in `notebooks/`, plus `run_pipeline.py` CLI runner for one-command reproducibility
- `src/config.py` — centralised paths, seeds, preprocessing parameters
- `run_pipeline.py` — end-to-end runner (`python run_pipeline.py` or `--phase N`)

### Phase 1 — Data Loading & Preprocessing (DONE, 2026-03-14):
- `src/data_loading.py` + `notebooks/01_data_loading.ipynb`
  - 10,232 raw reviews → **8,571 filtered reviews** (4-5 stars, null-dropped)
  - Rating split: 2,414 four-star + 6,157 five-star
  - Output: `data/processed/mbs_filtered_reviews.csv`
- `src/preprocessing.py` + `notebooks/02_preprocessing.ipynb`
  - 8,571 reviews → **70,085 sentences** (after language filter, dedup, length threshold)
  - Median sentence length: 15 tokens; 8.4 sentences per review avg
  - Output: `data/processed/mbs_sentences.csv`
- Both notebooks and `run_pipeline.py --phase 1` verified to produce identical outputs

### Phase 2 — Exploratory Data Analysis (DONE, 2026-03-14):
- `src/eda.py` + `notebooks/03_eda.ipynb`
  - MBS dataset statistics: rating distribution, review length, sentence length, temporal patterns
  - Domain gap analysis: SemEval (software forums) vs. MBS (hotel reviews) — sentence length distributions, modal verb frequencies, vocabulary overlap
  - Suggestion signal comparison: modal verbs and imperative patterns across datasets
  - 6 publication-quality figures saved to `outputs/figures/`:
    - `rating_distribution.png`, `review_length_distribution.png`, `sentence_length_distribution.png`
    - `temporal_distribution.png`, `signal_comparison.png`, `domain_sentence_lengths.png`

### Phase 3 — Annotation (DONE, 2026-03-19):
- `src/annotation.py` + `notebooks/04_annotation.ipynb` — code fully implemented
- `docs/annotation_guidelines.md` v1.1 (revised after calibration round)
- `docs/annotation_workflow.md` — step-by-step team guide
- **Calibration round COMPLETE (2026-03-15):**
  - All 4 annotators labelled 100 shared sentences independently
  - Fleiss' kappa = 0.379 (fair) — primary issue: over-labelling of complaints as suggestions
  - Guidelines revised: added §4.8 (complaints/narrative), clarified §4.6 (informational vs. advisory)
  - 89 sentences resolved by majority vote, 11 ties resolved by team consensus
  - Gold labels: 83 non-suggestions, 17 suggestions (17% suggestion rate)
- **Split annotation COMPLETE (2026-03-19):**
  - 300 additional sentences labelled (75 per annotator): Chase 9, Clara 9, Leah 22, Xiayi 16 suggestions
  - `merge_annotations()` added to `src/annotation.py` — uses `gold_label` from calibration CSV directly
  - **Total: 400 annotated sentences (73 suggestions, 18.2%)**
  - Stratified 80/20 train/test split:
    - `data/processed/mbs_annotated_train.csv` — 320 sentences (18.1% suggestions)
    - `data/processed/mbs_annotated_test.csv` — 80 sentences (18.8% suggestions)
  - Full merged file: `data/processed/mbs_annotated_full.csv`

### Infrastructure:
- Repository structure: `docs/`, `data/raw/`, `data/processed/`, `notebooks/`, `src/`, `external/`, `outputs/`
- Raw data acquired: `tripadvisor_mbs_review_from201501_v2.xlsx` (3.7 MB)
- SemEval-2019 reference data in `external/semeval2019task9/` (training, trial, test sets + evaluation script)
- README with setup instructions, pipeline overview, architecture description
- `data/README.md` with data placement instructions
- `requirements.txt` with all dependencies (including BERTopic, statsmodels for IAA)
- `.gitignore` updated for project structure (tracks `data/processed/`, `outputs/figures/`; ignores `data/raw/`, `.claude/`, model checkpoints)
- `notebooks/00_full_pipeline.ipynb` — integrated notebook running all 6 phases end-to-end (complements individual notebooks 01–08)
- `notebooks/08_error_analysis.ipynb` — standalone error analysis notebook (Phase 6 wrapper for `src/error_analysis.py`)
- `docs/results_cheatsheet.md` — consolidated reference with all metrics, figure paths, and data points for report writing

### Phase 4 — Model Development (DONE, 2026-03-19):
- `src/baselines.py`: Regex classifier + TF-IDF+LR evaluated on both SemEval and MBS test sets
- `src/bert_model.py`: BERT stage-1 (SemEval) and stage-2 (SemEval+MBS) both trained and evaluated
- `src/evaluation.py`: Shared metrics (PR-AUC, PR curves, confusion matrices, comparison tables)
- **Full results:**

  | Model | Dataset | F1 (sugg) | PR-AUC | Precision | Recall | Accuracy |
  |---|---|---|---|---|---|---|
  | Regex | SemEval | 0.386 | — | 0.405 | 0.368 | 0.878 |
  | Regex | MBS | 0.471 | — | 0.421 | 0.533 | 0.775 |
  | TF-IDF+LR | SemEval | 0.596 | 0.601 | 0.486 | 0.770 | 0.891 |
  | TF-IDF+LR | MBS | 0.190 | 0.333 | 0.333 | 0.133 | 0.788 |
  | BERT (SemEval) | SemEval | 0.716 | 0.805 | 0.624 | 0.839 | 0.930 |
  | BERT (SemEval) | MBS | 0.286 | 0.508 | 0.500 | 0.200 | 0.813 |
  | **BERT (SemEval+MBS)** | **MBS** | **0.581** | **0.531** | **0.563** | **0.600** | **0.838** |

- **Key findings:**
  - TF-IDF+LR collapses cross-domain (F1: 0.596 → 0.190) — vocabulary mismatch
  - BERT stage-1 also degrades cross-domain (F1: 0.716 → 0.286) — high precision but very low recall on MBS
  - Stage-2 domain adaptation nearly doubles MBS F1 (0.286 → 0.581) — validates two-stage fine-tuning
  - Regex outperforms learned models on MBS without adaptation (F1 = 0.471) due to pattern-based recall
- **Checkpoints:** `outputs/models/bert_stage1/` and `outputs/models/bert_stage2/` (~440MB each, git-ignored)
- **Full-dataset re-prediction:** BERT stage-2 on all 70,085 MBS sentences → **4,179 suggestions (6.0%)**, saved to `data/processed/mbs_predictions.csv`
- **Figures:** confusion matrices (7), PR curves (3), model comparison chart

### Phase 5 — Topic Modeling (DONE, re-run with stage-2 predictions 2026-03-19):
- `src/topic_modeling.py` + `notebooks/07_topic_modeling.ipynb`
- Stage-2 predictions: **4,179 suggestions** fed into BERTopic
- BERTopic clustering: **56 topics** found (1,101 outlier sentences, 26%)
  - Much more granular than earlier stage-1 run (17 topics)
  - Specific actionable topics: minibar sensors, passport for casino, elevator waits, dress code, booking in advance, sunscreen at rooftop pool, coffee machines, USB charging ports, valet parking, light/laser show timing, Spago restaurant tips
- Aspect-based grouping: room 23.7%, facilities 15.3%, food & beverage 11.7%, value 7.4%, service 5.6%, location 5.2%, other 31.1%
- Figures saved: `topic_barchart.png`, `topic_wordclouds.png`, `aspect_distribution.png`
- Summaries saved: `topic_summary.csv`, `aspect_summary.csv`

### Phase 6 — Error Analysis (DONE, 2026-03-19):
- `src/error_analysis.py`: Systematic error categorisation across all models
- **Categorisation heuristics** classify each sentence into: explicit, implicit, comparative, complaint, positive_modal, imperative, other
- **Error analysis on MBS test set (80 sentences, 15 suggestions):**

  | Model | FP | FN | Key FP pattern | Key FN pattern |
  |---|---|---|---|---|
  | Regex | 11 | 7 | Imperative endorsements | Implicit suggestions |
  | TF-IDF+LR | 4 | 13 | Low recall | Nearly all categories missed |
  | BERT (SemEval) | 3 | 12 | Imperative endorsements | Low recall |
  | **BERT (SemEval+MBS)** | **7** | **6** | Imperative endorsements, positive modals | Implicit suggestions |

- **Key patterns:** "recommend" used for endorsement (main FP source); implicit suggestions without keyword triggers (main FN source); BERT stage-2 has best FP/FN balance (7 vs 6)
- Figures: per-model error breakdowns (4) + cross-model comparison chart

### NOT done:
- Final report (Phase 7) — LaTeX draft exists on `report-writing` branch with Sections 1–4 drafted; Sections 5–7 + Abstract need final results filled in
- Presentation recording (Phase 8) — not started

---

## 4. What Needs to Happen Next

**12 calendar days remain until deadline (31 March 2026).** All pipeline phases (1–6) are complete. Remaining work is report writing and presentation.

### Priority 1: Final Report (NOW — by ~28 March)

| Task | Owner | Status | Notes |
|---|---|---|---|
| Update LaTeX report with final results | Chase / Clara | **Ready** | All results available; `report-writing` branch has draft with [MISSING] placeholders |
| Fill Sections 5–7 (Results, Discussion, Conclusion) | Clara / Xiayi / Leah | **Not started** | Evidence exists in `notes/coding_assistant_update.md` |
| Fill Abstract with key finding | Clara | **Not started** | — |
| Fill annotation subsection in Section 3 | Clara | **Not started** | IAA + label distribution data available |
| Reproducibility check | Chase | **Not started** | Run `python run_pipeline.py` end-to-end |
| Compile final PDF, verify word count | All | **Waiting** | Must be ≤ 4,000 words (excl. figures/tables/appendix) |

- **Key narrative threads:**
  - Domain gap story: SemEval → hotel transfer degradation → stage-2 adaptation recovery
  - Three-tier model comparison: regex → TF-IDF+LR → BERT
  - IAA process: kappa 0.379, guideline revision, transparent reporting
  - Actionable insights: 56 topics across 6 hotel aspects

### Priority 2: Presentation (by 31 March)

| Task | Owner | Status |
|---|---|---|
| Slide deck (8–10 slides) | Xiayi (lead) | **Not started** |
| Recording (5 min ±30 sec) | All members | **Not started** |

### Optional Improvements (if time permits)

- Tune BERTopic `min_topic_size` to reduce outlier count (1,101/4,179 = 26%)
- Add TF-IDF+LR trained on MBS training set (in-domain small data comparison)
- Cross-validation for stage-2 (k=5) for more robust metrics on small test set

---

## 5. Key Risks & Recommendations

| Risk | Status | Mitigation |
|---|---|---|
| **Domain gap** — SemEval → hotel transfer | **Addressed** | Two-stage BERT fine-tuning validates the approach; F1 0.286 → 0.581 |
| **Annotation quality** — low initial IAA | **Addressed** | Fleiss' kappa 0.379 reported transparently; guidelines revised; gold labels resolved |
| **Class imbalance** — suggestions are minority class | **Addressed** | 18.2% suggestion rate; metrics focus on F1_sugg and PR-AUC, not accuracy |
| **Small test set volatility** — 80 sentences, 15 suggestions | **Known risk** | Single misclassification changes F1 by several points; acknowledge in report |
| **Leah's annotation tendency** — 29% suggestion rate vs 12% for others | **Known risk** | Inherent to split annotation; note in report limitations |
| **Reproducibility** — code must run end-to-end | **To verify** | Run `python run_pipeline.py` before submission |
| **Word count** — 4,000 word limit is tight | **Active** | Use tables/figures (excluded); focus on domain-transfer narrative |
| **Report timeline** — 12 days for report + presentation | **Active** | LaTeX draft exists; Sections 1–4 drafted; fill results from completed pipeline |

---

## 6. Team Workload Split — Remaining Work

| Member | Report section(s) | Other |
|---|---|---|
| Chase | Reproducibility check, pipeline support | Update LaTeX with final results, compile PDF |
| Clara | Abstract, Data (§3), Methods (§4), Results (§5) | — |
| Leah | Introduction (§1), Related Work (§2), Conclusion (§7) | — |
| Xiayi | Error Analysis & Discussion (§6) | Presentation slides (lead) |

*All pipeline coding is complete. Remaining deliverables: final report (LaTeX, `report-writing` branch) and 5-minute recorded presentation.*
