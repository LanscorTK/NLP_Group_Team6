# Implementation Plan

**Project:** Suggestion Mining in High-Rated Hotel Reviews
**Created:** 2026-03-14
**Last updated:** 2026-03-20
**Deadline:** 2026-03-31, 10:00 AM UK time
**Days remaining:** 11

---

## Coding Standards

### Progress bars (tqdm)

Use `tqdm` for any loop or `.apply()` that processes **1,000+ items**. This includes:
- DataFrame `.iterrows()` / `.itertuples()` loops over large datasets
- `.apply()` / `.progress_apply()` on large Series
- Manual `for` loops over text collections (vocab building, counting)
- Model training epochs and batch iteration

**Pattern for loops:**
```python
from tqdm import tqdm
for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
```

**Pattern for `.apply()`:**
```python
from tqdm import tqdm
tqdm.pandas(desc="Filtering")
result = df["col"].progress_apply(my_func)
```

Skip tqdm for operations under 1,000 items (annotation, kappa computation).

---

## Code Architecture: Hybrid (src/ modules + thin notebooks + runner script)

All pipeline logic lives in `src/` Python modules. Notebooks in `notebooks/` are thin wrappers that import from `src/`, call functions, and add narrative + visualisation. A `run_pipeline.py` at the project root runs the full pipeline without Jupyter for reproducibility testing.

```
src/
├── config.py            Centralised paths, seeds, parameters
├── data_loading.py      Load + filter reviews (Phase 1)
├── preprocessing.py     Sentence segmentation + cleaning (Phase 1)
├── eda.py               Plotting functions, domain gap stats (Phase 2 — DONE)
├── annotation.py        Sampling, IAA computation, train/test split (Phase 3 — DONE)
├── baselines.py         Regex classifier, TF-IDF+LR (Phase 4 — DONE)
├── bert_model.py        BERT fine-tuning, evaluation (Phase 4 — DONE, Stage-1 + Stage-2)
├── evaluation.py        Shared metrics, confusion matrix, comparison tables (Phase 4/6 — DONE)
├── topic_modeling.py    BERTopic clustering, aspect grouping (Phase 5 — DONE, re-run with stage-2)
└── error_analysis.py    Error categorisation, summary tables, plots (Phase 6 — DONE)

notebooks/               Thin wrappers: import from src/, add display + narrative
├── 00_full_pipeline.ipynb   Full end-to-end pipeline (all 6 phases)
├── 01–08_*.ipynb            Per-phase notebooks
run_pipeline.py          CLI runner: python run_pipeline.py [--phase N]
```

**Why this architecture:**
- **No code duplication** — shared utilities (metrics, plotting) in `src/evaluation.py` reused across notebooks 05, 06, 07
- **Reproducibility** — `python run_pipeline.py` runs end-to-end without Jupyter; notebooks produce identical outputs
- **Testability** — functions in `src/` are debuggable and testable outside notebook context
- **Report-friendly** — notebooks retain narrative flow for Jupyter Notebook PDF export

---

### Hardware note — Apple Silicon Mac (local GPU)

Chase's primary development machine is an **Apple-chip MacBook**. This enables local GPU-accelerated training via PyTorch's MPS (Metal Performance Shaders) backend, eliminating the need for Colab or DeepFlow for most tasks:

- **PyTorch MPS:** Available in PyTorch >= 2.0. Use `device = torch.device("mps")` instead of `"cuda"`. Most HuggingFace `Trainer` workflows auto-detect MPS.
- **BERT fine-tuning:** `bert-base-uncased` (~110M params) runs comfortably on Apple Silicon with 16GB+ unified memory. Batch size 16 should work; reduce to 8 if memory-pressure warnings appear.
- **BERTopic / sentence-transformers:** Embedding generation with `all-MiniLM-L6-v2` works on MPS. UMAP/HDBSCAN (used internally by BERTopic) are CPU-only but fast enough for the data sizes involved.
- **Known MPS quirks:** Some PyTorch ops fall back to CPU silently (e.g., certain loss functions, `torch.linalg` operations). If training is unexpectedly slow, check for CPU fallback warnings. Set `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` to avoid OOM crashes on memory-constrained machines.
- **Reproducibility:** MPS results may differ slightly from CUDA/CPU due to floating-point non-determinism. Set seeds and document the device used. For the report, note the hardware (e.g., "Apple M2, 16GB unified memory, PyTorch 2.x MPS backend").

This means **Colab/DeepFlow are fallbacks**, not the primary compute plan. Local training avoids session timeouts and simplifies checkpoint management.

---

## Phase 1: Data Loading & Preprocessing — DONE

### Step 1.1 — Load raw reviews — DONE
- **Module:** `src/data_loading.py` → **Notebook:** `01_data_loading.ipynb`
- **Owner:** Chase
- **Input:** `data/raw/tripadvisor_mbs_review_from201501_v2.xlsx`
- **Output:** `data/processed/mbs_filtered_reviews.csv`
- **Results:**
  - Raw Excel columns: `user_id`, `contributions`, `date_of_stay`, `ratings`, `title`, `content`
  - 10,232 raw reviews → 8,572 after 4-5 star filter → **8,571** after dropping 1 null content row
  - Output columns: `review_id`, `rating`, `date`, `review_title`, `review_text`
  - Rating split: 2,414 four-star + 6,157 five-star

> **:white_check_mark: RESOLVED — Column names**
> Raw columns are: `user_id`, `contributions`, `date_of_stay` (datetime), `ratings` (int 1-5), `title`, `content`. Only 1 null in `content`, 315 missing dates. Only 2 non-English reviews in the entire dataset.

### Step 1.2 — Sentence segmentation & cleaning — DONE
- **Module:** `src/preprocessing.py` → **Notebook:** `02_preprocessing.ipynb`
- **Owner:** Chase
- **Input:** `data/processed/mbs_filtered_reviews.csv`
- **Output:** `data/processed/mbs_sentences.csv`
- **Results:**
  - 8,571 reviews → 72,391 sentences (8.4 avg per review)
  - Language filter (ASCII ratio < 0.5): removed 1 sentence
  - Dedup: removed 1,137 exact duplicates
  - Length filter (4-100 tokens): removed 1,168 (1,115 too short, 53 too long)
  - **Final: 70,085 sentences** from 8,559 unique reviews
  - Median sentence length: 15 tokens
  - Output columns: `sentence_id`, `review_id`, `rating`, `sentence_text`, `n_tokens`

> **:white_check_mark: RESOLVED — spaCy sentence boundaries**
> Spot-checked 10 random sentences — segmentation quality is acceptable for informal hotel review text. No custom rules added; occasional over-splits on ellipses are minor and tolerable for 70k sentences.

> **:white_check_mark: RESOLVED — Language detection**
> Only 1 non-English sentence detected (Japanese/encoded text). ASCII ratio threshold of 0.5 is conservative and sufficient. Mixed-language reviews are not a meaningful issue in this dataset.

---

## Phase 2: Exploratory Data Analysis 

### Step 2.1 — MBS dataset statistics
- **Notebook:** `03_eda.ipynb`
- **Owner:** Clara
- **Actions:**
  1. Distribution of reviews by star rating (4 vs. 5)
  2. Distribution of review length (word count per review)
  3. Distribution of sentence length (tokens per sentence) after segmentation
  4. Temporal distribution: reviews per month/year — look for COVID dip, post-pandemic recovery
  5. Top 50 unigrams and bigrams (with and without stopwords) — word cloud optional
- **Visualisations needed:** At least 3-4 publication-quality plots (histograms, bar charts, time series)
- **Automatable:** Yes — fully scriptable

### Step 2.2 — Modal verb and suggestion-signal analysis
- **Notebook:** `03_eda.ipynb`
- **Owner:** Clara
- **Actions:**
  1. Count frequency of modal verbs (`should`, `would`, `could`, `might`, `need to`, `ought to`) across all MBS sentences
  2. Count imperative-form indicators (`please`, `recommend`, `suggest`, `consider`, `try`)
  3. Compare these frequencies to the SemEval training data (same counts computed on `V1.4_Training.csv`)
  4. Plot side-by-side bar chart: MBS vs. SemEval modal verb distributions
- **Automatable:** Yes — fully scriptable

### Step 2.3 — Domain gap analysis (SemEval vs. MBS)
- **Notebook:** `03_eda.ipynb`
- **Owner:** Clara
- **Actions:**
  1. Load SemEval `V1.4_Training.csv` — parse `id`, `text`, `label` columns
  2. Compare sentence length distributions (SemEval vs. MBS)
  3. Compare vocabulary overlap: what fraction of MBS vocabulary appears in SemEval, and vice versa?
  4. Compare suggestion-class base rates: what % of SemEval sentences are suggestions vs. expected % in MBS?
  5. Identify SemEval-specific jargon (e.g., "API", "SDK", "developer") that will never appear in hotel reviews — this motivates domain adaptation
  6. Produce a summary table for the report
- **Automatable:** Yes — fully scriptable

> **:warning: ISSUE — SemEval CSV parsing**
> The SemEval `V1.4_Training.csv` uses an unusual format: `id,text,label` where text is double-quoted and may contain internal commas and quotes. Verify that `pd.read_csv` parses it correctly — you may need `quotechar='"'` or manual cleaning. Inspect the first 20 rows after loading.

---

## Phase 3: Annotation — DONE

### Step 3.1 — Draft annotation guidelines :page_facing_up: MANUAL
- **Output:** `docs/annotation_guidelines.md`
- **Owner:** Leah (lead), all members review
- **Actions:**
  1. Start from SemEval-2019 Task 9 Subtask A definition: "A suggestion is a sentence that proposes an improvement, gives advice, or recommends an action"
  2. Adapt for hotel domain — add hotel-specific examples and edge cases:
     - **Clear suggestion:** "They should add more vegetarian options at breakfast" → `1`
     - **Clear non-suggestion:** "The room was beautiful and spacious" → `0`
     - **Edge case — wish/desire:** "I wish the pool was open later" → `1` (expresses desired improvement)
     - **Edge case — implicit comparison:** "Other hotels in the area offer free airport shuttle" → `1` (implies the hotel should too)
     - **Edge case — conditional praise:** "The breakfast was decent but could have more variety" → `1` (contains actionable suggestion)
     - **Edge case — past-tense complaint without action:** "The check-in took over an hour" → `0` (complaint, not a suggestion)
     - **Edge case — generic positive with modal:** "I would definitely stay here again" → `0` (modal verb but not a suggestion)
  3. Define the decision boundary: **if the sentence proposes a concrete action the hotel could take, label `1`; if it only describes an experience or opinion without an actionable direction, label `0`**
  4. All 4 members read and discuss the guidelines before labelling begins
- **Automatable:** No — requires human judgement to define edge cases

> **:warning: ISSUE — The hardest annotation decision**
> The boundary between "complaint" and "implicit suggestion" is genuinely ambiguous. "The walls were thin and we could hear everything" — is this a suggestion (implying the hotel should soundproof) or just a complaint? The team must agree on a consistent rule. Recommendation: label as suggestion **only if** the sentence contains a forward-looking or actionable element, not just a description of a negative experience.

### Step 3.2 — IAA calibration round :page_facing_up: MANUAL
- **Notebook:** `04_annotation.ipynb` (for sampling and kappa computation)
- **Owners:** All 4 members
- **Actions:**
  1. Randomly sample 100 sentences from the preprocessed MBS dataset (use `RANDOM_SEED = 42` for reproducibility)
  2. Export to a shared spreadsheet (Google Sheets or Excel) with columns: `sentence_id`, `sentence_text`, `label_chase`, `label_clara`, `label_leah`, `label_xiayi`
  3. Each member independently labels all 100 sentences as `1` (suggestion) or `0` (non-suggestion)
  4. **No discussion during labelling** — independence is critical for valid IAA
  5. Compute Fleiss' kappa (4 raters, binary) in the notebook using `sklearn.metrics.cohen_kappa_score` (pairwise) or `statsmodels`/`nltk` for Fleiss
  6. If kappa < 0.6: discuss the top disagreements, refine guidelines, re-label the disputed sentences
  7. If kappa >= 0.6: proceed with split annotation
  8. Record the final kappa score and disagreement resolution process — this goes directly into the Methods section of the report
- **Automatable:** Sampling and kappa computation are scriptable; **labelling is fully manual**

> **:warning: ISSUE — Target kappa and fallback**
> A kappa of 0.6-0.8 is "substantial agreement" and acceptable for a coursework project. If you get < 0.5 after the calibration round, the guidelines need serious revision. Budget an extra half-day for this. Do not skip IAA — it is one of the easiest ways to demonstrate methodological rigour and score marks.

### Step 3.3 — Full annotation :page_facing_up: MANUAL
- **Notebook:** `04_annotation.ipynb` (for sampling, merging, and split management)
- **Owners:** All 4 members
- **Actions:**
  1. After calibration, randomly sample an additional 200-400 sentences (total target: 300-500 including the calibration 100)
  2. Split the additional sentences into 4 roughly equal batches — one per member
  3. Each member labels their batch independently using the refined guidelines
  4. Merge all labels into a single DataFrame: `sentence_id`, `sentence_text`, `label`, `annotator`
  5. Create train/test split (80/20, stratified by label) — save as:
     - `data/processed/mbs_annotated_train.csv`
     - `data/processed/mbs_annotated_test.csv`
  6. Report label distribution (% suggestions vs. non-suggestions) — expect suggestions to be the minority class (~10-25% of sentences in high-star reviews)
- **Automatable:** Splitting, merging, and saving are scriptable; **labelling is fully manual**

> **:warning: ISSUE — Class imbalance**
> In high-star reviews, genuine suggestions are rare. If only 10-15% of sentences are suggestions, the annotated test set may have very few positive examples (e.g., 10-15 out of 100). This makes F1 volatile. Consider oversampling the annotation pool: pre-filter sentences containing modal verbs or suggestion signals to increase the suggestion rate in the annotation set. Document this enrichment strategy clearly — it is methodologically sound as long as you acknowledge it.

---

## Phase 4: Model Development — DONE

### Step 4.1 — Regex baseline
- **Notebook:** `05_baseline_regex.ipynb`
- **Owner:** Xiayi
- **Actions:**
  1. Define pattern lists:
     - Modal suggestion patterns: `should`, `would be better`, `could improve`, `might want to`, `need to`, `ought to`
     - Imperative patterns: `please`, `recommend`, `suggest`, `consider`, `try to`, `make sure`
     - Conditional patterns: `it would be nice if`, `I wish`, `if only`, `hopefully`
  2. Classify a sentence as suggestion if **any** pattern matches (case-insensitive)
  3. Evaluate on:
     - SemEval test set (`SubtaskA_EvaluationData_labeled.csv`) — in-domain baseline
     - MBS annotated test set — cross-domain baseline
  4. Report Precision, Recall, F1 for each
  5. **Log false positives** — sentences like "I would definitely come back" that match modal patterns but are not suggestions. These examples motivate why ML is needed.
- **Automatable:** Yes — fully scriptable

### Step 4.2 — TF-IDF + Logistic Regression baseline
- **Notebook:** `05_baseline_regex.ipynb` (same notebook, second section)
- **Owner:** Xiayi
- **Actions:**
  1. Load SemEval training data (`V1.4_Training.csv`)
  2. Build a TF-IDF vectoriser (unigrams + bigrams, max 10k features, min_df=2)
  3. Train Logistic Regression with `sklearn` (use `class_weight='balanced'` if imbalanced)
  4. Evaluate on:
     - SemEval test set — in-domain performance
     - MBS annotated test set — cross-domain performance (this is the key comparison)
  5. Report Precision, Recall, F1 for each
  6. Compare to regex: the F1 gap on SemEval (in-domain) shows the value of learned features; the F1 gap on MBS (cross-domain) shows how much domain shift degrades a bag-of-words model
- **Automatable:** Yes — fully scriptable

> **:warning: ISSUE — TF-IDF vocabulary mismatch across domains**
> The TF-IDF vectoriser is fitted on SemEval (software forum) vocabulary. Hotel-domain words like "breakfast", "concierge", "amenities" will be out-of-vocabulary or very low weight. This is expected and is itself a finding — document it. Consider also fitting a second TF-IDF+LR model on the small MBS annotated training set as a comparison point (in-domain small data vs. out-of-domain large data).

### Step 4.3 — BERT stage 1: fine-tune on SemEval
- **Notebook:** `06_bert_model.ipynb`
- **Owner:** Chase
- **Platform:** Local Apple Silicon Mac (MPS backend); Colab/DeepFlow as fallback
- **Actions:**
  1. Load `bert-base-uncased` from HuggingFace `transformers`
  2. Prepare SemEval training data as a HuggingFace `Dataset` (tokenise with BERT tokeniser, max_length=128)
  3. Set device: `torch.device("mps")` — verify with a small forward pass before full training
  4. Fine-tune for 3-4 epochs with:
     - Learning rate: 2e-5
     - Batch size: 16 (reduce to 8 if memory-pressure warnings on MPS)
     - Weight decay: 0.01
     - Warmup steps: 10% of total
  5. Evaluate on SemEval test set — this is the **in-domain benchmark** (expect F1 ~0.75-0.85 based on SemEval-2019 leaderboard)
  6. Save the fine-tuned model checkpoint locally (no session timeout risk unlike Colab)
- **Automatable:** Yes — scriptable, runs on local MPS GPU

> **:warning: ISSUE — MPS compatibility**
> Most HuggingFace `Trainer` workflows auto-detect MPS, but some operations may silently fall back to CPU. If training is unexpectedly slow (>1 hour for 7k sentences), check for fallback warnings. Known workaround: set `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` in the environment to prevent OOM. If MPS causes errors on specific ops, fall back to Colab free GPU as a backup.

> **:warning: ISSUE — Hyperparameter selection**
> The brief rewards process understanding over raw performance. Rather than grid-searching, pick standard hyperparameters from the literature (the values above are from the original BERT paper) and justify them. If time permits, try one alternative (e.g., learning rate 5e-5) and report the comparison. This shows awareness without burning time.

### Step 4.4 — BERT stage 2: fine-tune on MBS annotated data
- **Notebook:** `06_bert_model.ipynb`
- **Owner:** Chase
- **Actions:**
  1. Load the stage-1 checkpoint (SemEval-tuned BERT)
  2. Further fine-tune on `mbs_annotated_train.csv` (likely 240-400 sentences)
  3. Use a lower learning rate (1e-5) and fewer epochs (2-3) to avoid overfitting on the small dataset
  4. Evaluate on `mbs_annotated_test.csv`
  5. **Key comparison:** stage-1 (SemEval only) vs. stage-2 (SemEval + MBS) performance on MBS test set — the gap quantifies the benefit of domain adaptation
  6. Save the stage-2 model checkpoint
- **Automatable:** Yes — scriptable, requires GPU

> **:warning: ISSUE — Small training set overfitting**
> With only 200-400 training sentences, stage-2 fine-tuning risks overfitting. Watch for: training loss going to near-zero while validation loss increases. Mitigations: use early stopping (patience=1), keep the learning rate very low, freeze lower BERT layers, or use k-fold cross-validation (k=5) instead of a single train/test split. The cross-validation approach is stronger methodologically and more stable with small data, but takes longer to code and run.

---

## Phase 5: Insight Generation — DONE (re-run with stage-2 predictions)

### Step 5.1 — BERTopic clustering on extracted suggestions
- **Notebook:** `07_topic_modeling.ipynb`
- **Owner:** Leah
- **Actions:**
  1. Collect all sentences predicted as suggestions by the best-performing model (stage-2 BERT)
  2. Run BERTopic with default sentence-transformers embeddings (`all-MiniLM-L6-v2`) — embedding generation runs on MPS; UMAP/HDBSCAN run on CPU (fast enough for expected data size of <1k suggestions)
  3. Inspect the resulting topics — are they coherent and actionable?
  4. Visualise: topic word clouds, topic frequency bar chart, inter-topic distance map
  5. Manually label the top 5-10 topics with descriptive names (e.g., "breakfast quality", "room maintenance", "pool/spa hours")
  6. If BERTopic clusters are too noisy, fall back to **aspect-based grouping**: define 5-6 hotel aspects (room, food & beverage, service, facilities, location, value) and classify each suggestion into an aspect using keyword matching or zero-shot classification (`transformers` pipeline with `facebook/bart-large-mnli`)
- **Automatable:** Clustering and visualisation are scriptable; **topic labelling is manual**

> **:warning: ISSUE — BERTopic minimum cluster size**
> If the total number of extracted suggestions is small (e.g., 50-100), BERTopic may produce only 1-2 topics or assign most sentences to the outlier topic (-1). Lower `min_topic_size` to 3-5 and consider using `nr_topics="auto"` to reduce topic count. If results are still poor, the aspect-based grouping fallback is simpler and guaranteed to produce interpretable results.

### Step 5.2 — Actionable suggestions summary
- **Notebook:** `07_topic_modeling.ipynb`
- **Owner:** Leah
- **Actions:**
  1. For each topic/aspect, list the top 5 representative suggestion sentences
  2. Create a summary table: topic, count, example sentences, potential hotel action
  3. This table goes directly into the Results section and the presentation
- **Automatable:** Table generation is scriptable; **interpreting actionability is manual**

---

## Phase 6: Evaluation & Error Analysis — DONE

### Step 6.1 — Quantitative evaluation
- **Notebook:** `06_bert_model.ipynb` (evaluation section)
- **Owner:** Chase
- **Actions:**
  1. For each model (regex, TF-IDF+LR, BERT stage-1, BERT stage-2), compute on MBS test set:
     - Precision, Recall, F1 (macro and per-class)
     - Confusion matrix (2x2)
  2. Also compute on SemEval test set for all models — this gives the in-domain vs. cross-domain comparison
  3. Create a summary comparison table:

     | Model | SemEval F1 | MBS F1 | Delta |
     |---|---|---|---|
     | Regex | ? | ? | ? |
     | TF-IDF + LR | ? | ? | ? |
     | BERT (SemEval only) | ? | ? | ? |
     | BERT (SemEval + MBS) | ? | ? | ? |

  4. Plot confusion matrices as heatmaps (4 panels, one per model)
- **Automatable:** Yes — fully scriptable

### Step 6.2 — Manual validation :page_facing_up: MANUAL
- **Owner:** Xiayi
- **Actions:**
  1. From the BERT stage-2 predictions on the **full unlabelled MBS dataset** (not just the annotated test set), randomly sample:
     - 100 sentences predicted as suggestions
     - 100 sentences predicted as non-suggestions
  2. Manually verify each: is the model's prediction correct?
  3. Compute manual-validation precision and recall
  4. This provides an estimate of real-world performance beyond the small annotated test set
- **Automatable:** Sampling is scriptable; **validation labelling is fully manual**

### Step 6.3 — Error analysis by suggestion type
- **Notebook:** `06_bert_model.ipynb` or a dedicated section
- **Owner:** Xiayi
- **Actions:**
  1. Categorise false negatives (missed suggestions) and false positives (non-suggestions labelled as suggestions) from the MBS test set into types:
     - **Explicit suggestions:** "They should add..." / "I recommend..."
     - **Implicit suggestions:** "It would be nice if..." / "I wish..."
     - **Comparative suggestions:** "Other hotels offer..." / "Compared to..."
     - **Disguised complaints:** "The breakfast was disappointing" (complaint, not suggestion — model may over-predict)
     - **Positive modals:** "I would definitely return" (modal verb, not a suggestion — model may over-predict)
  2. Count how many errors fall into each category
  3. Write 2-3 paragraphs interpreting the patterns — this is the core of the Error Analysis & Discussion section
- **Automatable:** Categorisation is partially scriptable (keyword heuristics for type), but **interpretation is manual**

> **:warning: ISSUE — Error analysis sample size**
> If the MBS test set is small (60-100 sentences) and the model is reasonably accurate, there may be very few errors to analyse (e.g., 10-15 total mistakes). This is fine — analyse every error individually rather than computing statistics over types. Quality of interpretation matters more than quantity.

---

## Phase 7: Report Writing :page_facing_up: MANUAL

### Step 7.1 — Report structure and drafting
- **Format:** LaTeX (ACL template) or Jupyter Notebook PDF
- **Max:** 4,000 words (figures, tables, appendices, references excluded from count)
- **Owners:** Assigned by section (see below)

| Section | Owner | Key content | Approx. words |
|---|---|---|---|
| **Title page** | All | Title, member names + student numbers, contribution breakdown | (excluded) |
| **Abstract** | Clara | 150-word summary of question, method, findings | 150 |
| **1. Introduction** | Leah | Research question, motivation (high-star reviews gap), contribution | 500 |
| **2. Related Work** | Leah | SemEval-2019, suggestion mining literature, domain adaptation in NLP | 400 |
| **3. Data** | Clara | MBS dataset description, SemEval data, preprocessing steps, annotation process + IAA | 600 |
| **4. Methods** | Clara | Three-tier model description, two-stage BERT, BERTopic | 700 |
| **5. Results** | Clara | Comparison table, confusion matrices, BERTopic topics, actionable suggestions | 600 |
| **6. Error Analysis & Discussion** | Xiayi | Error types, domain gap findings, limitations, what worked and what didn't | 700 |
| **7. Conclusion & Future Work** | Leah | Summary of answer to research question, next steps | 250 |
| **References** | All | (excluded from count) |  |
| **Appendix** | All | Full annotation guidelines, additional figures | (excluded) |

- **Automatable:** No — writing is fully manual. Figures and tables can be exported from notebooks.

> **:warning: ISSUE — 4,000 word limit is tight**
> With 7 phases to cover, every section must be concise. Strategy:
> - Move detailed numbers into **tables** (excluded from word count)
> - Move supplementary figures into an **appendix** (excluded from word count)
> - The Introduction should motivate the question, not survey the field exhaustively — save that for Related Work
> - The Methods section should describe *what* you did and *why*, not provide a code tutorial
> - Do NOT describe every preprocessing step in prose — summarise in a pipeline diagram (a figure)

> **:warning: ISSUE — Member contribution breakdown**
> The brief requires this on the first page. Agree on contributions early, not the night before submission. Suggested approach: each member writes a 1-sentence summary of their contribution; compile into a table.

### Step 7.2 — Reproducibility check
- **Owner:** Chase
- **Actions:**
  1. Run `python run_pipeline.py` from project root — verify all phases complete without error
  2. Run each notebook individually (01 → 07) — verify outputs match pipeline script
  3. Verify `src/config.py` paths work correctly
  4. Verify `requirements.txt` installs all needed packages (including BERTopic dependencies)
  5. Ensure data files are included or have clear download instructions in `data/README.md`
  6. Ensure the spaCy model download instruction is documented
- **Automatable:** `run_pipeline.py` provides one-command verification; notebook-by-notebook debugging is manual

---

## Phase 8: Presentation :page_facing_up: MANUAL

### Step 8.1 — Slide deck
- **Owner:** Xiayi (lead), all members contribute
- **Format:** 8-10 slides for 5 minutes
- **Suggested structure:**
  1. Title + team
  2. Motivation: why mine suggestions from high-star reviews?
  3. Data: MBS dataset, annotation process
  4. Method: three-tier pipeline (one slide, visual diagram)
  5. Results: comparison table + key finding (domain gap)
  6. Insight: top suggestion topics for hotel management
  7. Limitations & future work
  8. Q&A / Thank you
- **Automatable:** No — slide design and narrative are manual

### Step 8.2 — Recording
- **Owner:** All members
- **Actions:**
  1. Each member presents 1-2 slides (~75 seconds each)
  2. Record using Zoom, Teams, or QuickTime
  3. Final video must be 4:30-5:30 (5 min +/- 30 sec)
  4. Submit as MP4 or equivalent
- **Automatable:** No — fully manual

---

## Summary: Tasks That Must Be Done Manually

These tasks cannot be automated and require human judgement, coordination, or creative input:

| # | Task | Phase | Owner(s) | Why manual? |
|---|---|---|---|---|
| 1 | Draft hotel-domain annotation guidelines | 3.1 | Leah + all | Requires defining ambiguous linguistic boundaries specific to the hotel domain |
| 2 | IAA calibration: label 100 shared sentences | 3.2 | All 4 members | Independent human judgement required for valid IAA |
| 3 | Disagreement discussion and guideline refinement | 3.2 | All 4 members | Requires group deliberation on edge cases |
| 4 | Full annotation: label 200-400 additional sentences | 3.3 | All 4 members | Manual binary labelling |
| 5 | BERTopic topic labelling | 5.1 | Leah | Interpreting cluster content and assigning meaningful names |
| 6 | Manual validation of 200 model predictions | 6.2 | Xiayi | Human verification of model output on unlabelled data |
| 7 | Error analysis interpretation | 6.3 | Xiayi | Qualitative analysis of why the model fails |
| 8 | Report writing (all sections) | 7.1 | All (by section) | Academic writing, critical analysis, narrative construction |
| 9 | Member contribution breakdown | 7.1 | All | Requires team agreement |
| 10 | Slide deck design | 8.1 | Xiayi + all | Visual design, narrative flow |
| 11 | Presentation recording | 8.2 | All | Speaking, recording, editing |

---

## Summary: Issues Requiring Closer Attention

| # | Issue | Phase | Severity | Status |
|---|---|---|---|---|
| 1 | ~~Raw Excel column names unknown~~ | 1.1 | ~~Low~~ | **RESOLVED** — columns: `user_id`, `contributions`, `date_of_stay`, `ratings`, `title`, `content` |
| 2 | ~~spaCy sentence boundary errors on informal text~~ | 1.2 | ~~Medium~~ | **RESOLVED** — spot-checked; quality acceptable for 70k sentences |
| 3 | ~~Language detection on mixed-language reviews~~ | 1.2 | ~~Low~~ | **RESOLVED** — only 1 non-English sentence; ASCII ratio 0.5 threshold sufficient |
| 4 | ~~SemEval CSV parsing quirks~~ | 2.3 | ~~Low~~ | **RESOLVED** — spot-checked, default `pd.read_csv` works correctly |
| 5 | ~~Complaint vs. implicit suggestion boundary~~ | 3.1 | ~~High~~ | **RESOLVED** — guidelines v1.1 §4.8 added with concrete MBS examples; team consensus on 11 ties |
| 6 | ~~Low IAA (kappa < 0.6)~~ | 3.2 | ~~High~~ | **RESOLVED** — Fleiss' kappa = 0.379; guidelines revised; gold labels finalised; report transparently |
| 7 | ~~Class imbalance in annotated data~~ | 3.3 | ~~Medium~~ | **RESOLVED** — 18.2% suggestion rate achieved via 50% enrichment sampling; metrics focus on F1_sugg and PR-AUC |
| 8 | ~~TF-IDF vocabulary mismatch~~ | 4.2 | ~~Low~~ | **RESOLVED** — confirmed: F1 drops 0.596 → 0.190 cross-domain; documented as finding |
| 9 | ~~MPS compatibility~~ | 4.3 | ~~Medium~~ | **RESOLVED** — `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` set; `use_mps_device` removed for newer transformers |
| 10 | ~~Small-data overfitting in BERT stage 2~~ | 4.4 | ~~High~~ | **RESOLVED** — stage-2 trained with lr=1e-5, 3 epochs; F1 0.581 on MBS test set (no overfitting observed) |
| 11 | ~~BERTopic minimum cluster size~~ | 5.1 | ~~Medium~~ | **RESOLVED** — 4,179 suggestions (stage-2) yielded 56 coherent topics; 26% outliers (acceptable) |
| 12 | ~~Small error analysis sample~~ | 6.3 | ~~Low~~ | **RESOLVED** — every error analysed individually; categorised into 7 types across all 4 models |
| 13 | **4,000 word limit** — 7-phase pipeline is hard to cover concisely | 7.1 | Medium | Move numbers to tables, details to appendix, use pipeline diagrams |
| 14 | **Reproducibility** — code must run end-to-end from clean environment | 7.2 | Medium | `run_pipeline.py` provides one-command verification; test before submission |
