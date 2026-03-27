# NLP_Team6 — Suggestion Mining in High-Rated Hotel Reviews

## Project Overview

Online customer reviews are a crucial source of feedback in the hotel and hospitality industry. While prior NLP research has largely focused on sentiment analysis and low-rated reviews, high-star (4–5 star) reviews may still contain implicit and actionable suggestions embedded within otherwise positive narratives.

This project investigates whether NLP models can identify such "hidden suggestions" in high-rated luxury hotel reviews and transform them into operational insights for service improvement.

## Research Question

**To what extent can NLP models detect actionable improvement suggestions within high-star hotel reviews?**

We focus on a luxury hospitality context (Marina Bay Sands, Singapore), where dissatisfaction and advice may be expressed indirectly or politely despite high satisfaction scores. A central analytical theme is **domain transfer**: the SemEval-2019 training data comes from software developer forums, so we investigate how well suggestion-detection models transfer to the hotel domain and what adaptations are needed.

## Datasets

| Dataset | Role | Source |
|---|---|---|
| Marina Bay Sands TripAdvisor reviews (10k+) | Primary (unlabelled) — for inference and insight generation | [Kaggle](https://www.kaggle.com/datasets/lucashkliu/marina-bay-sands-hotel-reviews-on-tripadvisor) |
| SemEval-2019 Task 9 Subtask A | Reference (labelled) — for training and benchmarking | [GitHub](https://github.com/Semeval2019Task9/Subtask-A) |

- Scope: filtered to **4–5 star reviews** only
- Unit of analysis: **sentence-level** (reviews segmented into sentences using spaCy)
- Raw data files are not tracked in Git. See `data/README.md` for placement instructions.

## Methodology (Pipeline)

1. **Data Collection & Preprocessing**
   - Load raw TripAdvisor reviews, filter to 4–5 stars
   - Sentence segmentation with spaCy `en_core_web_sm`
   - Language filtering, deduplication, length thresholding
   - Retain stopwords (`should`, `would`, `if`) — critical suggestion cues

2. **Exploratory Data Analysis**
   - Sentence length distributions, modal verb frequencies
   - Domain gap analysis: SemEval (software forums) vs. MBS (hotel reviews)
   - Temporal review patterns

3. **Annotation**
   - Binary sentence-level labelling: Suggestion vs. Non-Suggestion
   - Hotel-domain-adapted guidelines based on SemEval-2019 definitions
   - Inter-annotator agreement (IAA) measured via Fleiss' kappa

4. **Model Development** — Three-tier comparison:
   - **Tier 1 — Regex baseline:** Rule-based pattern matching on modal verbs and imperative forms
   - **Tier 2 — TF-IDF + Logistic Regression:** Traditional ML baseline trained on SemEval data
   - **Tier 3 — BERT classifier:** Two-stage fine-tuning (SemEval → MBS annotated data) for domain adaptation

5. **Insight Generation**
   - BERTopic clustering on extracted suggestion sentences
   - Aspect-based grouping (room, food, service, facilities, location)

6. **Evaluation & Validation**
   - Precision / Recall / F1 on held-out test set (80 sentences, 15 suggestions)
   - Cross-domain comparison (SemEval test vs. MBS test)
   - Catastrophic forgetting analysis (stage-2 evaluated on SemEval test)

7. **Error Analysis**
   - Systematic categorisation of false positives (imperative endorsements, positive modals) and false negatives (implicit suggestions)
   - Cross-model error comparison across all 4 model configurations

## Repository Structure

```
NLP_Team6/
├── data/
│   ├── raw/                                    Raw TripAdvisor Excel file (git-ignored)
│   └── processed/                              Filtered reviews, segmented sentences, annotated data
├── notebooks/
│   └── 00_full_pipeline.ipynb                  Full end-to-end pipeline (all 6 phases)
├── src/                                        Core pipeline logic (all reusable functions)
│   ├── config.py                               Centralised paths, seeds, and parameters
│   ├── data_loading.py                         Load + filter reviews
│   ├── preprocessing.py                        Sentence segmentation + cleaning
│   ├── eda.py                                  EDA plots and domain gap analysis
│   ├── annotation.py                           Sampling, IAA, merge, train/test split
│   ├── baselines.py                            Regex classifier + TF-IDF + Logistic Regression
│   ├── bert_model.py                           Two-stage BERT fine-tuning and prediction
│   ├── evaluation.py                           Shared metrics, PR curves, confusion matrices
│   ├── topic_modeling.py                       BERTopic clustering + aspect grouping
│   └── error_analysis.py                       Error categorisation and comparison plots
├── MSIN0221_GroupProject_Team6_Proposal.pdf    Project proposal
├── MSIN0221_GroupProject_Team6_Report.pdf      Project report (ACL template) + references + appendices
├── run_pipeline.py                             CLI runner: python run_pipeline.py [--phase N]
├── outputs/                                    Figures (25 PNGs), model checkpoints (git-ignored)
├── external/                                   External reference code and data (SemEval)
└── requirements.txt
```

**Architecture:** All logic lives in `src/` modules. Notebooks are thin wrappers that call `src/` functions and add display/narrative. `run_pipeline.py` runs the same functions for one-command reproducibility testing.

## Setup

### 1) Clone

```bash
git clone <repo-url>
cd NLP_Team6
```

### 2) Environment (uv + venv) — Recommended

**macOS:**
```bash
brew install uv
uv venv --seed .venv
source .venv/bin/activate
uv pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

**Windows (PowerShell):**
```bash
winget install astral-sh.uv
uv venv --seed .venv
.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

**VS Code:** Use "Python: Select Interpreter" and choose the `.venv` interpreter.

### 3) Verification

```bash
python -c "import spacy, pandas, sklearn, torch, transformers; print('Core OK')"
python -c "import bertopic; print('BERTopic OK')"
```

### 4) Data placement

1. Download the Marina Bay Sands dataset from [Kaggle](https://www.kaggle.com/datasets/lucashkliu/marina-bay-sands-hotel-reviews-on-tripadvisor).
2. Place the raw file at: `data/raw/tripadvisor_mbs_review_from201501_v2.xlsx`
3. SemEval data is already included in `external/semeval2019task9/`.

## How to Run

### Option A: Pipeline script (recommended for reproducibility testing)

```bash
python run_pipeline.py              # run all available phases
python run_pipeline.py --phase 1    # run only Phase 1
```

### Option B: Notebook (recommended for exploration)

Open `notebooks/00_full_pipeline.ipynb` to run all 6 phases interactively with narrative and inline visualisations.

Both methods call the same `src/` functions and produce identical outputs. Processed data is saved to `data/processed/`.

## Team Members

| Name | Student Number |
|---|---|
| Chase Sun | 25095830 |
| Clara Yang | 25149169 |
| Leah Luo | 25072251 |
| Xiayi Zhu | 25062124 |

Contributions are detailed in the final report.

## Reproducibility

- All paths and random seeds are centralised in `src/config.py`
- All pipeline logic lives in `src/` modules — notebooks and `run_pipeline.py` produce identical outputs
- `python run_pipeline.py` provides one-command end-to-end verification
- `requirements.txt` pins minimum dependency versions
- Hardware: developed on Apple Silicon Mac (PyTorch MPS backend); Windows/CUDA also supported

## Acknowledgements

- [SemEval-2019 Task 9: Suggestion Mining](https://aclanthology.org/S19-2151/) (Negi et al., 2019)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [spaCy](https://spacy.io/)
- [BERTopic](https://maartengr.github.io/BERTopic/)
