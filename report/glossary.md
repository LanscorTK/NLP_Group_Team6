# Glossary — Canonical Terminology for the Final Report

Use these exact terms in all report prose. Do not introduce synonyms or alternative labels.

---

## Datasets

| Term | Use | Do NOT use |
|------|-----|------------|
| **MBS dataset** | The full collection of TripAdvisor reviews for Marina Bay Sands | "hotel dataset", "MBS corpus", "our dataset" (ambiguous) |
| **MBS sentences** | The 70,085 sentences extracted from MBS reviews after preprocessing | "MBS data" (ambiguous — could mean reviews or sentences) |
| **SemEval dataset** | The SemEval-2019 Task 9 Subtask A labelled data from software forums | "SemEval corpus", "forum data", "source domain data" |
| **SemEval training set** | ~8,500 labelled sentences used for model training (Phases 1-2 of modelling) | "SemEval train" |
| **SemEval test set** | ~800 labelled sentences used for in-domain evaluation | "SemEval eval", "SemEval validation" |
| **MBS calibration sample** | 100 sentences labelled by all 4 annotators for IAA | "calibration set", "pilot sample" |
| **MBS annotated set** | All human-labelled MBS sentences (calibration + split annotation) | "gold set", "labelled data" (too vague) |
| **MBS train split** / **MBS test split** | 80/20 stratified split of the MBS annotated set | "train set" / "test set" (ambiguous without "MBS") |

---

## Models

| Term | Use | Do NOT use |
|------|-----|------------|
| **regex baseline** | Rule-based classifier using hand-crafted modal/imperative/conditional patterns | "rule-based model", "pattern matcher" |
| **TF-IDF + LR baseline** | TF-IDF vectoriser with logistic regression, trained on SemEval | "ML baseline", "logistic regression model" (incomplete) |
| **BERT stage-1** | bert-base-uncased fine-tuned on SemEval training data only | "SemEval BERT", "base BERT" |
| **BERT stage-2** | BERT stage-1 further fine-tuned on MBS train split | "adapted BERT", "domain BERT", "MBS BERT" |
| **three-tier pipeline** | The progression regex -> TF-IDF + LR -> BERT | "model cascade", "ensemble" (it is not an ensemble) |
| **BERTopic** | Unsupervised topic model applied to predicted suggestions | "topic model" (acceptable as shorthand after first use) |

---

## Annotation

| Term | Use | Do NOT use |
|------|-----|------------|
| **suggestion** (label = 1) | A sentence that proposes an improvement, gives advice, or recommends an action | "positive", "target class" |
| **non-suggestion** (label = 0) | A sentence that describes, praises, or complains without proposing an action | "negative", "other" |
| **explicit suggestion** | Uses direct modal or imperative language (e.g. "They should add...") | "clear suggestion" |
| **implicit suggestion** | Proposes change indirectly via wishes, desires, or comparisons (e.g. "I wish the pool...") | "indirect suggestion", "hidden suggestion" |
| **comparative suggestion** | Implies improvement by referencing a competitor or expectation | "benchmarking suggestion" |
| **enrichment** / **signal enrichment** | Sampling strategy: ~50% of annotation sentences contain suggestion signals | "oversampling" (different meaning), "biased sampling" |
| **calibration round** | Phase where all 4 annotators label the same 100 sentences for IAA | "pilot round", "practice round" |
| **split annotation** | Phase where each annotator labels a separate batch of sentences | "divided annotation" |

---

## Evaluation

| Term | Use | Do NOT use |
|------|-----|------------|
| **suggestion-class F1** | F1 computed on label = 1 only; the primary metric | "F1" alone (ambiguous — specify class or macro) |
| **macro F1** | F1 averaged across both classes | "average F1", "overall F1" |
| **in-domain evaluation** | Testing on SemEval test set (same domain as training) | "SemEval evaluation" (acceptable but less precise) |
| **cross-domain evaluation** | Testing on MBS test split (different domain from training) | "out-of-domain", "transfer evaluation" |
| **domain gap** / **performance delta** | Difference in F1 between in-domain and cross-domain evaluation | "domain shift" (refers to the cause, not the measured effect) |
| **domain shift** | The linguistic differences between software forums and hotel reviews | "domain mismatch", "distribution shift" |

---

## Architecture & Infrastructure

| Term | Use | Do NOT use |
|------|-----|------------|
| **two-stage fine-tuning** | Training BERT first on SemEval, then on MBS | "transfer learning" alone (too broad), "sequential fine-tuning" |
| **domain adaptation** | The process of adjusting a model to a new target domain via stage-2 fine-tuning | "domain transfer" (acceptable in context) |
| **aspect** | One of six hotel categories: room, food & beverage, service, facilities, location, value | "theme" (reserved for BERTopic topics), "category" |
| **topic** | A BERTopic-discovered cluster of semantically similar sentences | "theme" (acceptable after first use), "aspect" (different concept) |

---

## Usage Rules

1. On first use in any section, write the full term. After that, the short form in the table is acceptable.
2. Always qualify "F1" — write "suggestion-class F1" or "macro F1", never bare "F1".
3. Always qualify "test set" — write "SemEval test set" or "MBS test split", never bare "test set".
4. Always qualify "training data" — write "SemEval training set" or "MBS train split".
5. Hyphenate "stage-1" and "stage-2" when used as modifiers (e.g. "stage-1 model") but not as standalone nouns (e.g. "in stage 1").
6. Use "suggestion" and "non-suggestion" (hyphenated) as class labels, not "positive/negative".
