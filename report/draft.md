# Mining Actionable Suggestions from High-Rated Hotel Reviews: A Cross-Domain Transfer Learning Approach

**Authors:** Chase Sun, Clara Yang, Leah Luo, Xiayi Zhu

| Member | Contribution |
|--------|-------------|
| Chase  | Pipeline architecture, preprocessing, BERT fine-tuning, reproducibility |
| Clara  | EDA, report drafting (Methods, Results, Data) |
| Leah   | Annotation guidelines, BERTopic clustering, report (Intro, Related Work, Discussion) |
| Xiayi  | Baselines (regex, TF-IDF+LR), error analysis, manual validation, presentation |

---

## Abstract (~150 words)

[MISSING — requires final quantitative results from MBS evaluation]

To what extent can NLP models detect actionable improvement suggestions within high-star hotel reviews? We address this question by applying suggestion mining — a sentence-level binary classification task — to 70,085 sentences extracted from 8,571 four- and five-star TripAdvisor reviews of Marina Bay Sands, Singapore. We compare three model tiers of increasing complexity: a regex baseline using hand-crafted patterns, a TF-IDF with logistic regression classifier, and a BERT model with two-stage fine-tuning. All models are first trained or evaluated on the SemEval-2019 Task 9 dataset from software forums, then tested on human-annotated hotel review sentences to measure cross-domain transfer. We further apply BERTopic clustering to characterise the thematic content of predicted suggestions. [MISSING — key quantitative finding: best model suggestion-class F1 on MBS test split]. Our results demonstrate [MISSING — main takeaway about domain gap and actionable themes].

---

## 1. Introduction (~500 words)

Hotel reviews on platforms such as TripAdvisor are widely used by travellers and hotel operators alike. Reviews with four or five stars are generally assumed to express satisfaction, yet they frequently contain actionable suggestions for improvement embedded within otherwise positive text. A guest who rates a hotel five stars may still write, "I wish the breakfast had more vegetarian options." Standard sentiment analysis classifies such reviews as positive, missing the actionable content within them.

Suggestion mining is the task of identifying sentences that propose an improvement, give advice, or recommend an action (Negi, de Rijke and Buitelaar, 2018). Framed as sentence-level binary classification, it distinguishes suggestions from non-suggestions. The task was formalised in the SemEval-2019 Task 9 shared task (Negi, Daudert and Buitelaar, 2019), which provided labelled data from software developer forums. That shared task demonstrated that BERT-based models achieved the best performance for in-domain suggestion detection but also revealed that cross-domain transfer — applying a model trained on one domain to another — remains a challenge (Park et al., 2019).

No prior work, to our knowledge, applies suggestion mining specifically to high-rated hotel reviews. This creates a gap: hotel reviews differ from software forums in vocabulary, sentence structure, and the way suggestions are expressed. Software forum users write explicit feature requests ("The API should support batch uploads"), while hotel guests embed suggestions within narrative descriptions ("The room was lovely but the pillows could be firmer"). This domain shift makes direct transfer from existing datasets non-trivial, and raises the question of how much domain-specific adaptation is needed.

Our research question is: **To what extent can NLP models detect actionable improvement suggestions within high-star hotel reviews?**

We address this question through four contributions:

- **Domain-adapted annotation guidelines** for identifying suggestions in hotel reviews, with inter-annotator agreement measured on a calibration sample of 100 sentences.
- **A three-tier model comparison** — regex baseline, TF-IDF with logistic regression (TF-IDF + LR) baseline, and BERT classifier — that quantifies the value of increasingly rich representations for suggestion detection.
- **Two-stage fine-tuning** of BERT: first on the SemEval-2019 training data, then on annotated hotel review sentences, with a cross-domain evaluation that measures the performance delta between in-domain and cross-domain settings.
- **BERTopic-based thematic analysis** of extracted suggestions, grouping them into actionable categories (room, food and beverage, service, facilities, location, value) relevant to hotel management.

---

## 2. Related Work (~400 words)

### Suggestion mining

Negi, de Rijke and Buitelaar (2018) formalised suggestion mining as the task of identifying sentences that express a suggestion, defined as a sentence that proposes an improvement, gives advice, or recommends a course of action. Earlier approaches relied on hand-crafted rules targeting modal verbs and imperative constructions (Negi and Buitelaar, 2017). The SemEval-2019 Task 9 shared task (Negi, Daudert and Buitelaar, 2019) established benchmark datasets and attracted 33 participating teams. The best-performing systems used BERT-based architectures, with the top system achieving 77.78% suggestion-class F1 on Subtask A (in-domain). Subtask B required cross-domain generalisation from software forums to hotel reviews, where performance dropped for many systems, indicating that domain shift is a core challenge in suggestion mining.

### Domain adaptation and BERT fine-tuning

Pre-trained language models such as BERT (Devlin et al., 2019) learn general linguistic representations that can be adapted to downstream tasks via fine-tuning. However, fine-tuned models can be brittle when the target domain differs from the training domain. Park et al. (2019) showed that BERT is unstable for out-of-domain suggestion mining, with high variance across runs on cross-domain data. Sai Prasanna and Seelan (2019) addressed this by applying tri-training, a semi-supervised bootstrapping approach, to label target-domain data without manual annotation, achieving 81.94% suggestion-class F1 on the cross-domain subtask. These findings motivate our two-stage fine-tuning strategy: first training on the larger SemEval dataset for task knowledge, then adapting to hotel-domain data with a smaller annotated sample.

### Sentiment analysis versus suggestion mining

Suggestion mining is distinct from sentiment analysis. A five-star review is positive in sentiment but may contain sentences that propose improvements. Aspect-based sentiment analysis identifies opinions about specific features (e.g. "the pool was clean"), but does not distinguish between descriptive praise and actionable suggestions (e.g. "they should heat the pool"). Our work targets this gap: extracting the subset of sentences that contain actionable content, regardless of overall review sentiment, and grouping them thematically using BERTopic (Grootendorst, 2022) to produce insights relevant to hotel management.

---

## 3. Data (~600 words)

### 3.1 Marina Bay Sands Review Dataset

We collected 10,232 English-language reviews of Marina Bay Sands, Singapore from TripAdvisor, spanning January 2015 to the date of collection. We filtered for four-star and five-star reviews to focus on high-rated reviews that are generally positive but may contain embedded suggestions. After removing null entries, 8,571 reviews remained: 2,414 four-star and 6,157 five-star.

We segmented reviews into sentences using the spaCy `en_core_web_sm` model (Honnibal et al., 2020). This produced 70,085 sentences, with a median length of 15 tokens and a mean of 8.4 sentences per review. We applied three cleaning steps: (1) a language filter removing sentences with less than 50% ASCII characters, (2) exact deduplication, and (3) a length filter retaining sentences of 4–100 tokens. Table 1 summarises the dataset statistics.

**Table 1: MBS dataset statistics**

| Statistic | Value |
|-----------|-------|
| Raw reviews | 10,232 |
| After 4–5 star filter | 8,571 |
| Four-star reviews | 2,414 |
| Five-star reviews | 6,157 |
| Sentences after segmentation | 70,085 |
| Median sentence length (tokens) | 15 |
| Mean sentences per review | 8.4 |

### 3.2 SemEval-2019 Task 9 Dataset

We use the publicly available SemEval-2019 Task 9 Subtask A dataset (Negi, Daudert and Buitelaar, 2019), collected from the Uservoice software feedback forum. The training set contains 8,500 sentences (2,085 suggestions, 6,415 non-suggestions; 24.5% suggestion rate). The test set contains 833 sentences (87 suggestions, 746 non-suggestions; 10.4% suggestion rate). This class imbalance, particularly in the test set, motivates our use of suggestion-class F1 rather than accuracy as the primary evaluation metric.

The SemEval dataset differs from our MBS sentences in several respects. Software forum posts tend to be explicit feature requests using modal verbs ("The system should allow..."), while hotel review suggestions are often implicit, embedded within descriptive or comparative language ("I wish the breakfast had..."). Sentence lengths and vocabulary also differ: the SemEval data contains technical terms absent from hospitality reviews. These differences constitute the domain shift that our cross-domain evaluation measures.

**Table 2: SemEval-2019 Task 9 dataset statistics**

| Split | Total | Suggestions | Non-suggestions | Suggestion rate |
|-------|-------|-------------|-----------------|----------------|
| Training | 8,500 | 2,085 | 6,415 | 24.5% |
| Test | 833 | 87 | 746 | 10.4% |

### 3.3 Annotation

[MISSING — blocked on annotation completion. Section skeleton below.]

We developed annotation guidelines adapted from the SemEval-2019 task definition (Negi, Daudert and Buitelaar, 2019; Negi, de Rijke and Buitelaar, 2018), with extensions for hotel-domain edge cases such as comparative suggestions and implicit improvement requests.

**Calibration round.** All four annotators independently labelled the same MBS calibration sample of 100 sentences. To increase the proportion of informative examples, we used signal enrichment: 50% of the sample was drawn from sentences containing suggestion-signal keywords (modal verbs, imperatives, conditionals), and 50% was drawn randomly. Fleiss' kappa = [MISSING]. Pairwise Cohen's kappa: [MISSING].

**Disagreement resolution.** Sentences with majority agreement (3 or 4 of 4 annotators) received the majority label. Ties (2-vs-2) were flagged for discussion and resolved by group consensus. [MISSING — number of ties and resolution outcomes.]

**Split annotation.** After calibration, the remaining annotation was divided among annotators, with each annotator labelling a separate batch. [MISSING — total annotated sentences, label distribution.]

**Train/test split.** The MBS annotated set was split 80/20 (stratified by label) into MBS train split and MBS test split. [MISSING — exact split sizes.]

---

## 4. Methods (~700 words)

### 4.1 Task formulation

We frame suggestion mining as binary sentence classification: each sentence is labelled as suggestion (1) or non-suggestion (0). Classification operates at the sentence level, consistent with the SemEval-2019 task definition. We do not use review-level context; each sentence is classified independently.

### 4.2 Tier 1: Regex baseline

The regex baseline classifies a sentence as a suggestion if it matches any of 18 hand-crafted regular expression patterns. These patterns cover three categories: modal suggestions (e.g. `\bshould\b`, `\bcould improve\b`, `\bneed to\b`), imperatives (e.g. `\bplease\b`, `\brecommend\b`, `\bconsider\b`), and conditionals (e.g. `\bi wish\b`, `\bit would be nice\b`, `\bhopefully\b`). Matching is case-insensitive. This baseline establishes a performance floor and demonstrates the limitations of surface patterns — for instance, "I would definitely stay again" matches the modal pattern `\bwould\b` but is not a suggestion.

### 4.3 Tier 2: TF-IDF + LR baseline

The TF-IDF + LR baseline uses a TF-IDF vectoriser (unigrams and bigrams, maximum 10,000 features, minimum document frequency 2, sublinear term-frequency scaling) followed by logistic regression (regularisation C=1.0, balanced class weights, maximum 1,000 iterations). The model is trained on the SemEval training set. Balanced class weights address the class imbalance by upweighting the minority suggestion class. This baseline shows the value of learned features over hand-crafted rules while remaining interpretable and fast to train.

### 4.4 Tier 3: BERT classifier with two-stage fine-tuning

Our primary model is bert-base-uncased (Devlin et al., 2019), a 110-million-parameter pre-trained transformer, fine-tuned for binary classification with an additional linear output layer.

**Stage 1: SemEval fine-tuning.** We fine-tune on the SemEval training set (8,500 sentences), using a 90/10 stratified train/validation split. Sentences are tokenised with a maximum length of 128 tokens. Training uses the AdamW optimiser with learning rate 2×10⁻⁵, batch size 16, 3 epochs, linear warmup over 10% of training steps, and weight decay 0.01. We select the checkpoint with the best macro F1 on the validation split. On the SemEval validation split, BERT stage-1 achieves macro F1 of 0.886 and suggestion-class F1 of 0.828.

**Stage 2: MBS domain adaptation.** Starting from the stage-1 checkpoint, we further fine-tune on the MBS train split, using an 85/15 stratified train/validation split. The learning rate is reduced to 1×10⁻⁵ to avoid overwriting task knowledge learned in stage 1. All other hyperparameters remain the same. This two-stage approach follows the principle of progressive fine-tuning: pre-trained representations → task-specific fine-tuning → domain-specific adaptation.

All training was performed on Apple Silicon (MPS backend) with a fixed random seed of 42 for reproducibility. Hyperparameters follow the ranges recommended by Devlin et al. (2019).

### 4.5 Insight generation: BERTopic clustering

To transform binary predictions into actionable thematic insights, we apply BERTopic (Grootendorst, 2022) to all sentences predicted as suggestions by the best-performing model. Sentence embeddings are computed using all-MiniLM-L6-v2 (Reimers and Gurevych, 2019). Dimensionality reduction uses UMAP (5 components, 15 neighbours, cosine metric, minimum distance 0.0). Clustering uses HDBSCAN (minimum cluster size 5). Topic representations are extracted via a class-based TF-IDF procedure. The number of topics is determined automatically.

As a complementary analysis, we apply keyword-based aspect grouping to the same set of predicted suggestions. Six hotel-relevant aspects are defined: room, food and beverage, service, facilities, location, and value. Each sentence is assigned to the aspect whose keywords appear most frequently, or to "other" if no keywords match. This provides a coarser but more interpretable categorisation than BERTopic.

### 4.6 Evaluation

Our primary metric is suggestion-class F1, computed on label 1 only. We also report precision, recall, and macro F1. All models are evaluated on both the SemEval test set (in-domain evaluation) and the MBS test split (cross-domain evaluation). The difference between in-domain and cross-domain suggestion-class F1 — the performance delta — quantifies the domain gap. [MISSING — manual validation protocol: 100 predicted suggestions + 100 predicted non-suggestions from full MBS sentences, manually checked.]

---

## 5. Results (~600 words)

[MISSING — blocked on MBS annotation and evaluation. Section skeleton below.]

### 5.1 Model comparison

**Table 3: Suggestion-class F1 across models and evaluation settings**

| Model | SemEval test F1 | MBS test F1 | Delta |
|-------|----------------|-------------|-------|
| Regex baseline | [MISSING] | [MISSING] | [MISSING] |
| TF-IDF + LR baseline | [MISSING] | [MISSING] | [MISSING] |
| BERT stage-1 (SemEval only) | [MISSING] | [MISSING] | [MISSING] |
| BERT stage-2 (SemEval + MBS) | N/A | [MISSING] | [MISSING] |

Key observations to report:
1. Each tier improves over the previous on in-domain data.
2. The SemEval → MBS delta quantifies the domain gap for each model.
3. BERT stage-2 reduces the domain gap through hotel-domain fine-tuning.

[MISSING — confusion matrices for each model on MBS test split.]

### 5.2 Domain transfer analysis

[MISSING — compare in-domain vs. cross-domain performance; analyse which model is most robust to domain shift; discuss what the F1 delta reveals about the nature of the domain gap.]

### 5.3 Suggestion themes (BERTopic)

BERTopic analysis of [MISSING — number] predicted suggestions from BERT stage-1 identified 16 topics, with 1,184 sentences in the largest cluster and topics 1–15 ranging from 5 to 43 sentences each.

**Table 4: Aspect distribution of predicted suggestions**

| Aspect | Count | Percentage |
|--------|-------|-----------|
| Other | 610 | 35.1% |
| Room | 384 | 22.1% |
| Facilities | 242 | 13.9% |
| Food & beverage | 175 | 10.1% |
| Service | 125 | 7.2% |
| Value | 120 | 6.9% |
| Location | 82 | 4.7% |

Room-related suggestions dominate (22.1%), followed by facilities (13.9%) and food and beverage (10.1%). The "other" category (35.1%) captures suggestions that do not match predefined aspect keywords, indicating either novel themes or multi-aspect sentences.

[NEEDS EVIDENCE — representative example sentences for each aspect; BERTopic topic details with top words.]

### 5.4 Manual validation

[MISSING — requires manual review of 200 predictions from full MBS sentences.]

---

## 6. Error Analysis and Discussion (~700 words)

[MISSING — blocked on MBS evaluation. Section skeleton below.]

### 6.1 Error analysis

[MISSING — categorise errors from MBS test split: false negatives (explicit, implicit, comparative suggestions missed) and false positives (disguised complaints, positive modals, generic praise). Show 3–5 concrete examples.]

### 6.2 Domain gap discussion

[MISSING — analyse specific linguistic differences causing cross-domain degradation. Expected: vocabulary mismatch (SemEval technical terms vs. hospitality terms), suggestion expression style (explicit vs. implicit), sentence length differences. Report how much BERT stage-2 closes the gap.]

### 6.3 Limitations

- **Annotation scale.** The MBS annotated set contains [MISSING] sentences. Test-set metrics may be volatile with a small test split.
- **Single hotel.** All MBS data comes from one property. Generalisability to other hotels or hospitality segments is unknown.
- **Annotation subjectivity.** The boundary between complaints and suggestions is inherently fuzzy. Different annotators may disagree on implicit suggestions.
- **No human upper bound.** We did not compute human–model agreement on the test split.
- **Sentence isolation.** Each sentence is classified independently, without review-level context. Suggestions that span multiple sentences may be missed.
- **BERTopic on stage-1 predictions.** [VERIFY — if stage-2 model was used for topic analysis, update this.] Topic analysis uses cross-domain predictions from BERT stage-1, which may introduce noise from false positives.

### 6.4 What we would do differently

- Annotate 1,000+ sentences to reduce test-set variance.
- Include multiple hotels for cross-property generalisation.
- Investigate few-shot or prompt-based approaches as alternatives to fine-tuning.
- Use sentence-pair context (previous + current sentence) to detect implicit suggestions.

---

## 7. Conclusion (~250 words)

[MISSING — blocked on final results. Section skeleton below.]

To what extent can NLP models detect actionable improvement suggestions within high-star hotel reviews? Our experiments show that [MISSING — answer with evidence].

Our contributions are:
1. Domain-adapted annotation guidelines for hotel review suggestions, with Fleiss' kappa of [MISSING] on a 100-sentence calibration sample.
2. A three-tier model comparison demonstrating that [MISSING — e.g. contextual representations (BERT) substantially outperform surface patterns (regex) and bag-of-words features (TF-IDF + LR)].
3. Two-stage fine-tuning that reduces the cross-domain performance delta by [MISSING] F1 points, from [MISSING] to [MISSING].
4. Thematic analysis revealing that room-related suggestions account for the largest share (22.1%) of predicted suggestions, followed by facilities (13.9%) and food and beverage (10.1%).

Future work should scale annotation to more hotels and review platforms, investigate prompt-based suggestion detection using large language models in zero-shot and few-shot settings, explore multi-label classification of suggestion types (explicit, implicit, comparative), and develop real-time suggestion monitoring tools for hotel management.

---

## References

Devlin, J., Chang, M.-W., Lee, K. and Toutanova, K. (2019) 'BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding', *Proceedings of NAACL-HLT 2019*, Minneapolis, Minnesota, pp. 4171–4186.

Grootendorst, M. (2022) 'BERTopic: Neural topic modeling with a class-based TF-IDF procedure', arXiv preprint arXiv:2203.05794.

Honnibal, M., Montani, I., Van Landeghem, S. and Boyd, A. (2020) 'spaCy: Industrial-strength Natural Language Processing in Python', DOI: 10.5281/zenodo.1212303. [VERIFY exact year and DOI]

Negi, S. and Buitelaar, P. (2017) 'Inducing Distant Supervision in Suggestion Mining through Part-of-Speech Embeddings', arXiv preprint arXiv:1709.07403.

Negi, S., Daudert, T. and Buitelaar, P. (2019) 'SemEval-2019 Task 9: Suggestion Mining from Online Reviews and Forums', *Proceedings of the 13th International Workshop on Semantic Evaluation (SemEval-2019)*, Minneapolis, Minnesota, pp. 1267–1274.

Negi, S., de Rijke, M. and Buitelaar, P. (2018) 'Open Domain Suggestion Mining: Problem Definition and Datasets', arXiv preprint arXiv:1806.02179.

Park, C., Kim, J., Lee, H.-G., Amplayo, R.K., Kim, H., Seo, J. and Lee, C. (2019) 'ThisIsCompetition at SemEval-2019 Task 9: BERT is unstable for out-of-domain samples', *Proceedings of SemEval-2019*, Minneapolis, Minnesota.

Pedregosa, F. et al. (2011) 'Scikit-learn: Machine Learning in Python', *Journal of Machine Learning Research*, 12, pp. 2825–2830. [VERIFY author list]

Reimers, N. and Gurevych, I. (2019) 'Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks', *Proceedings of EMNLP-IJCNLP 2019*, Hong Kong, pp. 3982–3992.

Sai Prasanna and Sri Ananda Seelan (2019) 'Zoho at SemEval-2019 Task 9: Semi-supervised Domain Adaptation using Tri-training for Suggestion Mining', *Proceedings of SemEval-2019*, Minneapolis, Minnesota.

Wolf, T. et al. (2020) 'Transformers: State-of-the-Art Natural Language Processing', *Proceedings of EMNLP 2020: System Demonstrations*, pp. 38–45. [VERIFY]
