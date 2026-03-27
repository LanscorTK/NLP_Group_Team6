# Final Report Outline — Suggestion Mining in High-Rated Hotel Reviews

**Target:** 4,000 words max (excluding references, appendices, figure/table captions)
**Format:** LaTeX (ACL template) or Jupyter Notebook PDF
**Rubric weights:** Structure & Format (5%), Methodology & Analysis (30%), Quality of Narrative (10%), Code/Equations/Referencing (5%)

---

## Front Matter (excluded from word count)

- **Title:** e.g. "Mining Actionable Suggestions from High-Rated Hotel Reviews: A Cross-Domain Transfer Learning Approach"
- **Authors:** Chase Sun, Clara Yang, Leah Luo, Xiayi Zhu
- **Member Contribution Table** (REQUIRED by brief — first page)

| Member | Contribution |
|--------|-------------|
| Chase  | Pipeline architecture, preprocessing, BERT fine-tuning, reproducibility |
| Clara  | EDA, report drafting (Methods, Results, Data) |
| Leah   | Annotation guidelines, BERTopic clustering, report (Intro, Related Work, Discussion) |
| Xiayi  | Baselines (regex, TF-IDF+LR), error analysis, manual validation, presentation |

---

## Abstract (~150 words)

**Job:** Standalone summary — question, method, key finding, implication. A reader should know what you did and what you found without reading further.

**Must contain:**
- Research question (one sentence)
- Data source + scale (MBS TripAdvisor, 70k sentences, 4-5 stars)
- Method summary (three-tier: regex → TF-IDF+LR → two-stage BERT; BERTopic clustering)
- Key quantitative finding (best model F1 on MBS test set)
- Main takeaway (domain gap insight + actionable suggestion themes)

**Common mistakes to avoid:**
- Writing the abstract last but rushing it — it's the most-read part
- Including results you don't have yet (use [MISSING] placeholders)
- Vague language ("good results") instead of numbers

**Rubric alignment:** Concise summary (Structure & Format criterion)

---

## 1. Introduction (~500 words)

**Job:** Motivate the research question, establish the gap, state contributions.

**Structure:**
1. **Hook** (2-3 sentences): High-rated hotel reviews are assumed positive, but they contain hidden actionable suggestions. Sentiment analysis misses these.
2. **Problem framing** (3-4 sentences): Suggestion mining as a sentence-level binary classification task. Why it matters for hospitality management.
3. **Gap / challenge** (3-4 sentences): Existing suggestion mining datasets (SemEval-2019) come from software forums — domain transfer to hotel reviews is non-trivial. No prior work applies suggestion mining specifically to luxury hotel reviews.
4. **Research question** (1 sentence, clearly stated): "To what extent can NLP models detect actionable improvement suggestions within high-star hotel reviews?"
5. **Contributions** (bulleted list, 3-4 items):
   - Domain-adapted annotation guidelines for hotel review suggestions
   - Three-tier model comparison quantifying the value of contextual representations
   - Two-stage BERT transfer learning (SemEval → hotel domain) with domain gap analysis
   - BERTopic-based thematic analysis of extracted hotel suggestions

**Evidence required:**
- [ ] Citation for SemEval-2019 Task 9 (Negi et al., 2019)
- [ ] Citation for suggestion mining definition/prior work
- [ ] Brief motivation for why high-rated reviews specifically (not all reviews)

**Common mistakes to avoid:**
- Spending too long on background (save for Related Work)
- Not clearly stating the research question
- Listing methods instead of contributions
- Over-claiming novelty — this is a coursework project, not a NAACL paper

**What would make it stronger:**
- A concrete example sentence from the dataset showing a hidden suggestion in a 5-star review
- Framing domain transfer as a finding, not just a limitation

---

## 2. Related Work (~400 words)

**Job:** Position the project within existing work. Show you understand the field and where your work fits.

**Structure:**
1. **Suggestion mining** (1 paragraph): SemEval-2019 Task 9 (Negi et al., 2019), definition of suggestions, prior approaches (rule-based, ML, BERT), best systems from the shared task.
2. **Domain adaptation in NLP** (1 paragraph): Why models trained on one domain degrade on another. Transfer learning with BERT. Relevant precedents for cross-domain text classification.
3. **Sentiment vs. suggestion** (1 short paragraph): Why suggestion mining is distinct from sentiment analysis — a 5-star review can contain suggestions. This motivates the project gap.

**Evidence required:**
- [ ] SemEval-2019 Task 9 overview paper (Negi et al., 2019)
- [ ] 2-3 papers on suggestion mining or related sentence classification
- [ ] 1-2 papers on domain adaptation / transfer learning in NLP (e.g. ULMFiT, BERT fine-tuning)
- [ ] Brief mention of aspect-based sentiment analysis (to contrast with suggestion mining)

**Common mistakes to avoid:**
- Turning this into a textbook survey — keep it focused on what directly motivates your approach
- Citing papers you haven't read
- Not connecting literature back to your own project decisions

**What would make it stronger:**
- Explicitly stating what gap in the literature your project addresses
- Noting that SemEval data is from software forums and no prior work applies it to hotel reviews

---

## 3. Data (~600 words)

**Job:** Fully describe both datasets, preprocessing, and the annotation process. A reader should be able to reproduce your dataset from this section.

**Structure:**

### 3.1 Marina Bay Sands Review Dataset
- Source: TripAdvisor, Marina Bay Sands, reviews from 2015 onwards
- Raw size: 10,232 reviews → 8,571 after 4-5 star filter + null removal
- Rating split: 2,414 four-star, 6,157 five-star
- Sentence segmentation: spaCy → 70,085 sentences (median 15 tokens, 8.4 sentences/review avg)
- Cleaning: language filter (ASCII ratio), deduplication, length threshold (4-100 tokens)
- **Table 1:** Dataset statistics summary (excluded from word count)

### 3.2 SemEval-2019 Task 9 Data
- Source: Software developer forum posts (Uservoice)
- Size: ~8,500 training + ~800 test sentences
- Label distribution in training set
- **Key domain differences from hotel data** (sentence length, vocabulary, suggestion signal frequencies) — reference the EDA figures

### 3.3 Annotation
- Guidelines: adapted from SemEval definition, with hotel-domain edge cases
- Calibration: 100 sentences × 4 annotators, enriched sampling (50% signal-enriched)
- IAA: Fleiss' kappa = [MISSING — needs calibration results]
- Disagreement resolution process
- Split annotation: [MISSING — needs completion] additional sentences, one batch per annotator
- Final annotated set: [MISSING] sentences, [MISSING]% suggestions
- Train/test split: 80/20 stratified
- **Table 2:** Annotation statistics (calibration kappa, label distribution, split sizes)

**Evidence required:**
- [ ] Dataset statistics: review count, sentence count, rating split, sentence length distribution
- [ ] SemEval training/test sizes and label distribution
- [ ] Domain gap figures from EDA (signal_comparison.png, domain_sentence_lengths.png)
- [ ] Annotation guidelines summary (full guidelines in appendix)
- [ ] Fleiss' kappa from calibration round
- [ ] Final label distribution (suggestion % in annotated set)
- [ ] Train/test split sizes
- [ ] Enrichment strategy justification

**Common mistakes to avoid:**
- Describing every preprocessing step in prose — use a pipeline diagram figure instead
- Not reporting IAA — this is one of the strongest signals of methodological rigour
- Not justifying the enrichment strategy (could look like cherry-picking)
- Forgetting to mention the SemEval data source and how it was used

**What would make it stronger:**
- A pipeline diagram figure showing data flow (raw → filtered → segmented → annotated → train/test)
- Explicit comparison table: MBS vs. SemEval key statistics side-by-side
- Reporting exact sentence counts at each preprocessing step (shows transparency)

---

## 4. Methods (~700 words)

**Job:** Describe the models and experimental setup clearly enough for replication. Justify choices.

**Structure:**

### 4.1 Task Formulation
- Binary sentence classification: suggestion (1) vs non-suggestion (0)
- Input: individual sentence (no review-level context)
- Why sentence-level: finer granularity, consistent with SemEval task definition

### 4.2 Tier 1: Rule-Based Baseline (Regex)
- Pattern lists: modal verbs (should, could, would + improvement verbs), imperative forms (please, recommend, consider), conditional patterns (I wish, it would be nice if)
- Classification rule: any pattern match → suggestion
- Purpose: establishes a floor and demonstrates limitations of surface patterns (e.g. "I would stay again" = false positive)

### 4.3 Tier 2: TF-IDF + Logistic Regression
- TF-IDF vectoriser: unigrams + bigrams, max 10k features, min_df=2
- Trained on SemEval training data
- class_weight='balanced' for imbalance handling
- Purpose: shows value of learned features over hand-crafted rules; but bag-of-words can't capture context

### 4.4 Tier 3: BERT Classifier (Two-Stage Fine-Tuning)
- Base model: bert-base-uncased (110M params)
- **Stage 1:** Fine-tune on SemEval training data (~7k sentences)
  - Hyperparameters: lr=2e-5, batch=16, epochs=3-4, warmup=10%, weight_decay=0.01
  - Hardware: Apple Silicon MPS backend
- **Stage 2:** Further fine-tune on MBS annotated training data (~240-400 sentences)
  - Lower lr (1e-5), fewer epochs (2-3) to avoid overfitting
  - Purpose: domain adaptation — quantify performance gain from hotel-specific training data
- Justify hyperparameter choices (cite original BERT paper)

### 4.5 Insight Generation: BERTopic Clustering
- Applied to all sentences predicted as suggestions by best model
- Embeddings: all-MiniLM-L6-v2
- Clustering: UMAP + HDBSCAN (via BERTopic)
- Supplementary: aspect-based grouping (room, food, service, facilities, location, value)
- Purpose: transform binary predictions into actionable thematic insights

### 4.6 Evaluation Metrics
- Primary: Precision, Recall, F1 (suggestion class)
- Secondary: Macro F1, confusion matrix
- Cross-domain comparison: all models evaluated on both SemEval test and MBS test
- Manual validation: 100 predicted suggestions + 100 predicted non-suggestions from full MBS corpus

**Evidence required:**
- [ ] Regex pattern list (exact patterns used)
- [ ] TF-IDF + LR hyperparameters
- [ ] BERT training config (lr, batch size, epochs, device, seed)
- [ ] Stage-1 validation metrics during training
- [ ] Stage-2 training config (lower lr, fewer epochs)
- [ ] BERTopic configuration (min_topic_size, embedding model)
- [ ] Aspect keyword lists for aspect-based grouping
- [ ] Justification for each model choice (why three tiers, why two-stage BERT)

**Common mistakes to avoid:**
- Writing a code tutorial instead of a methods description
- Not justifying hyperparameter choices
- Forgetting to mention the evaluation protocol (what's the test set, how many runs)
- Describing BERTopic as a model rather than a post-hoc analysis tool

**What would make it stronger:**
- A figure showing the three-tier architecture with data flow arrows
- Connecting each tier to a specific research question (rules vs. features vs. context)
- Mentioning what you did NOT do and why (e.g. no GPT-based models — compute/cost constraints)
- Noting MPS-specific considerations for reproducibility

---

## 5. Results (~600 words)

**Job:** Present findings clearly with tables and figures. Let the numbers speak — interpretation goes in Discussion.

**Structure:**

### 5.1 Model Comparison
- **Table 3:** Main results table (the centrepiece of the paper)

| Model | SemEval Test F1 | MBS Test F1 | Delta |
|-------|----------------|-------------|-------|
| Regex baseline | [MISSING] | [MISSING] | [MISSING] |
| TF-IDF + LR | [MISSING] | [MISSING] | [MISSING] |
| BERT Stage 1 (SemEval only) | [MISSING] | [MISSING] | [MISSING] |
| BERT Stage 2 (SemEval + MBS) | [MISSING] | [MISSING] | [MISSING] |

- Key observations: (1) each tier improves over the last, (2) the SemEval→MBS delta quantifies domain gap, (3) stage-2 adaptation reduces the gap
- Confusion matrices for each model on MBS test set (Figure)

### 5.2 Domain Transfer Analysis
- Compare in-domain (SemEval test) vs. cross-domain (MBS test) performance for each model
- Which model is most robust to domain shift? Why?
- What the F1 delta tells us about the nature of the domain gap

### 5.3 Suggestion Themes (BERTopic)
- Number of topics found, outlier rate
- Top 5-10 topics with representative sentences
- Aspect distribution: room (22%), facilities (14%), food (10%), service (7%), etc.
- **Table 4:** Topic summary with counts and example sentences
- **Figure:** Topic bar chart and/or aspect distribution pie chart

### 5.4 Manual Validation (if completed)
- Precision/recall estimated from 200 manual checks on full corpus predictions
- Discrepancies between test-set metrics and manual validation

**Evidence required:**
- [ ] F1 scores for all 4 models on both test sets (SemEval + MBS)
- [ ] Precision and Recall for all models (not just F1)
- [ ] Confusion matrices (4 models × 2 test sets = up to 8, but focus on MBS)
- [ ] PR curves (already generated: pr_curve_baselines_semeval.png)
- [ ] BERTopic results: topic count, topic names, top words, representative sentences
- [ ] Aspect distribution breakdown
- [ ] Manual validation results (100+100 predictions)
- [ ] BERT stage-1 vs stage-2 comparison on MBS test

**Common mistakes to avoid:**
- Reporting only accuracy (misleading with class imbalance)
- Not reporting both precision and recall alongside F1
- Interpreting results here instead of in Discussion
- Presenting BERTopic results without explaining what actionable insight they provide

**What would make it stronger:**
- The delta column in the results table — makes the domain gap story immediately visible
- Showing specific sentences where BERT succeeds and regex/TF-IDF fail
- Linking BERTopic themes back to the original motivation (what can hotel management learn?)

---

## 6. Error Analysis & Discussion (~700 words)

**Job:** This is where you demonstrate critical thinking. Analyse WHY things worked or didn't. This is the highest-value section for marks (Methodology & Analysis + Quality of Narrative criteria).

**Structure:**

### 6.1 Error Analysis
- Categorise errors from MBS test set by type:
  - **False negatives (missed suggestions):** explicit, implicit, comparative
  - **False positives (over-predicted):** disguised complaints, positive modals, generic praise
- Show 3-5 concrete error examples with analysis of why the model failed
- Which suggestion types are hardest? (likely implicit suggestions and comparisons)

### 6.2 Domain Gap Discussion
- What specific linguistic differences cause cross-domain degradation?
- Vocabulary mismatch (SemEval: "API", "SDK"; MBS: "breakfast", "concierge")
- Suggestion expression style: software forums are more explicit; hotel reviews are more implicit
- How much did stage-2 fine-tuning close the gap? What remains?

### 6.3 Limitations
- **Annotation scale:** ~300-500 sentences is small; test set volatility
- **Single hotel:** MBS only — generalisability to other hotels/hospitality segments unknown
- **Annotation subjectivity:** complaint vs. suggestion boundary is inherently fuzzy
- **No human upper bound:** didn't compute human-model agreement on test set
- **BERTopic on stage-1 predictions:** if stage-2 model not available, topic analysis uses cross-domain predictions (noisier)
- **Sentence isolation:** labelling without review context may miss suggestions that span multiple sentences

### 6.4 What We Would Do Differently
- More annotation data (target 1,000+ sentences)
- Multiple hotels for cross-property generalisation
- Investigate few-shot / prompt-based approaches (GPT-style) as alternative to fine-tuning
- Sentence-pair context (previous + current sentence) for implicit suggestions

**Evidence required:**
- [ ] Error examples: 3-5 false positives and 3-5 false negatives with analysis
- [ ] Error type distribution (how many of each category)
- [ ] Specific vocabulary/pattern differences causing cross-domain errors
- [ ] IAA disagreement examples (what was hardest to annotate)
- [ ] Honest assessment of what didn't work or was attempted and abandoned

**Common mistakes to avoid:**
- Listing limitations as a boring checklist without analysis
- Not connecting error analysis back to model design choices
- Being defensive about weaknesses instead of analytically honest
- Ignoring the domain gap as "just a limitation" instead of a central finding

**What would make it stronger:**
- Framing the domain gap as the project's most interesting finding, not just a problem
- Concrete actionable recommendations for each limitation
- Connecting error types to annotation guideline edge cases (shows coherence)
- Distinguishing between limitations of the approach vs. limitations of the execution (time/resources)

---

## 7. Conclusion & Future Work (~250 words)

**Job:** Directly answer the research question. Summarise contributions. Point forward.

**Structure:**
1. **Answer the research question** (2-3 sentences): To what extent CAN NLP models detect suggestions? Answer with evidence — e.g. "BERT with two-stage fine-tuning achieves F1 of X on hotel review data, demonstrating that [X]"
2. **Key contributions** (3-4 bullet points, same as Intro but now with evidence)
3. **Future work** (3-4 concrete, specific suggestions):
   - Scale annotation to more hotels and review platforms
   - Investigate prompt-based suggestion detection (LLM zero-shot/few-shot)
   - Multi-label classification (suggestion type: explicit/implicit/comparative)
   - Real-time suggestion monitoring dashboard for hotel management

**Evidence required:**
- [ ] Best F1 score to cite in the conclusion
- [ ] Whether the answer to the RQ is "yes, to X extent" or "partially, because Y"

**Common mistakes to avoid:**
- Introducing new information not discussed earlier
- Over-claiming ("our model solves suggestion mining")
- Generic future work ("use more data") — be specific

**What would make it stronger:**
- The conclusion should be a satisfying narrative closure — the reader should feel the question was honestly addressed
- Future work should be feasible and specific, not hand-wavy

---

## References (excluded from word count)

- Use Harvard or Vancouver style consistently
- Must include: SemEval-2019 Task 9, BERT paper, BERTopic paper, spaCy, scikit-learn, TripAdvisor data source
- Must acknowledge AI tools (Claude — assistive use for coding/planning)
- Cite any pre-trained models used (bert-base-uncased, all-MiniLM-L6-v2)

---

## Appendices (excluded from word count)

- **Appendix A:** Full annotation guidelines (from docs/annotation_guidelines.md)
- **Appendix B:** Additional figures (EDA plots not included in main body)
- **Appendix C:** Hyperparameter details / training curves (if space permits)
- **Appendix D:** AI tool acknowledgement statement

---

## Word Budget Summary

| Section | Target Words | % of Total |
|---------|-------------|------------|
| Abstract | 150 | 3.8% |
| 1. Introduction | 500 | 12.5% |
| 2. Related Work | 400 | 10.0% |
| 3. Data | 600 | 15.0% |
| 4. Methods | 700 | 17.5% |
| 5. Results | 600 | 15.0% |
| 6. Error Analysis & Discussion | 700 | 17.5% |
| 7. Conclusion & Future Work | 250 | 6.3% |
| Buffer | 100 | 2.5% |
| **Total** | **4,000** | **100%** |

Tables, figures, captions, references, and appendices are all EXCLUDED from the word count. Use them aggressively to carry information.
