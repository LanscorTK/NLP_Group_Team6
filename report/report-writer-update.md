# Report Writer Update — 2026-03-19 (Session 6)

## Session 6 Summary

Addressed remaining proofreading review issues within word budget (3,996/4,000).

### Changes made
1. **Statistical fragility caveat (M3):** Added bootstrap CI caveat to limitations; softened "doubling" to "approximately doubling" in abstract, results, and conclusion
2. **Figure caption fix (S2):** Updated model_comparison figure caption to explicitly state it shows macro F1 (not suggestion-class F1) with cross-reference to Table 1
3. **Catastrophic forgetting note (M4):** Added new limitation bullet acknowledging stage-2 was not evaluated on SemEval test set
4. **Thematic analysis caveat (N2):** Strengthened conclusion caveat with specific precision-based false positive estimate (44%)
5. **Tense consistency (S4):** Methods section converted to past tense throughout
6. **Varied examples (S5):** Replaced repeated "I would definitely stay again" in Methods with different example
7. **Reduced "actionable" overuse (S6):** Replaced 3 instances with "concrete", "constructive", "improvement-oriented"
8. **SemEval recall added to Table 3 (S3):** Added R column for SemEval side (computed from P and F1)
9. **Word budget trimmed:** Cut ~50 words across abstract, intro, methods, related work, conclusion, discussion
10. **Engineer requests filed:** Added M4/S2/M7 requests to `comments/engineer_update.md`

### Proofreading review triage
Many issues from `comments/proofreading_review.md` were already resolved before this session:
- M1 (BERTopic on stage-1) — already fixed by pipeline re-run on stage-2
- N1 (novelty claim) — already scoped in introduction
- C1-C5 (citations) — already addressed
- M5, M6, M8, M9 — already in text

### Deferred to Engineer
- M4 number: BERT stage-2 SemEval test F1 (catastrophic forgetting check)
- S2 figure: Regenerate model_comparison.png with suggestion-class F1
- M7: Per-step cleaning removal counts

---

# Report Writer Update — 2026-03-19 (Session 5)

## 1. What Was Finished

### Previous sessions (1-4)
- Full workspace audit: docs, artifacts, figures, CSVs, SemEval data
- Evidence checklist created: 34/65 items available, 31 missing (blocked on annotation)
- 7-section report outline confirmed and rubric-aligned
- Glossary populated (32 canonical terms, 6 usage rules)
- arXiv + Semantic Scholar MCP servers installed
- Literature search completed — all key citations verified
- 7 reference summaries created in `references/summaries/`
- Switched report format to LaTeX (ACL 2023 template)
- Sections 1–4 fully drafted, Sections 5–7 + Abstract as skeletons with [MISSING] placeholders
- Pipeline diagram created (B16)
- VSCode config created

### This session (Session 5)
- **Merged main branch** into report-writing branch — brought in all annotation data, model results, error analysis code, and new figures
- **Filled all [MISSING] placeholders** across 7 .tex files:
  - **Section 3.3 (Annotation):** Fleiss' κ = 0.379, pairwise Cohen's κ (0.212–0.672), 11 ties resolved, 400 total annotated, train/test split (320/80)
  - **Section 5 (Results):** Complete results table (4 models × 2 datasets), domain transfer analysis, BERTopic section updated with 1,738 predictions
  - **Section 6 (Error Analysis & Discussion):** Error categorisation (7 FP, 6 FN for BERT stage-2), domain gap discussion, updated limitations
  - **Section 7 (Conclusion):** Research question answered with evidence, all 4 contributions filled with numbers
  - **Abstract:** Key finding (F1 0.581), main takeaway about domain adaptation
  - **Appendix:** AI tool acknowledgement and annotation guidelines summary written
  - **Methods:** Removed manual validation placeholder, added note about limitation
- **Model comparison figure inserted** (Figure 1 in Results section)
- **Evidence checklist updated:** 66/68 items now verified (was 42/65)
- **LaTeX compiles cleanly** — 8 pages, no errors, no remaining [MISSING]/[VERIFY]/[TODO]
- **Word count:** 3,403 words (under 4,000 limit)

## 2. Metrics Extracted (from confusion matrix figures)

### Suggestion-class F1

| Model | SemEval | MBS | Δ |
|-------|---------|-----|---|
| Regex | 0.386 | 0.471 | +0.085 |
| TF-IDF + LR | 0.595 | 0.190 | −0.405 |
| BERT stage-1 | 0.716 | 0.286 | −0.430 |
| BERT stage-2 | — | 0.581 | — |

### Confusion Matrices (MBS Test, 80 sentences)

| Model | TN | FP | FN | TP |
|-------|----|----|----|----|
| Regex | 54 | 11 | 7 | 8 |
| TF-IDF + LR | 61 | 4 | 13 | 2 |
| BERT stage-1 | 62 | 3 | 12 | 3 |
| BERT stage-2 | 58 | 7 | 6 | 9 |

## 3. Files Changed

| File | Action | Session |
|------|--------|---------|
| `report/sections/abstract.tex` | **Filled all [MISSING]** | Session 5 |
| `report/sections/data.tex` | **Filled Section 3.3 annotation** | Session 5 |
| `report/sections/methods.tex` | **Removed manual validation [MISSING]** | Session 5 |
| `report/sections/results.tex` | **Filled all [MISSING], added figure** | Session 5 |
| `report/sections/discussion.tex` | **Filled all [MISSING]** | Session 5 |
| `report/sections/conclusion.tex` | **Filled all [MISSING]** | Session 5 |
| `report/sections/appendix.tex` | **Written (guidelines + AI ack)** | Session 5 |
| `report/main.tex` | **Updated graphicspath** | Session 5 |
| `report/evidence_checklist.md` | **Updated (66/68 verified)** | Session 5 |
| `report/report-writer-update.md` | **Updated** | Session 5 |

## 4. Current Draft Status

| Section | Status | Word count (approx) |
|---------|--------|-------------------|
| Abstract | **Complete** | ~140 |
| 1. Introduction | **Complete** | ~430 |
| 2. Related Work | **Complete** | ~350 |
| 3.1 MBS Dataset | **Complete** | ~160 |
| 3.2 SemEval Dataset | **Complete** | ~150 |
| 3.3 Annotation | **Complete** | ~230 |
| 4. Methods | **Complete** | ~620 |
| 5. Results | **Complete** | ~480 |
| 6. Error Analysis & Discussion | **Complete** | ~550 |
| 7. Conclusion | **Complete** | ~200 |
| Appendix | **Complete** | ~200 |
| **Total** | | **~3,400** |

All sections are now complete with verified evidence. No [MISSING], [VERIFY], or [TODO] placeholders remain.

## 5. What Remains Next

### Polish and review
1. **Glossary compliance check** — verify all terminology matches glossary exactly
2. **Style guide check** — verify prose follows output-style.md rules
3. **Cross-reference check** — verify all \ref{} and \citep{} resolve correctly
4. **Peer review pass** — team members review for factual accuracy
5. **Final word count trim** if needed (currently 3,403/4,000)

### Optional improvements (if time permits)
6. **Add more figures** — pipeline diagram, confusion matrices, error breakdown
7. **Generate BERT training curves** (B15) — would require modifying bert_model.py
8. **Run topic modelling on stage-2 predictions** — would update thematic analysis

### Separate deliverable
9. **Recorded presentation** — 5 min ±30 sec, due 2026-03-31

## 6. Warnings & Things to Verify

1. **Topic count:** Report says "16 topics plus outlier cluster" based on topic_summary.csv having 16 data rows (topics 0–15). The outlier cluster (topic -1) is the 17th group. Consistent with BERTopic defaults.
2. **BERTopic used stage-1 predictions.** This is documented in Limitations. If stage-2 is re-run on full MBS, topic analysis should be updated.
3. **Test set size (80 sentences, 15 suggestions)** is small. Metrics have wide confidence intervals. This is documented in Limitations.
4. **Word budget:** 3,403 of 4,000 words used. Room for ~600 more words if needed.
5. **LaTeX is the canonical format.** Edit `report/sections/*.tex` files. Compile with: `cd report && pdflatex main && bibtex main && pdflatex main && pdflatex main`.
