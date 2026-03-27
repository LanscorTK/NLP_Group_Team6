# Writing Workflow & Risk Assessment

---

## Missing Information / Blockers

### Critical Blockers (report cannot be completed without these)

| # | Blocker | Depends On | Owner | Deadline |
|---|---------|-----------|-------|----------|
| 1 | **Annotation calibration labels** — 4 members must label 100 shared sentences | Manual work by all 4 members | All | ASAP (by ~17 March) |
| 2 | **Fleiss' kappa score** — IAA measurement | Blocker #1 | Chase (computation) | Day after #1 |
| 3 | **Split annotation completion** — 200-400 additional labelled sentences | Blockers #1-2 | All | ~19-20 March |
| 4 | **MBS test set evaluation** — F1/P/R for all 4 models on hotel data | Blocker #3 | Chase (models), Xiayi (baselines) | ~22-23 March |
| 5 | **BERT Stage-2 training** — domain-adapted model | Blocker #3 | Chase | ~21 March |
| 6 | **Error analysis examples** — concrete FP/FN with analysis | Blocker #4 | Xiayi | ~24 March |

### Non-Critical but Valuable

| # | Item | Status | Fallback |
|---|------|--------|----------|
| 7 | Manual validation (100+100 predictions) | Not started | Can omit — nice to have, not essential |
| 8 | Re-run BERTopic with stage-2 predictions | Preliminary run done with stage-1 | Use stage-1 results and note the caveat |
| 9 | Literature search (2-3 additional papers) | Not started | Can write with SemEval + BERT citations minimum |
| 10 | Pipeline diagram figure | Not created | Can describe in text, but figure is much better |

---

## Likely Weaknesses If Writing Only From the Proposal

1. **No empirical results on hotel data.** The proposal describes a plan; the report needs numbers. Without MBS evaluation, Results and Discussion sections are empty.
2. **No IAA measurement.** Annotation without IAA looks methodologically weak and loses easy marks in the Methodology & Analysis criterion (30% of report grade).
3. **Planned ≠ completed.** The proposal lists 7 phases. If Phases 3-6 are incomplete, the report must honestly say so — framing incomplete work as completed is fabrication.
4. **No error analysis.** The Discussion section would be generic observations rather than evidence-based analysis of specific failure modes.
5. **Domain gap is assertion, not finding.** Without cross-domain F1 comparisons, the "domain gap" story is speculation rather than a quantified result.

---

## Recommended Writing Workflow

### Phase 1: Parallel Foundation (Now — 17 March)

**Can start immediately, independent of annotation:**

| Task | Owner | Notes |
|------|-------|-------|
| Draft Section 1 (Introduction) | Leah | Can write from proposal + project motivation |
| Draft Section 2 (Related Work) | Leah | Requires literature search; start with known papers |
| Draft Section 3.1-3.2 (Data — MBS + SemEval description) | Clara | All stats available from completed Phases 1-2 |
| Draft Section 4 (Methods — model descriptions) | Clara | Can describe all 4 models from existing code |
| Draft Section 3.3 skeleton (Annotation — with placeholders) | Clara | Write structure, leave IAA and final stats as [MISSING] |
| Compile reference list | Leah | Gather all citations needed |
| Create pipeline diagram figure | Chase or Clara | Data flow: raw → filtered → segmented → annotated → models → topics |
| **All 4 members: complete calibration labelling** | **All** | **THIS IS THE CRITICAL PATH** |

### Phase 2: Fill Results (18-24 March)

**Unlocked once annotation is complete:**

| Task | Owner | Dependency |
|------|-------|-----------|
| Compute IAA, fill Section 3.3 | Chase → Clara | Calibration labels done |
| Run baselines on MBS test set | Xiayi | Annotation complete |
| Train BERT Stage-2, evaluate | Chase | Annotation complete |
| Fill Section 5 (Results) with real numbers | Clara | Model evaluation done |
| Build main results table (Table 3) | Clara | All model results available |
| Generate MBS confusion matrices | Chase | Model evaluation done |
| Re-run BERTopic with stage-2 model (optional) | Leah | Stage-2 checkpoint available |

### Phase 3: Error Analysis & Discussion (24-27 March)

| Task | Owner | Dependency |
|------|-------|-----------|
| Categorise errors from MBS test set | Xiayi | MBS evaluation done |
| Draft Section 6 (Error Analysis & Discussion) | Xiayi | Error examples collected |
| Manual validation of 100+100 predictions (if time) | Xiayi | Best model predictions available |
| Draft Section 7 (Conclusion) | Leah | Results and Discussion drafted |
| Draft Abstract | Clara | All sections drafted |

### Phase 4: Integration & Polish (27-30 March)

| Task | Owner | Notes |
|------|-------|-------|
| Merge all sections into single document | Clara (lead) | Resolve inconsistencies, unify voice |
| Word count check — must be ≤ 4,000 | All | Cut from verbose sections; move details to tables/appendix |
| Cross-check: every claim has an artifact backing it | All | Use evidence_checklist.md |
| Verify all figures are referenced in text | All | Each figure should be discussed |
| Compile appendices (annotation guidelines, extra figures) | Leah | |
| Write AI tool acknowledgement | Chase | Required by brief |
| Write member contribution table (first page) | All | Must agree on wording |
| Final proofread | All | Minimum 2 people read the full draft |
| Format as LaTeX/Notebook PDF | Chase | |

### Phase 5: Submission (30-31 March)

| Task | Owner | Notes |
|------|-------|-------|
| Final PDF export and quality check | Chase | Verify figures render, word count OK |
| Submit report to Moodle | Any member | By 10:00 AM UK time, 31 March |
| Submit code repository (ensure reproducibility) | Chase | `run_pipeline.py` works end-to-end |

---

## Top 5 Report Risks

### Risk 1: Annotation Not Completed in Time
**Probability:** Medium-High (currently blocked on manual work)
**Impact:** Critical — without labels, Sections 3.3, 5, and 6 cannot be written with real data.
**Mitigation:**
- Make calibration labelling the absolute top priority for all 4 members.
- If full annotation cannot be completed, write the report honestly: present SemEval results as the primary evaluation, note that MBS evaluation was attempted but not completed, and discuss what the expected outcomes would be based on the domain gap analysis. Partial honesty scores better than fabricated completeness.
- Minimum viable: even 100 calibration sentences with majority-vote labels can serve as a small test set.

### Risk 2: Fabricating or Embellishing Results
**Probability:** Low (but tempting under time pressure)
**Impact:** Academic misconduct risk + the markers will likely notice inconsistencies.
**Mitigation:**
- Every number in the report must trace to a code output or data file.
- Use evidence_checklist.md to verify before submission.
- If a result is missing, use [MISSING] in the draft and fill it when available.
- It is FAR better to write "we were unable to complete X due to time constraints" than to invent numbers.

### Risk 3: Exceeding the 4,000 Word Limit
**Probability:** High (7-phase pipeline + domain gap story is a lot to cover)
**Impact:** 10 percentage point deduction.
**Mitigation:**
- Word budget is allocated in outline.md — stick to it.
- Move ALL numbers into tables (excluded from count).
- Move preprocessing details into a pipeline figure.
- Move annotation guidelines into appendix.
- Every sentence must earn its place. Cut ruthlessly.

### Risk 4: Weak Error Analysis / Discussion
**Probability:** Medium
**Impact:** Loses marks on the highest-weighted criterion (Methodology & Analysis, 30%).
**Mitigation:**
- Even with a small test set (60-100 sentences), analyse every single error individually.
- Frame the domain gap as a finding, not just a limitation.
- Connect error types back to annotation guideline edge cases.
- Discuss what the errors reveal about the task itself (e.g. implicit suggestions are fundamentally harder).

### Risk 5: Report Reads Like a Technical Log, Not a Scientific Paper
**Probability:** Medium (common in student projects)
**Impact:** Loses marks on Quality of Narrative (10%) and Structure & Format (5%).
**Mitigation:**
- The report should tell a STORY: "We asked whether NLP can find suggestions in hotel reviews. Here's what we tried, what we found, and what we learned."
- Don't describe every preprocessing step — summarise in a diagram.
- Don't describe every hyperparameter in prose — put them in a table.
- Lead each section with the WHY before the WHAT.
- The three-tier model comparison should feel like a progressive argument (rules → features → context), not a list of experiments.
- Use the domain gap as a narrative thread that connects Introduction → Methods → Results → Discussion.
