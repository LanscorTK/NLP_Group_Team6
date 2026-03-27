# Why Guest-to-Guest Advice Gets Label 1 — Dataset Flow Explanation

## The Short Answer

The annotation step (Phase 3) intentionally uses a **broad** definition of "suggestion" that includes guest-to-guest advice. The **narrowing** to hotel-actionable insights happens later, in Phase 5 (Topic Modelling). This is by design, not an oversight.

---

## The Two Datasets and How They Connect

### Dataset 1: SemEval-2019 Task 9 (external, pre-labelled)

- **Source:** Online software forum posts (e.g., Uservoice feedback for Microsoft products)
- **Labels:** Binary — suggestion (1) vs non-suggestion (0)
- **Size:** ~9,000 sentences (training + test), already labelled by the task organisers
- **Definition of "suggestion":** Any sentence that proposes an improvement, gives advice, or recommends an action — **regardless of who the target audience is**
- **Role in our project:** Used as the primary training data for all models (regex, TF-IDF+LR, BERT stage 1)

### Dataset 2: MBS Hotel Reviews (our domain, needs annotation)

- **Source:** 8,571 TripAdvisor reviews of Marina Bay Sands (4-5 stars only)
- **Labels:** We annotate ~300-500 sentences ourselves
- **Size of labelled data:** Small (~300-500 sentences, split 80/20 into train/test)
- **Role in our project:**
  - The **test set** (~60-100 sentences) is used to evaluate cross-domain performance of all models
  - The **train set** (~240-400 sentences) is used for BERT stage-2 domain adaptation
  - The **full unlabelled dataset** (70,085 sentences) is what we ultimately run predictions on

---

## How the Datasets Flow Through the Pipeline

```
Phase 3: ANNOTATION
─────────────────────────────────────────────────────────
  MBS 70k sentences ──→ sample 300-500 ──→ human annotation
                                              │
                                              ▼
                                    mbs_annotated_train.csv (80%)
                                    mbs_annotated_test.csv  (20%)

  Label definition: BROAD (matches SemEval)
  ├── Hotel-directed suggestions     → label 1
  ├── Guest-to-guest advice          → label 1  ← THIS IS THE ONE YOU ASKED ABOUT
  ├── Wishes and desires             → label 1
  ├── Implicit comparisons           → label 1
  └── Complaints, opinions, praise   → label 0


Phase 4: MODEL DEVELOPMENT
─────────────────────────────────────────────────────────

  Step 1: Train models on SemEval data (large, out-of-domain)
  ┌─────────────────────────────┐
  │ SemEval train (7k sentences)│──→ Train regex, TF-IDF+LR, BERT stage 1
  │ (software forum domain)     │
  └─────────────────────────────┘
          │
          ▼ evaluate on
  ┌─────────────────────────────┐
  │ SemEval test (1-2k)         │──→ In-domain F1 (sanity check)
  └─────────────────────────────┘
  ┌─────────────────────────────┐
  │ MBS annotated test (60-100) │──→ Cross-domain F1 (the key metric)
  └─────────────────────────────┘

  Step 2: Domain adaptation — BERT stage 2
  ┌─────────────────────────────┐
  │ BERT stage 1 checkpoint     │
  │ + MBS annotated train       │──→ BERT stage 2 (hotel-adapted)
  └─────────────────────────────┘
          │
          ▼ evaluate on
  ┌─────────────────────────────┐
  │ MBS annotated test          │──→ Adapted F1 (should beat stage 1)
  └─────────────────────────────┘

  ⚠️ KEY POINT: For BERT stage 2 to work, our MBS labels MUST use the
  same definition as SemEval. If we used a narrower definition (only
  hotel-directed suggestions), the model would learn conflicting signals:
  - SemEval says "try the laksa downstairs" = 1 (advice)
  - Our labels would say "try the laksa downstairs" = 0 (not hotel-actionable)
  The model can't learn from contradictory training data.


Phase 5: TOPIC MODELLING — THIS IS WHERE NARROWING HAPPENS
─────────────────────────────────────────────────────────

  Best model (BERT stage 2)
          │
          ▼ predict on
  ┌───────────────────────────────────┐
  │ Full MBS dataset (70k sentences)  │
  └───────────────────────────────────┘
          │
          ▼ extract predicted suggestions
  ┌───────────────────────────────────┐
  │ All predicted suggestions (~1-5k) │
  └───────────────────────────────────┘
          │
          ▼ BERTopic clustering
  ┌───────────────────────────────────────────────────────┐
  │ Topic clusters:                                       │
  │                                                       │
  │  Cluster A: "breakfast quality"        ← ACTIONABLE   │
  │  Cluster B: "room maintenance"         ← ACTIONABLE   │
  │  Cluster C: "tips for fellow guests"   ← NOT hotel-   │
  │                                          actionable   │
  │  Cluster D: "pool/spa hours"           ← ACTIONABLE   │
  │  Cluster E: "booking advice"           ← NOT hotel-   │
  │                                          actionable   │
  └───────────────────────────────────────────────────────┘
          │
          ▼ human interpretation (Step 5.2)
  ┌───────────────────────────────────────────────────────┐
  │ Final output: actionable suggestion summary table     │
  │                                                       │
  │  Topic              | Count | Example          | Action│
  │  Breakfast quality  |  45   | "add more veggie"|  ✓   │
  │  Room maintenance   |  32   | "fix the AC"     |  ✓   │
  │  Guest tips         |  28   | "book bay view"  |  ✗   │
  │  Pool hours         |  20   | "extend to 11pm" |  ✓   │
  └───────────────────────────────────────────────────────┘

  At this stage, you can:
  1. Filter out non-hotel-actionable clusters entirely
  2. Or keep them as a separate finding ("guests also value X")
  Both are valid and interesting for the report.
```

---

## Summary: Why This Two-Stage Approach Is Correct

| Stage | Definition | Why |
|---|---|---|
| **Annotation (Phase 3)** | Broad: any advice or recommendation, regardless of target | Must match SemEval definition so models trained on SemEval transfer correctly to MBS data |
| **Topic Modelling (Phase 5)** | Narrow: filter/group by actionability | This is where we separate hotel-directed suggestions from guest-to-guest advice |

If we narrowed the definition at the annotation stage:
1. **Label inconsistency with SemEval** — the two-stage BERT training would learn contradictory signals from SemEval (broad) vs MBS (narrow)
2. **Lost signal** — guest-to-guest advice like "book a bay-view room" actually tells the hotel which features guests value most, which is itself an insight
3. **Harder annotation** — annotators would need to judge *intent* (who is the target?), which is much more subjective than judging *form* (does this propose an action?)

The broad-then-narrow approach is standard in NLP pipelines: cast a wide net with the classifier, then use clustering/analysis to extract the specific insights you care about.

---

## Why Transfer Learning Instead of Training Directly on MBS Data

A natural question: if we end up annotating MBS data anyway, why not skip SemEval entirely and just train and test on our own hotel data?

### The data size problem

| Approach | Training data | What happens |
|---|---|---|
| MBS-only | ~240-400 sentences | BERT has ~110M parameters. With only ~300 training examples, the model memorises the training set instead of learning generalisable patterns. Training loss drops to near-zero; test performance is poor and unstable. |
| SemEval → MBS (transfer learning) | ~7,000 + ~300 sentences | BERT first learns what suggestions look like in general from 7k examples (enough data to generalise), then adapts to hotel language from 300 examples. |

To put it concretely: training a 110-million-parameter model on 300 sentences is like teaching someone an entire language using a single page of text. They might memorise that page perfectly, but they won't understand anything else.

### What each stage actually teaches the model

**Stage 1 (SemEval, ~7k sentences)** teaches BERT domain-independent suggestion patterns:
- Modal verb constructions: "should", "could improve", "need to"
- Imperative forms: "please add", "consider removing"
- Wish/desire expressions: "I wish", "it would be nice if"
- The difference between a suggestion ("they should fix X") and a non-suggestion ("I would stay again")

These linguistic patterns are **universal** — they work the same way in software forums, hotel reviews, or any other domain. 7,000 labelled examples is enough for BERT to learn them reliably.

**Stage 2 (MBS, ~300 sentences)** teaches BERT domain-specific adjustments:
- Hotel vocabulary: "breakfast", "concierge", "amenities", "pool hours" — words that never appear in software forums
- Hotel-specific suggestion patterns: "the buffet could use more variety" (suggestion in hotel context, meaningless in software context)
- Domain-specific non-suggestions that look like suggestions: "I would recommend this hotel to anyone" (positive modal, not a suggestion)

300 sentences is enough for these small adjustments because the model isn't learning from scratch — it's just recalibrating.

### Why not just collect more MBS annotations?

In theory, if we annotated 5,000+ MBS sentences, we could skip SemEval entirely. But:

1. **Time constraint** — We have 4 team members and 17 days until the deadline. Annotating 5,000 sentences at ~10 seconds each = ~14 hours of labelling per person. The calibration round, disagreement resolution, and quality checks would add more. This is not feasible.
2. **The two-stage approach is itself a finding** — The performance gap between stage 1 (SemEval only) and stage 2 (SemEval + MBS) directly quantifies the value of domain adaptation. This is interesting to report and demonstrates understanding of transfer learning, which is what the assignment rewards.
3. **Methodological precedent** — The SemEval-2019 Task 9 paper and subsequent work on suggestion mining all use transfer learning for cross-domain suggestion detection. Our approach is consistent with the literature.

### The analogy

Think of it as learning to cook:

| Approach | Analogy |
|---|---|
| MBS-only (300 sentences) | Someone who has never cooked tries to learn Italian cuisine from 5 recipes. They can reproduce those 5 dishes but can't improvise. |
| SemEval → MBS (transfer learning) | A trained chef (general cooking skills from thousands of recipes) learns Italian cuisine from 5 recipes. They quickly adapt because they already understand techniques, flavour profiles, and kitchen logic. |

The "general cooking skills" are the suggestion-detection patterns BERT learns from SemEval. The "Italian specialisation" is the hotel-domain adaptation from MBS data.
