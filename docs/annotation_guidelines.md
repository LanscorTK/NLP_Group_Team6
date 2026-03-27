# Annotation Guidelines — Suggestion Mining in Hotel Reviews

**Project:** NLP_Team6 — Suggestion Mining in High-Rated Hotel Reviews  
**Version:** 1.1 (revised after calibration round)
**Date:** 2026-03-14  
**Adapted from:** SemEval-2019 Task 9 Subtask A (Negi et al., 2019)

---

## 1. Task Definition

You will label individual **sentences** extracted from 4-5 star Marina Bay Sands TripAdvisor reviews. Each sentence receives one of two labels:

| Label | Meaning |
|---|---|
| **1** (Suggestion) | The sentence proposes an improvement, gives advice, or recommends an action |
| **0** (Non-Suggestion) | The sentence describes an experience, states an opinion, or gives praise/complaint without actionable direction |

## 2. Core Decision Rule

Ask yourself one question:

> **"Does this sentence propose a concrete action the hotel could take to improve?"**

- If **yes** → label **1**
- If **no** → label **0**

The key word is **actionable**. A suggestion points toward something that can be changed, added, or improved. A complaint or observation by itself, without an actionable direction, is **not** a suggestion.

## 3. Clear Examples

### Definite Suggestions (label = 1)

| # | Sentence | Why it's a suggestion |
|---|---|---|
| 1 | "They should add more vegetarian options at breakfast." | Explicit action: add vegetarian options |
| 2 | "Please provide more towels in the room." | Direct request with "please" |
| 3 | "I recommend booking a room facing the bay." | Advice to future guests |
| 4 | "The hotel could improve the Wi-Fi speed." | "Could improve" signals an actionable area |
| 5 | "Consider extending the pool hours past 10 PM." | "Consider" + specific action |

### Definite Non-Suggestions (label = 0)

| # | Sentence | Why it's NOT a suggestion |
|---|---|---|
| 1 | "The room was beautiful and spacious." | Pure praise, no action proposed |
| 2 | "We had an amazing time at the infinity pool." | Positive experience description |
| 3 | "The check-in process was smooth." | Factual observation |
| 4 | "This is one of the best hotels in Singapore." | General opinion |
| 5 | "We stayed for 3 nights in July." | Factual statement |

## 4. Edge Cases — Decision Guide

These are the hard cases. Read each carefully and apply the rules consistently.

### 4.1 Wishes and Desires → **label 1**

Sentences expressing a wish for something to be different **are** suggestions, because they imply a desired change.

| Sentence | Label | Reasoning |
|---|---|---|
| "I wish the pool was open later." | **1** | Implies: extend pool hours |
| "If only the breakfast had more variety." | **1** | Implies: add breakfast variety |
| "It would have been nice to have a kettle in the room." | **1** | Implies: provide kettles |

### 4.2 Implicit Comparisons → **label 1**

Sentences comparing the hotel unfavourably to others **are** suggestions, because they imply the hotel should match the competitor.

| Sentence | Label | Reasoning |
|---|---|---|
| "Other hotels in the area offer free airport shuttle." | **1** | Implies: add airport shuttle |
| "Compared to other 5-star hotels, the amenities were lacking." | **1** | Implies: improve amenities |

### 4.3 Conditional Praise (but + could/should) → **label 1**

Sentences that give praise but then suggest improvement **are** suggestions. The "but" clause contains the actionable part.

| Sentence | Label | Reasoning |
|---|---|---|
| "The breakfast was decent but could have more variety." | **1** | "Could have more variety" is actionable |
| "Great hotel, but they need to fix the elevator wait times." | **1** | "Need to fix" is actionable |
| "Nice room, though the bathroom could use a renovation." | **1** | "Could use a renovation" is actionable |

### 4.4 Complaints WITHOUT Actionable Direction → **label 0**

Pure complaints that describe a negative experience but do NOT propose what should be done **are not** suggestions. This is the most important distinction.

| Sentence | Label | Reasoning |
|---|---|---|
| "The check-in took over an hour." | **0** | Describes problem, no action proposed |
| "The walls were thin and we could hear everything." | **0** | Describes problem, no specific fix proposed |
| "The breakfast was disappointing." | **0** | Negative opinion, but no direction for improvement |
| "Our room wasn't cleaned until 4 PM." | **0** | Factual complaint, no suggestion |

**The test:** If you can rewrite the sentence as "The hotel should [do X]" and X is specific and concrete, it's a suggestion. If you can only rewrite it as "Something was bad," it's a complaint.

### 4.5 Positive Modal Verbs (would/could for praise) → **label 0**

Modal verbs like "would" and "could" are NOT always suggestion signals. When used for praise or future intent, they are **not** suggestions.

| Sentence | Label | Reasoning |
|---|---|---|
| "I would definitely stay here again." | **0** | "Would" expresses intent, not a suggestion |
| "I could not have asked for a better experience." | **0** | "Could not" is praise |
| "We would recommend this hotel to anyone." | **0** | Recommendation of the hotel, not a suggestion for improvement |

### 4.6 Advice to Other Guests (not to the hotel) → **label 1**

Advice directed at future guests (e.g., "make sure to book early") is still a suggestion in the SemEval definition, even though it doesn't ask the hotel to change. A suggestion requires a **directive element** — words like "try", "make sure", "I recommend", "pick", "book", or similar.

| Sentence | Label | Reasoning |
|---|---|---|
| "Make sure to book a bay-view room." | **1** | Advice/recommendation |
| "Try the laksa at the food court downstairs." | **1** | Recommendation |
| "Arrive early to get a good spot at the pool." | **1** | Advice to guests |
| "I recommend to ask for a room with a city view on a high floor." | **1** | Explicit recommendation |
| "If you can, pick city skyline or premier room, which are higher." | **1** | Directive: "pick" |

**Important — informational statements are NOT suggestions:** Sentences that simply describe what options exist, or provide navigational/factual information, are **not** suggestions even if they mention places or choices.

| Sentence | Label | Reasoning |
|---|---|---|
| "You can opt for Rise (in the lobby) or Spago (55th floor)." | **0** | Lists options without recommending one — informational |
| "Need to walk to Tower 1 where it linked to the Shoppes." | **0** | Navigational description, no directive |
| "There are less people and you'll get to enjoy the view." | **0** | Describes a fact, no action proposed |

### 4.7 Questions → **label 0** (usually)

Questions are generally not suggestions unless they are rhetorical suggestions.

| Sentence | Label | Reasoning |
|---|---|---|
| "Why don't they offer late checkout?" | **1** | Rhetorical — implies they should offer it |
| "I wonder what the gym looks like?" | **0** | Genuine question, no suggestion |

### 4.8 Complaints, Observations & Narrative → **label 0**

**If the sentence only describes what happened or how the reviewer felt, without proposing what should change, label 0 — even if the experience was negative.**

This is the most common source of over-labelling. Many negative sentences feel like they *imply* a suggestion, but unless the sentence actually states or clearly implies a specific action, it is a complaint or observation, not a suggestion. Apply the rewrite test: can you rewrite the sentence as "The hotel should [do X]" where X is specific and concrete? If you can only rewrite it as "Something was bad," it is a complaint.

| Sentence | Label | Reasoning |
|---|---|---|
| "And I think the beer prices must have gone up." | **0** | Observation about pricing, no proposed action |
| "The club sandwich was dry and the chicken was bland." | **0** | Food complaint, no direction for change |
| "It felt quite chaotic." | **0** | Subjective impression, no action |
| "Check in was fairly slow and took us about 15 minutes." | **0** | Describes experience, no proposed fix |
| "The third had a foldout bed which was a little disappointing." | **0** | Mild complaint, no suggestion |
| "I can't believe a high end hotel would actually offer a guest to give up 2 rooms." | **0** | Expresses disbelief/frustration, no specific action proposed |
| "The first night I wanted a quick drink but could not find anywhere." | **0** | Narrative about a negative experience |
| "Everyone had a camera in hand and capturing pictures." | **0** | Pure description, neutral |

**Contrast with actual suggestions that address similar topics:**

| Complaint (label 0) | Corresponding suggestion (label 1) |
|---|---|
| "The breakfast was disappointing." | "The breakfast could have more variety." |
| "Check-in took over an hour." | "They should streamline the check-in process." |
| "The beer prices have gone up." | "The hotel should offer more affordable drink options." |

## 5. Quick Reference Decision Flowchart

```
Is there a concrete, actionable change proposed?
├── YES → Does it point to something the hotel (or guest) could DO?
│   ├── YES → Label 1 (Suggestion)
│   └── NO  → Label 0
└── NO  → Is it a wish, desire, or implicit comparison?
    ├── YES → Label 1 (Suggestion)
    └── NO  → Label 0 (Non-Suggestion)
```

## 6. Important Rules

1. **Label the sentence in isolation.** Do not consider the surrounding review context. Each sentence stands alone.
2. **When in doubt, label 0.** It is better to miss a borderline suggestion than to over-label. Precision matters.
3. **Do not discuss labels with other annotators** during the calibration round. Independence is critical for valid inter-annotator agreement.
4. **Ignore spelling/grammar errors.** Label based on meaning, not quality of writing.
5. **Ignore the star rating.** These are all from 4-5 star reviews, but treat each sentence on its own merits.

## 7. Calibration Revision Log

**v1.0 → v1.1 (2026-03-15):** After the IAA calibration round (Fleiss' kappa = 0.379), two guideline gaps were identified:
1. **Section 4.8 added** — Complaints and narrative descriptions were being labelled as suggestions. Added explicit negative examples from calibration disagreements with the complaint-vs-suggestion contrast table.
2. **Section 4.6 clarified** — Informational/navigational statements (listing options, describing layout) were being confused with guest-directed advice. Added the "directive element" requirement and negative examples.

These revisions were agreed by all 4 annotators before proceeding to split annotation.

## 8. Enrichment Note

The annotation sample has been **enriched**: approximately 50% of sentences were selected because they contain suggestion signals (modal verbs like "should"/"could", imperative words like "please"/"recommend"). This is deliberate — it increases the expected proportion of suggestions in the sample and makes annotation more efficient. The enrichment strategy is documented in the Methods section of the report.
