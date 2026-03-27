# Presentation Script — Suggestion Mining in High-Rated Hotel Reviews

**Total target: 5 minutes (4:30–5:30)**
**Speakers: Chase, Clara, Leah, Xiayi (~75 sec each)**

---

## Slide 1: Title (Xiayi, ~15 sec)

> **Show:** Project title, team names, UCL School of Management logo

Hello everyone. Our project is "Suggestion Mining in High-Rated Hotel Reviews" — we're Chase, Clara, Leah, and Xiayi from UCL School of Management.

---

## Slide 2: Motivation (Leah, ~50 sec)

> **Show:** Example 5-star review containing a hidden suggestion (e.g., "Amazing hotel — but bring your passport for the casino"); contrast with sentiment analysis output

When a guest leaves a five-star review, most NLP systems classify it as positive and move on. But buried inside that praise, there are often specific, actionable suggestions — things the hotel could actually improve.

For example, a guest might write: "Loved everything about Marina Bay Sands, just wish the elevators were faster during peak hours." Sentiment analysis labels that positive. Our goal is to extract the suggestion.

This matters because high-rated reviews are where loyal customers give the most constructive feedback — and it's exactly the feedback that gets overlooked. Our research question is: to what extent can NLP models detect these actionable suggestions within high-star hotel reviews?

---

## Slide 3: Data & Annotation (Clara, ~60 sec)

> **Show:** Dataset pipeline diagram; annotation stats table; Fleiss' kappa value

We collected 10,232 TripAdvisor reviews for Marina Bay Sands, a luxury hotel in Singapore. After filtering to four- and five-star reviews and removing nulls, we had 8,571 reviews. We segmented these into over 70,000 individual sentences using spaCy.

For training data, we used the SemEval-2019 Task 9 dataset — but that comes from software developer forums, creating a significant domain gap with hotel language.

For our hotel-specific ground truth, all four of us annotated 400 sentences. We started with a 100-sentence calibration round where all four annotators labelled the same sentences. Our initial Fleiss' kappa was 0.379 — fair agreement. The main source of disagreement was distinguishing complaints from suggestions. We revised our guidelines, resolved ties through team discussion, and then split the remaining 300 sentences across the team. Our final dataset contains 73 suggestions out of 400 sentences — an 18.2% positive rate.

---

## Slide 4: Methods (Chase, ~75 sec)

> **Show:** Pipeline diagram (`pipeline_diagram.png`); three-tier model architecture visual

We built a three-tier modelling pipeline to compare approaches of increasing sophistication.

First, a regex baseline — simple pattern matching on modal verbs like "should," "could improve," and "I recommend." This captures explicit suggestions but misses subtle ones.

Second, TF-IDF with logistic regression — a standard text classification approach trained on SemEval data with balanced class weights.

Third, and most importantly, a two-stage BERT classifier. In stage one, we fine-tuned BERT-base on roughly 7,000 SemEval training sentences. In stage two, we further fine-tuned that model on our 320 annotated hotel sentences. This two-stage approach is the core of our domain adaptation strategy — can a model trained on software forums learn to detect suggestions in hotel reviews if we give it a small amount of in-domain data?

We evaluated all models using F1 score on the suggestion class and precision-recall AUC, since accuracy alone is misleading with an 18% positive class.

---

## Slide 5: Results (Chase, ~60 sec)

> **Show:** Model comparison table (4 models x 2 datasets); `model_comparison.png`

Here's what we found. On SemEval data, BERT stage one achieved an F1 of 0.716 — strong performance. But when we applied that same model to hotel reviews, F1 collapsed to 0.286. TF-IDF showed an even steeper drop — from 0.596 to just 0.190.

Interestingly, the regex baseline actually performed better cross-domain than the learned models, reaching 0.471 F1 on hotel data. Its reliance on domain-general modal patterns made it more robust to the domain shift.

But the headline result is BERT stage two. By fine-tuning on just 320 hotel sentences, we nearly doubled the F1 from 0.286 to 0.581, with recall jumping from 0.200 to 0.600. This demonstrates that even a small amount of in-domain annotation can substantially recover cross-domain performance — domain adaptation works.

---

## Slide 6: Insights (Clara, ~45 sec)

> **Show:** `aspect_distribution.png`; `topic_wordclouds.png`; example suggestions

We then applied our best model to all 70,000 sentences and identified 4,179 predicted suggestions — about 6% of the corpus. Using BERTopic, we clustered these into 56 coherent topics.

Grouping by hotel aspect: room-related suggestions were the largest category at 24%, followed by facilities at 15% and food and beverage at 12%.

The concrete suggestions are what make this actionable — guests recommended things like adding USB charging ports in rooms, improving elevator wait times, warning about the passport requirement for the casino, and providing sunscreen at the rooftop pool. These are specific, implementable improvements that a hotel manager could act on immediately.

---

## Slide 7: Error Analysis & Limitations (Xiayi, ~45 sec)

> **Show:** `error_comparison_models.png`; error pattern summary table

Looking at where our best model fails: the main false positive pattern is imperative endorsements — sentences like "I highly recommend this hotel" that use suggestion-like language but are actually praise. The main false negative pattern is implicit suggestions — where guests hint at improvements without using direct suggestion language.

We should be transparent about limitations. Our test set is small — 80 sentences with only 15 suggestions — so a single misclassification shifts F1 by about 7 points. We studied only one hotel, so generalisability is uncertain. And annotation subjectivity remains a challenge, as our kappa of 0.379 reflects genuine ambiguity in what counts as a suggestion.

---

## Slide 8: Conclusion & Future Work (Leah, ~30 sec)

> **Show:** Research question with answer; future directions bullet points

To answer our research question: NLP models can detect suggestions in high-star hotel reviews, but domain adaptation is essential. A model trained solely on out-of-domain data fails, while two-stage fine-tuning with a small in-domain dataset recovers meaningful performance.

For future work, large language models could improve detection of implicit suggestions. Expanding to multiple hotels and larger annotated datasets would strengthen generalisability. Thank you.

---

## Speaker Summary

| Speaker | Slides | Approx. Time |
|---------|--------|--------------|
| Xiayi   | 1, 7   | ~60 sec      |
| Leah    | 2, 8   | ~80 sec      |
| Clara   | 3, 6   | ~105 sec     |
| Chase   | 4, 5   | ~135 sec     |

**Notes:**
- Practice target: under 5:00 at natural pace (the written script reads ~5:30 at 150 wpm — pauses and natural delivery will compress it)
- Reference figures from `outputs/figures/` for each slide visual
- Transitions between speakers should be seamless — no "now I'll hand over to..."
