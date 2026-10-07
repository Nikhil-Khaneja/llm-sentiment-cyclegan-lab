# Failure / Error Analysis — Task 2 — Yelp Polarity Sentiment

**Member:** Nikhil (Member B)

## Recorded failed run
The first full attempt of B3 (BiLSTM + attention) produced `nan` training loss from the first logged step onward, so it was stopped and its partial output removed. Cause: a review whose cleaned text is empty is encoded as a single padding token (length 1), and the attention mask was built from `x == 0`, which masked every position and made the softmax over all `-inf` values return `nan`. Fix: mask by sequence length instead of by padding id (commit "Task 2: fix NaN in BiLSTM attention for empty reviews"); the full run after the fix has 0 non-finite steps. The raw log of the failed attempt was not kept.

## 20-error review (required)
The assignment asks me to review 20 of my own model's errors: 5 confident false positives, 5 confident false negatives, 5 near-threshold errors and 5 slice-specific failures, to assign an error type to each and to propose one testable fix.

Evidence pack (facts only): the strongest model is B3 (test accuracy 0.9538); it makes 1,065 false positives and 692 false negatives out of 38,000 test reviews. The 20 candidate rows were selected automatically (the `error_type` and `proposed_fix` columns are empty for you to fill) and are in `outputs/sentiment_nikhil_B_20260929-1913/error_review_candidates_B3_bilstm_attn.csv` (columns: test_row, category, label, pred, p_pos, length_bucket, has_negation, has_contrast, text, clean, error_type, proposed_fix). Highest-error slices for B3:

| Slice | n | Macro-F1 | Error rate |
|---|---|---|---|
| has_contrast = True | 21,531 | 0.9485 | 0.0512 |
| length_bucket = len>200 | 7,466 | 0.9480 | 0.0494 |
| length_bucket = len<=50 | 9,362 | 0.9492 | 0.0487 |
| has_negation = True | 28,044 | 0.9517 | 0.0469 |
| has_negation = False | 9,956 | 0.9398 | 0.0444 |
| length_bucket = len51-100 | 10,223 | 0.9557 | 0.0441 |

Error types are assigned from reading each review; "probable label noise" means the review text clearly contradicts its dataset label.

| # | Category | Review excerpt | True / pred (p_pos) | Error type (draft) | Testable fix (draft) |
|---|---|---|---|---|---|
| 1 | confident FP | "Wow love the place and everything is very clean and new! Great place… worth a try!" | neg / pos (0.9999) | Probable label noise (text is clearly positive) | Audit labels: re-label by 2 humans; measure accuracy on the cleaned set |
| 2 | confident FP | "One to two stars… Wasn't: Spicy… Had: Delicious sauce… Even BJs has a better Jambalaya" | neg / pos (0.9999) | Negation/contrast structure ("Wasn't … Had …") and comparative criticism | Keep negation scoping features; test a BiLSTM with explicit "Wasn't" handling vs current |
| 3 | confident FP | "Copper is definitely still my place, but Maharani was fine enough… Naan were all amazing" | neg / pos (0.9998) | Mixed sentiment, mostly positive words | Add a sentence-level sentiment pooling head and compare on contrast slice |
| 4 | confident FP | "The TrimTini I had was really delicious… it's not likely I will return… nice event, I enjoyed myself" | neg / pos (0.9998) | Mixed sentiment / probable label noise | Same label audit; report accuracy with ambiguous reviews removed |
| 5 | confident FP | "This was my second time dining at Company. And unfortunately, it will probably be the last…" (praise first, complaint late) | neg / pos (0.9996) | Late-reversal sentiment (positive body, negative conclusion), truncation-sensitive | Raise max length from 256 and compare error rate on >200-word slice |
| 6 | confident FN | Pharmacy review: price comparison, "Price difference is unbelievable!!… premiums are high" | pos / neg (0.0005) | Domain/lexical ambiguity (negative-sounding words about insurers, positive about the store) | Test the aspect-aware variant: pool attention only on opinion-bearing tokens |
| 7 | confident FN | "much better since they changed owners… it was terrible… horrible… Now its much better" | pos / neg (0.0005) | Temporal contrast (past negative, present positive) | Add time-contrast examples augmentation; test on `has_contrast` slice |
| 8 | confident FN | "EDIT: … Horrible service… disrespect your paying customers" | pos / neg (0.0005) | Probable label noise (review text is negative) | Label audit |
| 9 | confident FN | "Very small venue!… Plenty of great things to see, but … not a giant display" | pos / neg (0.0006) | Mild/neutral text, negative framing words ("small", "in and out in 30 minutes") | Add a neutral class check; calibrate on mild reviews |
| 10 | confident FN | "The food is crap… worst nachos… sad mess" | pos / neg (0.0007) | Probable label noise (text is clearly negative) | Label audit |
| 11 | near-threshold | "…extremely disappointed… wasted time… sales pitch" | neg / pos (0.5007) | Long review with negative content but model uncertain; truncation/length effect | Compare error vs max length (256 → 512) |
| 12 | near-threshold | "does two cuisines well… three major strengths… There are some rough spots, however" | pos / neg (0.4991) | Mixed sentiment (strengths plus caveats) | Aspect-level pooling test |
| 13 | near-threshold | "1) Hard to get in… 2) vagrant… 3) Pump system is easy to use" | neg / pos (0.5011) | Enumerated pros/cons, mixed | Same aspect-level test |
| 14 | near-threshold | "A solid two stars, yes! … food pretty good… pricey" | neg / pos (0.5013) | Mixed sentiment; rating phrase ("two stars") not seen as signal | Keep rating-number tokens in preprocessing; compare |
| 15 | near-threshold | "$49 for three rooms, ya right… would not recommend… On a positive note the techs are friendly" | neg / pos (0.5015) | Sarcasm ("ya right") plus concluding positive note | Preserve punctuation/"ya right" tokens; test a sarcasm slice |
| 16 | slice (contrast) | Las Vegas AYCE brunch, empanadas "best part", "a little under-enthused" | neg / pos (0.9996) | Mixed sentiment, contrast, long | Contrast-aware pooling / longer max length |
| 17 | slice (contrast) | Foodland: bakery "tops", "huge brownies"… | neg / pos (0.9995) | Praise of one aspect vs overall rating | Aspect-level evaluation |
| 18 | slice (contrast) | "NOTE: This was a 4-star review, but … gone down the tubes. See update below." | neg / pos (0.9995) | Update-reversal: positive body, negative update | Weight later sentences more; test a last-segment feature |
| 19 | slice (contrast) | "snobby… 2 stars only because of a young gentleman… super friendly" | neg / pos (0.9994) | Mixed sentiment; positive content about a person, negative overall | Aspect-level evaluation |
| 20 | slice (contrast) | "my husband had an omelette that was good. i had a blt, a little on the small side for $10, but bacon was great. Our server was awesome!" | neg / pos (0.9993) | Probable label noise (text is positive) | Label audit |

**Summary.** Of the 20, five are probable label noise (#1, 4, 8, 10, 20), the largest group is mixed or contrastive sentiment (#2, 3, 7, 12–14, 16, 17, 19), and the rest are late reversals / updates (#5, 18), sarcasm (#15), domain ambiguity (#6, 9) and length (#11). The single fix we would test first is a label audit of high-confidence disagreements, because it tells us how much of the remaining 4.6% error is not model error; the first model change is aspect- or sentence-level pooling evaluated on the `has_contrast` slice (error 0.0512 vs 0.0469 overall for negation).
