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

Fill in one row per candidate:

| # | Category | Review excerpt | True / predicted | Error type | Proposed testable fix |
|---|---|---|---|---|---|
| 1–20 | **[TODO – Nikhil]** | **[TODO – Nikhil]** | **[TODO – Nikhil]** | **[TODO – Nikhil]** | **[TODO – Nikhil]** |

**[TODO – Nikhil]** Summary: which error types are most common, and which single fix would you test first?
