# Results — Task 2 — Yelp Polarity Sentiment

**Member:** Nikhil (Member B)

## Architecture
What was built (facts from `src/models.py` and `src/configs/sentiment_nikhil_B.yaml`). All embeddings are learned from scratch; no pretrained vectors or language models.

| Model | Architecture |
|---|---|
| B1 baseline | embedding (128) → last-token pooling → fully connected classifier |
| B2 | embedding (128) → 1-D CNN, single kernel of width 7, 128 channels → global max pooling → classifier |
| B3 | embedding (128) → one-layer bidirectional LSTM (hidden 128 per direction) → learned attention pooling (padding masked) → classifier |

**[TODO – Nikhil]** Why these three models? What does each change relative to the baseline, and what did you expect to see? Why 128-dimensional embeddings and a 30,000-word vocabulary?

## Hyperparameters
| Setting | B1 | B2 | B3 |
|---|---|---|---|
| Dropout | 0.30 | 0.40 | 0.30 |
| Optimizer | Adam | AdamW (weight decay 0.01) | AdamW (weight decay 0.01) |
| Learning rate | 1e-3 | 5e-4 | 5e-4 |
| Batch size | 128 | 128 | 64 |
| Epochs | 8 | 8 | 8 |
| Gradient clipping | 5.0 | 5.0 | 5.0 |

Shared: seed 42, vocabulary 30,000 (incl. padding and unknown), maximum length 256 tokens, stratified 90/10 validation split of the official training set (504,000 train / 56,000 validation / 38,000 test), test unknown-token rate 0.67%. Preprocessing: lowercasing, punctuation and special-character removal, stopword removal with negations kept, Porter stemming. Data analysis (balanced classes, no missing or duplicate rows, review length distribution) is in `outputs/eda/`.

Hardware: NVIDIA GeForce RTX 5090.

**[TODO – Nikhil]** How did you choose the hyperparameters and the preprocessing choices (e.g. keeping negations, max length 256)?

## Metrics
Full list in `metrics_report.csv` (run `sentiment_nikhil_B_20260929-1913`).

| Model | Accuracy | Macro-F1 | ROC-AUC | PR-AUC | MCC | Brier | ECE | Parameters | Train s | Peak MB |
|---|---|---|---|---|---|---|---|---|---|---|
| B1 last-token baseline | 0.6539 | 0.6539 | 0.7207 | 0.7091 | 0.3078 | 0.2112 | 0.0087 | 3,840,129 | 50.7 | 91.5 |
| B2 CNN k=7 | 0.9424 | 0.9424 | 0.9868 | 0.9871 | 0.8850 | 0.0430 | 0.0030 | 3,954,945 | 74.4 | 559.0 |
| B3 BiLSTM + attention | 0.9538 | 0.9538 | 0.9910 | 0.9911 | 0.9077 | 0.0354 | 0.0163 | 4,170,497 | 755.3 | 607.1 |

95% bootstrap confidence intervals (1,000 resamples):

| Model | Accuracy | Macro-F1 | MCC |
|---|---|---|---|
| B1 last-token baseline | [0.6488, 0.6588] | [0.6488, 0.6588] | [0.2976, 0.3176] |
| B2 CNN k=7 | [0.9400, 0.9448] | [0.9400, 0.9448] | [0.8803, 0.8897] |
| B3 BiLSTM + attention | [0.9518, 0.9557] | [0.9518, 0.9557] | [0.9038, 0.9115] |

Paired McNemar tests against the baseline (b = baseline right and experiment wrong; c = baseline wrong and experiment right):

| Comparison | b | c | χ² (continuity corrected) | p |
|---|---|---|---|---|
| B1 last-token baseline vs B2 CNN k=7 | 1,112 | 12,075 | 9,112.4 | < 1e-300 |
| B1 last-token baseline vs B3 BiLSTM + attention | 811 | 12,205 | 9,972.4 | < 1e-300 |

Per-slice macro-F1 and error rate (review length, negation, contrast) are in `outputs/sentiment_nikhil_B_20260929-1913/slice_metrics.csv`; confusion matrices and curves are in `confusion_matrices.png` and `curves_roc_reliability.png`.

## Notes
- Evidence: raw log and manifest under `reproducibility/*/nikhil/` (run `task2_sentiment_nikhil_B_20260929-1913`), checkpoints in `checkpoints/`, executed notebook `src/task2_sentiment_nikhil.ipynb`.
- The last-token baseline reaches only 0.654 accuracy; the other two models are above 0.94.
- Comparison with the other member is in the team report.
