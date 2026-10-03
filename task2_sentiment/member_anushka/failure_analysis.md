# Task 2 Failure Analysis

## 1. Scope and experimental context

This analysis evaluates the three sentiment classifiers trained from scratch on the Yelp Polarity dataset. The executed run used a Tesla T4 GPU, seed `3963`, a 30,000-token vocabulary, maximum sequence length of 200 tokens, batch size 64, and 10 training epochs. No pretrained language model or pretrained embedding was used.

The models were:

1. **Baseline:** learned embedding, padding-aware mean pooling, and a two-layer MLP classifier.
2. **CNN:** learned embedding followed by parallel 1-D convolutions with kernel sizes 3, 4, and 5 and global max pooling.
3. **BiGRU:** learned embedding followed by a one-layer bidirectional GRU and an MLP classifier.

The BiGRU was selected automatically as the strongest model by macro-F1 and was therefore selected as the source model for the required error-review CSV.

## 2. Observed generalization failures

The primary failure pattern was overfitting in the two higher-capacity experimental models.

| Model | Final training loss | Final validation loss | Loss gap | Final training accuracy | Final validation accuracy |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.1257 | 0.1973 | 0.0716 | 0.9522 | 0.9315 |
| CNN | 0.0168 | 0.3057 | 0.2889 | 0.9945 | 0.9377 |
| BiGRU | 0.0149 | 0.3097 | 0.2948 | 0.9951 | 0.9469 |

The CNN and BiGRU continued reducing training loss after their validation loss had begun increasing. This indicates that both models learned increasingly specific training-set patterns that did not transfer to unseen reviews. The BiGRU still achieved the best test performance, but its later epochs were less well calibrated for generalization than its earlier validation checkpoints.

## 3. Slice-level failure analysis

Review length was used as a robustness slice. The BiGRU was strongest in every reported slice, but performance decreased for very short and very long reviews.

| BiGRU slice | Count | Macro-F1 | Error rate |
|---|---:|---:|---:|
| Very short: 0–20 words | 4,337 | 0.9367 | 0.0590 |
| Short: 21–50 words | 10,344 | 0.9524 | 0.0470 |
| Medium: 51–200 words | 19,875 | 0.9512 | 0.0486 |
| Long: over 200 words | 3,444 | 0.9071 | 0.0825 |

The long-review slice had the highest error rate. Likely technical causes include truncation at 200 tokens, dilution of sentiment by mixed opinions, and multiple sentiment transitions within the same review. Very short reviews also underperform because they provide fewer sentiment-bearing tokens and may rely on sarcasm, intensifiers, or context that is absent from the text.

## 4. Model-specific failure modes

### Baseline mean-pooling model

Mean pooling removes word order and local composition. Consequently, the baseline can treat reviews with opposite phrase structures as similar when they contain similar positive or negative vocabulary. Negation and contrastive constructions such as “not good,” “good but overpriced,” or “期待 was high, result was disappointing” are especially difficult because the averaged representation does not preserve the relationship between tokens.

### CNN model

The CNN captures local n-gram patterns effectively, which explains its improvement over the baseline. However, its validation loss increased sharply after the second epoch while training loss continued to fall. This suggests memorization of highly specific phrases or review templates. Fixed convolutional windows also provide limited representation of sentiment dependencies separated by more than five tokens.

### BiGRU model

The BiGRU achieved the best overall performance, but its recurrent state can still compress long reviews into a limited final representation. This creates failure risk for reviews containing multiple clauses, topic changes, or a positive conclusion after a long negative discussion. The model also processes padded sequences without explicit packed-sequence lengths, so padding positions are passed through the GRU. Although the embedding padding vector is initialized with `padding_idx`, explicit length-aware packing would be a cleaner implementation and may improve long-review robustness.

## 5. Required 20-error review procedure

The executed notebook generated `outputs/error_review_20_template.csv` from the BiGRU predictions. It includes the raw review, cleaned review, true label, predicted label, confidence, word length, and an initial false-positive or false-negative category. The notebook did not contain completed human annotations, so this report does not claim that the 20 examples were manually reviewed.

The required review should be completed using the following deterministic selection policy:

- **Five confident false positives:** true label 0, predicted label 1, highest predicted-positive confidence.
- **Five confident false negatives:** true label 1, predicted label 0, highest predicted-negative confidence.
- **Five near-threshold errors:** misclassified examples whose maximum class probability is closest to 0.50.
- **Five slice-specific failures:** misclassified examples selected from the weakest review-length slice, with priority given to long reviews and then very short reviews.

For each selected example, record the error type, linguistic observation, and one testable corrective intervention in the CSV. The annotations should distinguish lexical ambiguity, negation failure, sarcasm, mixed sentiment, truncation, insufficient context, and data/preprocessing artifacts.

## 6. Testable corrective actions

1. Add explicit sequence lengths and use `pack_padded_sequence` before the BiGRU to prevent recurrent computation over padding.
2. Replace hard truncation at 200 tokens with a longer limit or hierarchical chunk aggregation for long reviews.
3. Add a validation-based early-stopping rule and retain the checkpoint with the lowest validation loss rather than always using epoch 10.
4. Apply stronger regularization to the CNN and BiGRU, such as lower hidden width, increased dropout, or weight decay selected through validation experiments.
5. Preserve sentiment-bearing punctuation and negation markers instead of removing all punctuation during preprocessing.
6. Calibrate the strongest model using a held-out calibration split and reassess Brier score and expected calibration error.

## 7. Conclusion

The BiGRU was the strongest classifier overall, reaching 0.9476 test accuracy and 0.9476 macro-F1. Its main weaknesses were long-review robustness and late-epoch overfitting. The most defensible next experiment is validation-based checkpoint selection combined with length-aware recurrent packing and a longer or hierarchical representation for reviews exceeding 200 tokens.
