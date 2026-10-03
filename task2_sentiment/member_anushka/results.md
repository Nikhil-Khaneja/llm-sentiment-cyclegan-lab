# Task 2 Results: Yelp Polarity Sentiment Classification

## Experimental setup

The experiment used the `fancyzhx/yelp_polarity` dataset with a stratified 90/10 split of the original training data for training and validation. The original test split was retained for final evaluation.

| Split | Examples | Class 0 | Class 1 |
|---|---:|---:|---:|
| Train | 504,000 | 252,000 | 252,000 |
| Validation | 56,000 | 28,000 | 28,000 |
| Test | 38,000 | 19,000 | 19,000 |

Reviews were lowercased, punctuation and non-alphanumeric characters were removed, a defined stopword set was removed, and the remaining text was tokenized. The vocabulary was learned from the training split only and capped at 30,000 tokens. Each sequence was truncated or padded to 200 tokens. The mean cleaned-review length was approximately 91 words across the three splits.

## Hardware and configuration

- Device: CUDA
- GPU: Tesla T4
- Random seed: `3963`
- Batch size: `64`
- Epochs: `10`
- Embedding dimension: `128`
- Dropout: `0.30`
- Optimizer: AdamW
- Baseline learning rate: `1e-3`
- Experimental-model learning rate: `5e-4`
- Weight decay: `1e-4`
- Trainable embeddings: yes
- Pretrained embeddings or language models: no

## Model definitions

### Baseline

The baseline uses a learned embedding layer, padding-aware mean pooling, and an MLP with hidden width 64. It contains 3,848,386 trainable parameters.

### CNN experiment

The CNN uses a learned embedding layer and parallel 1-D convolution branches with kernel sizes 3, 4, and 5. Each branch has 128 channels; max pooling is applied over the sequence dimension before classification. It contains 4,086,530 trainable parameters.

### BiGRU experiment

The BiGRU uses a learned embedding layer followed by a one-layer bidirectional GRU with hidden size 128 per direction. The final forward and backward hidden states are concatenated and passed to an MLP classifier. It contains 4,071,298 trainable parameters.

## Training behavior

| Model | Final train loss | Final validation loss | Final train accuracy | Final validation accuracy |
|---|---:|---:|---:|---:|
| Baseline | 0.1257 | 0.1973 | 0.9522 | 0.9315 |
| CNN | 0.0168 | 0.3057 | 0.9945 | 0.9377 |
| BiGRU | 0.0149 | 0.3097 | 0.9951 | 0.9469 |

The CNN and BiGRU achieved very low training loss but developed substantial train-validation gaps. This is evidence of overfitting, especially after the middle of training. The baseline had the smallest generalization gap but lower overall predictive performance.

## Final test metrics

| Model | Accuracy | Macro precision | Macro recall | Macro-F1 | Test examples/sec | Test time (sec) | Training time (sec) | Peak GPU memory (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 0.9350 | 0.9351 | 0.9350 | 0.9350 | 13,488.35 | 2.817 | 488.15 | 90.76 |
| CNN | 0.9407 | 0.9409 | 0.9407 | 0.9407 | 11,142.64 | 3.410 | 882.22 | 198.65 |
| BiGRU | 0.9476 | 0.9477 | 0.9476 | 0.9476 | 14,700.18 | 2.585 | 828.05 | 355.36 |

Because the test set is balanced, micro-F1 equals accuracy to the displayed precision. The BiGRU was the best-performing model and was selected for the error-review template.

## Bootstrap confidence intervals

The notebook used 1,000 bootstrap repetitions with a fixed NumPy generator seed.

| Model | Accuracy 95% CI | Macro-F1 95% CI | MCC 95% CI |
|---|---|---|---|
| Baseline | [0.9327, 0.9374] | [0.9327, 0.9374] | [0.8655, 0.8750] |
| CNN | [0.9384, 0.9431] | [0.9384, 0.9431] | [0.8772, 0.8865] |
| BiGRU | [0.9452, 0.9497] | [0.9452, 0.9497] | [0.8905, 0.8995] |

## Paired McNemar comparisons

The McNemar tests compare paired predictions on the same 38,000 test examples.

| Comparison | Baseline-only correct | Experimental-only correct | Chi-square statistic | p-value |
|---|---:|---:|---:|---:|
| Baseline vs. CNN | 1,097 | 1,315 | 19.5228 | 9.94e-06 |
| Baseline vs. BiGRU | 860 | 1,340 | 104.2914 | 1.75e-24 |

Both experimental models produced statistically significant changes in paired classification outcomes relative to the baseline. The BiGRU comparison shows the strongest evidence of improvement.

## Review-length robustness

| Model | Very short F1 / error | Short F1 / error | Medium F1 / error | Long F1 / error |
|---|---|---|---|---|
| Baseline | 0.9238 / 0.0712 | 0.9423 / 0.0570 | 0.9360 / 0.0636 | 0.9000 / 0.0891 |
| CNN | 0.9264 / 0.0683 | 0.9442 / 0.0549 | 0.9450 / 0.0549 | 0.9050 / 0.0865 |
| BiGRU | 0.9367 / 0.0590 | 0.9524 / 0.0470 | 0.9512 / 0.0486 | 0.9071 / 0.0825 |

The long-review slice is the weakest slice for every model. The BiGRU remains the strongest model in all slices, but its error rate rises from 0.0486 on medium reviews to 0.0825 on reviews longer than 200 words.

## Limitations and future work

The preprocessing removes punctuation and uses hard sequence truncation, which can discard negation, emphasis, and late-review sentiment. The BiGRU also receives padded sequences without explicit packed lengths. Future work should use length-aware recurrent packing, validation-based early stopping, stronger regularization, hierarchical long-document encoding, and calibration on a held-out calibration split.

The executed notebook generated `outputs/error_review_20_template.csv` for the required manual review. The CSV still requires human annotation of the 20 selected errors; no fabricated annotations are included in this report.
