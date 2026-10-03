# DATA266 Lab 1 — Task 2

## Yelp Polarity Sentiment Classification

This folder contains Anushka’s Task 2 implementation for binary sentiment classification on the Yelp Polarity dataset. Three models were trained independently using PyTorch. All embeddings were learned from scratch, with no pretrained embeddings or pretrained language models.

## Models

1. **Baseline:** learned embedding, padding-aware mean pooling, and an MLP classifier.
2. **CNN:** learned embedding with parallel 1-D convolution branches using kernel sizes 3, 4, and 5.
3. **BiGRU:** learned embedding with a one-layer bidirectional GRU.

## Dataset and preprocessing

The experiment uses `fancyzhx/yelp_polarity`. The original training data was divided into stratified training and validation splits, while the original test split was retained for final evaluation.

| Split | Examples | Class 0 | Class 1 |
|---|---:|---:|---:|
| Train | 504,000 | 252,000 | 252,000 |
| Validation | 56,000 | 28,000 | 28,000 |
| Test | 38,000 | 19,000 | 19,000 |

Reviews were lowercased, punctuation and special characters were removed, selected stopwords were removed, and the cleaned text was tokenized. The vocabulary was learned from the training split only and capped at 30,000 tokens. Sequences were padded or truncated to 200 tokens.

## Configuration and hardware

- Runtime: Google Colab
- GPU: Tesla T4
- Random seed: `3963`
- Batch size: `64`
- Epochs: `10`
- Embedding dimension: `128`
- Dropout: `0.30`
- Optimizer: AdamW
- Baseline learning rate: `0.001`
- Experimental-model learning rate: `0.0005`
- Weight decay: `0.0001`

## Test results

| Model | Accuracy | Macro-F1 | ROC-AUC | PR-AUC | MCC | Training time (sec) | Peak GPU memory (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 0.9350 | 0.9350 | 0.9813 | 0.9812 | 0.8701 | 465.62 | 90.76 |
| CNN | 0.9426 | 0.9426 | 0.9856 | 0.9846 | 0.8851 | 888.58 | 198.65 |
| BiGRU | 0.9476 | 0.9476 | 0.9865 | 0.9862 | 0.8953 | 843.36 | 355.36 |

The BiGRU achieved the strongest overall performance and was selected for the required 20-example error review.

## Task 2 folder contents

```text
task2_sentiment/member_anushka/
├── src/
│   ├── Part2_MemberA_Yelp_Sentiment.ipynb
│   ├── config.json
│   └── vocabulary.json
├── checkpoints/
│   ├── baseline.pt
│   ├── cnn.pt
│   └── bigru.pt
├── logs/
│   ├── baseline_training_log.txt
│   ├── cnn_training_log.txt
│   └── bigru_training_log.txt
├── outputs/
│   ├── metrics_report.csv
│   ├── mcnemar_tests.csv
│   ├── slice_robustness.csv
│   ├── error_review_20.csv
│   ├── error_review_20_template.csv
│   ├── data_analysis.csv
│   ├── data_distribution.png
│   └── evaluation plots
├── metrics_report.csv
├── results.md
└── failure_analysis.md
```

## Repository-level reproducibility artifacts

The shared reproducibility artifacts are stored at the repository root, as required by the team repository structure:

```text
reproducibility/
├── manifests/
│   ├── task2_sentiment_anushka_manifest.json
│   └── task2_sentiment_anushka_requirements.txt
└── raw_logs/
    └── task2_sentiment_anushka/
        ├── baseline_training_log.txt
        ├── cnn_training_log.txt
        └── bigru_training_log.txt
```

The manifest maps the executed notebook, configuration, checkpoints, metrics, and raw logs. The raw logs are retained as evidence from the three model runs. The detailed metrics table is available at both `task2_sentiment/member_anushka/metrics_report.csv` and `outputs/metrics_report.csv`.

## Error review

`outputs/error_review_20.csv` contains the selected 20 model errors required by the assignment:

- 5 confident false positives
- 5 confident false negatives
- 5 near-threshold errors
- 5 slice-specific failures

The columns `reviewed_error_type`, `observation`, and `testable_fix` must be completed after manually inspecting the selected review text.

## Documentation

- `results.md` documents preprocessing, architectures, hyperparameters, metrics, statistical tests, hardware, and robustness results.
- `failure_analysis.md` documents overfitting, model-specific weaknesses, review-length failures, and testable improvements.
