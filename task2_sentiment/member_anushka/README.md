# DATA266 Lab 1 — Part 2

## Yelp Polarity Sentiment Classification

This project implements and evaluates three binary sentiment classifiers for the Yelp Polarity dataset. All textual embeddings are learned from scratch; no pretrained embeddings or pretrained language models are used.

## Models

1. **Baseline:** learned embedding, padding-aware mean pooling, and an MLP classifier.
2. **CNN:** learned embedding with parallel 1-D convolution branches using kernel sizes 3, 4, and 5.
3. **BiGRU:** learned embedding with a one-layer bidirectional GRU.

## Dataset and preprocessing

The experiment uses the `fancyzhx/yelp_polarity` dataset:

| Split | Examples | Class 0 | Class 1 |
|---|---:|---:|---:|
| Train | 504,000 | 252,000 | 252,000 |
| Validation | 56,000 | 28,000 | 28,000 |
| Test | 38,000 | 19,000 | 19,000 |

Reviews are lowercased, punctuation and special characters are removed, selected stopwords are removed, and the text is tokenized. The vocabulary is learned from the training split only and limited to 30,000 tokens. Sequences are padded or truncated to 200 tokens.

## Configuration

- Seed: `3963`
- Device: Tesla T4 GPU using CUDA
- Batch size: `64`
- Epochs: `10`
- Embedding dimension: `128`
- Dropout: `0.30`
- Optimizer: AdamW
- Baseline learning rate: `0.001`
- Experimental learning rate: `0.0005`
- Weight decay: `0.0001`

## Main results

| Model | Accuracy | Macro-F1 | Training time (sec) | Peak GPU memory (MB) |
|---|---:|---:|---:|---:|
| Baseline | 0.9350 | 0.9350 | 488.15 | 90.76 |
| CNN | 0.9407 | 0.9407 | 882.22 | 198.65 |
| BiGRU | 0.9476 | 0.9476 | 828.05 | 355.36 |

The BiGRU achieved the strongest overall test performance and was selected as the strongest model for error-review generation. The long-review slice was the weakest robustness slice for all models, primarily because sequences longer than 200 tokens are truncated.

## Generated artifacts

```text
Part2/
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
│   ├── data_analysis.csv
│   ├── data_distribution.png
│   ├── metrics_report.csv
│   ├── mcnemar_tests.csv
│   ├── slice_robustness.csv
│   ├── error_review_20_template.csv
│   ├── baseline_evaluation.png
│   ├── cnn_evaluation.png
│   └── bigru_evaluation.png
├── metrics_report.csv
├── README.md
├── results.md
└── failure_analysis.md
```

## Reproducibility

Run the notebook from the first cell after selecting a CUDA GPU. Set `RUN_FULL_TRAINING = True`, confirm that Cell 4 reports `Device: cuda` and `GPU: Tesla T4`, and preserve the generated logs and checkpoints as the evidence trail for the experiment.

## Documentation

- `results.md` summarizes preprocessing, architecture, metrics, statistical comparisons, and robustness results.
- `failure_analysis.md` discusses overfitting, model-specific weaknesses, and review-length failures.
