# DATA266 Lab 1 — LLM, Sentiment Classification, and CycleGAN

This repository contains the DATA266 Lab 1 team submission for Fall 2026. The lab covers GPT-style language modeling, Yelp Polarity sentiment classification, and unpaired CycleGAN image translation.

## Team ownership

| Task | Nikhil | Anushka |
|---|---|---|
| Task 1 — LLM from scratch |  | Completed implementation, training, evaluation, and failure analysis |
| Task 2 — Sentiment classification |  | Completed three-model Yelp Polarity experiment and evaluation |
| Task 3 — CycleGAN |  | Completed training, inference, evaluation, human audit, and submission artifacts |

Nikhil’s sections are intentionally blank until his independent artifacts and results are added.

## Repository layout

```text
.
├── README.md
├── task1_llm/
│   ├── data/
│   ├── member_anushka/
│   └── member_nikhil/
├── task2_sentiment/
│   ├── data/
│   ├── member_anushka/
│   └── member_nikhil/
├── task3_gan/
│   ├── data/monet_jpg/
│   ├── data/photo_jpg/
│   ├── member_anushka/
│   └── member_nikhil/
├── reproducibility/
│   ├── manifests/
│   └── raw_logs/
├── docs/assignment/
└── report/
```

Raw datasets are not committed. The `data/` directories are reserved for locally prepared datasets described in the assignment.

## Anushka’s work

### Task 1 — GPT-style LLM from scratch

Location: `task1_llm/member_anushka/`

The implementation uses character-level tokenization, learned token and positional embeddings, causal self-attention, transformer blocks, residual connections, training and validation loss tracking, text generation, checkpoints, metrics, and failure-case analysis.

Key artifacts include `src/Part1_LLM.ipynb`, `src/config.json`, `checkpoints/`, `data_processed/character_vocabulary.json`, `outputs/`, `metrics_report.csv`, `results.md`, and `failure_analysis.md`.

### Task 2 — Yelp Polarity sentiment classification

Location: `task2_sentiment/member_anushka/`

Three classifiers were trained from scratch: a learned-embedding mean-pooling baseline, a multi-kernel 1-D CNN with kernel sizes 3, 4, and 5, and a learned-embedding bidirectional GRU.

The full run used a Tesla T4 GPU, seed `3963`, vocabulary size 30,000, maximum sequence length 200, batch size 64, and 10 epochs. The BiGRU was the strongest model with 0.9476 test accuracy and 0.9476 macro-F1.

Key artifacts include the executed notebook under `src/`, configuration and vocabulary files, three checkpoints, `logs/`, evaluation outputs, `outputs/error_review_20.csv`, `metrics_report.csv`, `results.md`, and `failure_analysis.md`.

### Task 3 — CycleGAN image translation

Location: `task3_gan/member_anushka/`

The implementation trains an unpaired CycleGAN for both translation directions between Monet paintings and photographs. It includes generators, discriminators, cycle and identity losses, checkpoints, generated translations, loss curves, full metrics, local evaluation, human-audit data, and the generated submission.

Key artifacts include `src/Part3_CycleGAN.ipynb`, `src/config.json`, `checkpoints/`, `outputs/full_run_20261001_222745/`, `metrics_report.csv`, `full_metrics_report.csv`, `submission.csv`, `evaluate_local.py`, `results.md`, and `failure_analysis.md`.

## Reproducibility and evidence

The repository-level reproducibility folders follow the lab specification:

```text
reproducibility/
├── manifests/
│   ├── task1_llm_anushka_manifest.json
│   ├── task1_llm_anushka_requirements.txt
│   ├── task2_sentiment_anushka_manifest.json
│   ├── task2_sentiment_anushka_requirements.txt
│   ├── task3_gan_anushka_manifest.json
│   └── task3_gan_anushka_requirements.txt
└── raw_logs/
    ├── task1_llm_anushka/
    ├── task2_sentiment_anushka/
    └── task3_gan_anushka/
```

Raw logs are retained as the evidence trail for each training run. Manifests map configurations and checkpoints to their corresponding results.

## Setup

```bash
git lfs install && git clone https://github.com/Nikhil-Khaneja/llm-sentiment-cyclegan-lab.git && cd llm-sentiment-cyclegan-lab
```

The repository uses Git LFS for model checkpoints. Install the task-specific dependencies listed in `reproducibility/manifests/` before running a notebook.

## Smoke-test reproduction

For a notebook smoke test, set the task’s smoke configuration and execute the notebook with:

```bash
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=0 task2_sentiment/member_anushka/src/Part2_MemberA_Yelp_Sentiment.ipynb --output task2_sentiment/member_anushka/src/Part2_MemberA_Yelp_Sentiment_executed.ipynb
```

For Task 2, set `RUN_FULL_TRAINING = False` in a local copy before execution. Full training should use a CUDA-enabled GPU and the saved configuration. Preserve executed notebooks, outputs, checkpoints, and raw logs as evidence.

## Required member-level documentation

Each member’s task folder is expected to contain an executed source notebook, configuration and preprocessing artifacts, trained checkpoints, generated outputs, a metrics report, `results.md`, `failure_analysis.md`, raw logs, and a reproducibility manifest.

## Final report

The combined report belongs at:

```text
report/DATA266_Lab1_Report_Team_[Team Number].pdf
```

It should include the ownership statement, comparison tables for every member and task, evidence links for reported metrics, each member’s failure/error analysis, and the required paper citations.
