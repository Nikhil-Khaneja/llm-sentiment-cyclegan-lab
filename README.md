# DATA266 Lab 1 — LLM, Sentiment Classification, and CycleGAN

This repository contains the DATA266 Lab 1 team submission for Fall 2026. The lab covers GPT-style language modeling, Yelp Polarity sentiment classification, and unpaired CycleGAN image translation.

## Team ownership

| Task | Nikhil | Anushka |
|---|---|---|
| Task 1 — LLM from scratch | 2 blocks × 8 heads character-level GPT, full 12-epoch run | Completed implementation, training, evaluation, and failure analysis |
| Task 2 — Sentiment classification | B1 last-token baseline, B2 CNN (k=7), B3 BiLSTM + attention pooling | Completed three-model Yelp Polarity experiment and evaluation |
| Task 3 — CycleGAN | 9 residual blocks, 32 generator channels, batch 1, full 50-epoch run | Completed training, inference, evaluation, human audit, and submission artifacts |

Nikhil is Member B and Anushka is Member A in the two-member plan (`docs/assignment/DATA266_Lab1_Two_Member_Model_Plan.docx`). Both members built all three tasks independently, with different designs.

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

## Results at a glance

| Task | Nikhil (Member B) | Anushka (Member A) |
|---|---|---|
| 1 — validation perplexity | **2.326** (1.218 bits/char), 1.65 M params, 12 epochs | **1.760** (0.815 bits/char), 10.93 M params |
| 2 — test accuracy | B1 **0.654**, B2 **0.942**, B3 **0.954** (macro-F1 0.954) | baseline 0.935, CNN 0.943, BiGRU **0.948** |
| 3 — FID / MiFID (course script) | 108.03 / 0.413, 11.2 M params, 50 epochs | **96.89 / 0.410** (`task3_gan/member_anushka/submission.csv`) |

Task 3 FID and MiFID come from the course evaluation script (`task3_gan/data/Part3_Evaluation_Script.ipynb`, run through each member's `evaluate_local.py`): first 300 real versus 300 generated images, both directions averaged. This is the method behind the Kaggle leaderboard score, the negated mean of FID and MiFID. Nikhil's per-direction values are FID 107.73 / 108.34 and MiFID 0.404 / 0.422 for Photo→Monet / Monet→Photo, in `task3_gan/member_nikhil/submission_course_script.csv`. The two members' models differ in size, so the Task 1 numbers are not a controlled comparison. Team 42's leaderboard entry (rank 21 on 2026-10-06, score −48.6479) is Anushka's file. Full metric tables are in each member's `metrics_report.csv` / `full_metrics_report.csv`.

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

## Nikhil’s work (Member B)

Hardware for every run: NVIDIA GeForce RTX 5090, PyTorch 2.8.0 + CUDA 12.8; seeds are recorded in each config and manifest.

### Task 1 — Character-level GPT from scratch

Location: `task1_llm/member_nikhil/` — config `src/configs/gpt_nikhil_B.yaml`

Hand-written causal multi-head self-attention (no `nn.Transformer` / `MultiheadAttention`), pre-LayerNorm, 2 blocks × 8 heads, `d_model` 256, feed-forward 1024, dropout 0.10, block size 128, AdamW (lr 2.5e-4, cosine decay to 1e-5, 500 warm-up steps), batch 64, 12 epochs, generation temperature 0.9. Same 100,000 / 10,000 TinyStories sequences as the other member.

Key artifacts: `src/task1_gpt_nikhil.ipynb`, `checkpoints/gpt_nikhil_2L8H_B_*_best.pt`, `outputs/gpt_nikhil_2L8H_B_*/`, `metrics_report.csv`.

### Task 2 — Yelp Polarity sentiment classification

Location: `task2_sentiment/member_nikhil/` — config `src/configs/sentiment_nikhil_B.yaml`

Three classifiers with embeddings learned from scratch (no pretrained vectors or language models): **B1** embedding → last-token pooling baseline, **B2** single-kernel (k=7) CNN with global max pooling, **B3** one-layer BiLSTM with learned attention pooling. Cleaned text with lowercasing, punctuation removal, stopword removal (negations kept), Porter stemming, 30,000-word vocabulary, maximum length 256, 8 epochs each.

Key artifacts: `src/task2_sentiment_nikhil.ipynb`, three checkpoints, `outputs/sentiment_nikhil_B_*/` (metrics, McNemar tests, slice robustness, calibration, error-review candidates), `metrics_report.csv`.

### Task 3 — CycleGAN image translation

Location: `task3_gan/member_nikhil/` — config `src/configs/cyclegan_nikhil_B.yaml`

ResNet generators with 9 residual blocks and 32 base channels, two 70×70 PatchGAN discriminators, LSGAN loss, cycle weight 10, identity weight 5, Adam (2e-4, β 0.5/0.999), batch 1, 128 px training crops, 50 epochs (25 constant + 25 linear decay), checkpoints every 5 epochs, image history pool of 50. No pretrained model touches training or the submitted images; Inception and LPIPS networks are used only as fixed measuring instruments.

Key artifacts: `src/task3_cyclegan_nikhil.ipynb`, `checkpoints/…_epoch050.pt`, `outputs/cyclegan_nikhil_r9c32_B_*/` (loss curves, samples, 30-sample blinded human-audit sheets), `outputs/pred_A2B/`, `outputs/pred_B2A/`, `full_metrics_report.csv`, `submission.csv`, `submission_course_script.csv`, `evaluate_local.py` (the course script, writes `submission.csv`), `evaluate_full.py` (KID, precision/recall, LPIPS and the rest, writes `full_metrics_report.csv`), and `kaggle_score.py` (an unofficial estimate that does not match the course script).

`MEMBER_A_RUN_NOTE.md` in that folder explains one extra run: a 6-block / 64-channel CycleGAN (`cyclegan_nikhil_r6c64_*`) that follows the plan's Member A design, trained on Nikhil’s GPU by mistake and kept only as a recorded run. It is not Nikhil’s Task 3 submission.

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

Nikhil’s manifests and raw logs are in `reproducibility/manifests/nikhil/` and `reproducibility/raw_logs/nikhil/`, one `.json` and one `.log` per run, including smoke tests.

## Setup

```bash
git lfs install && git clone https://github.com/Nikhil-Khaneja/llm-sentiment-cyclegan-lab.git && cd llm-sentiment-cyclegan-lab
```

The repository uses Git LFS for model checkpoints. Install the task-specific dependencies listed in `reproducibility/manifests/` before running a notebook.

To set up a CUDA environment from the repository root:

```bash
python -m venv .venv                 # Windows: .venv\Scripts\activate | Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
# NVIDIA RTX 50-series (Blackwell) needs the CUDA 12.8 build:
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
```

Shared datasets (not committed):

```bash
python task1_llm/data/download_tinystories.py       # -> task1_llm/data/tinystories_raw.txt
python task2_sentiment/data/download_yelp.py        # -> task2_sentiment/data/{train,test}.csv
kaggle competitions download -c data-266-fall-2026-gan-image-style-transfer -p task3_gan/data
```

For Task 3, unpack the Kaggle zip so that `task3_gan/data/monet_jpg/` (300 images) and `task3_gan/data/photo_jpg/` (7,038 images) exist, and keep `real_stats.npz` next to them (see `task3_gan/data/README.md`).


## Smoke-test reproduction

For a notebook smoke test, set the task’s smoke configuration and execute the notebook with:

```bash
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=0 task2_sentiment/member_anushka/src/Part2_MemberA_Yelp_Sentiment.ipynb --output task2_sentiment/member_anushka/src/Part2_MemberA_Yelp_Sentiment_executed.ipynb
```

For Task 2, set `RUN_FULL_TRAINING = False` in a local copy before execution. Full training should use a CUDA-enabled GPU and the saved configuration. Preserve executed notebooks, outputs, checkpoints, and raw logs as evidence.

### Reproducing Nikhil’s runs

One command per task (each also has a quick `smoke_B.yaml` config for a sanity check):

```bash
python task1_llm/member_nikhil/src/train.py       --config task1_llm/member_nikhil/src/configs/gpt_nikhil_B.yaml
python task2_sentiment/member_nikhil/src/train.py --config task2_sentiment/member_nikhil/src/configs/sentiment_nikhil_B.yaml
python task3_gan/member_nikhil/src/train.py       --config task3_gan/member_nikhil/src/configs/cyclegan_nikhil_B.yaml
python task3_gan/member_nikhil/evaluate_local.py   # course script -> task3_gan/member_nikhil/submission.csv
```

The executed notebooks load the committed runs by default; set `RUN_DIR = None` in the training cell to retrain. Full-run times on an RTX 5090: Task 1 about 4 minutes, Task 2 about 16 minutes, Task 3 about 7.7 hours.

## Required member-level documentation

Each member’s task folder is expected to contain an executed source notebook, configuration and preprocessing artifacts, trained checkpoints, generated outputs, a metrics report, `results.md`, `failure_analysis.md`, raw logs, and a reproducibility manifest.

## Final report

The combined report belongs at:

```text
report/DATA266_Lab1_Report_Team_[Team Number].pdf
```

It should include the ownership statement, comparison tables for every member and task, evidence links for reported metrics, each member’s failure/error analysis, and the required paper citations.

The editable combined report with both members’ sections is `report/DATA266_Lab1_Team42_Report_v2.docx`; the PDF export is submitted under the name above.
