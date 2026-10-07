# DATA266 Lab 1 — LLM, Sentiment Classification, and CycleGAN

This repository contains the DATA266 Lab 1 team submission for Fall 2026. The lab covers GPT-style language modeling, Yelp Polarity sentiment classification, and unpaired CycleGAN image translation.

## Team ownership

| Task | Nikhil | Anushka |
|---|---|---|
| Task 1 — LLM from scratch | 2 blocks × 8 heads character-level GPT, full 12-epoch run | 6 blocks × 6 heads character-level GPT, d_model 384, d_ff 1536, context 512, 20 epochs, 10.93 M parameters |
| Task 2 — Sentiment classification | B1 last-token baseline, B2 CNN (k=7), B3 BiLSTM + attention pooling | Mean-pooling baseline, multi-kernel CNN (k=3/4/5), one-layer BiGRU, vocabulary 30,000, max length 200, 10 epochs |
| Task 3 — CycleGAN | 9 residual blocks, 32 generator channels, batch 1, full 50-epoch run | 9 residual blocks, 64 generator channels, 64 discriminator channels, 256 px, batch 1, full 50-epoch run |

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
| 1 — validation perplexity | **2.326** (1.218 bits/char), 1.65 M params, 12 epochs | **1.760** (0.815 bits/char), 10.93 M params, 20 epochs |
| 2 — test accuracy | B1 **0.654**, B2 **0.942**, B3 **0.954** (macro-F1 0.954) | baseline 0.935, CNN 0.943, BiGRU **0.948** (macro-F1 0.948), up to 4.09 M params, 10 epochs |
| 3 — FID / MiFID (course script) | 108.03 / 0.413, 11.2 M params, 50 epochs | **96.89 / 0.410**, 28.29 M params, 50 epochs (`task3_gan/member_anushka/submission.csv`) |

Task 3 FID and MiFID come from the course evaluation script (`task3_gan/data/Part3_Evaluation_Script.ipynb`, run through each member's `evaluate_local.py`): first 300 real versus 300 generated images, both directions averaged. This is the method behind the Kaggle leaderboard score, the negated mean of FID and MiFID. Nikhil's per-direction values are FID 107.73 / 108.34 and MiFID 0.404 / 0.422 for Photo→Monet / Monet→Photo, in `task3_gan/member_nikhil/submission_course_script.csv`. The two members' models differ in size, so the Task 1 numbers are not a controlled comparison. Team 42's leaderboard entry (rank 21 on 2026-10-06, score −48.6479) is Anushka's file. Nikhil’s own submission, made after joining the team, scored −54.2222; the team’s best score counts. Full metric tables are in each member's `metrics_report.csv` / `full_metrics_report.csv`.

## Anushka’s work

### Task 1 — GPT-style LLM from scratch

Location: `task1_llm/member_anushka/`

Anushka implemented a decoder-only character-level GPT from scratch on TinyStories. The model uses 6 pre-LayerNorm transformer blocks, 6 manually implemented causal-attention heads per block, learned token and positional embeddings, residual connections, a 4× feed-forward network, mixed-precision training, text generation, checkpointing, and failure-case analysis. The final run used 100,000 training sequences and 10,000 validation sequences of 512 characters for 20 epochs.

The final checkpoint achieved validation cross-entropy 0.5651, perplexity **1.760**, 0.815 bits per character, and 82.01% next-character accuracy with 10,926,443 parameters. Key artifacts include `src/Part1_LLM.ipynb`, `src/config.json`, `src/Dockerfile`, `checkpoints/best_model.pt`, `data_processed/character_vocabulary.json`, `outputs/`, `metrics_report.csv`, `results.md`, and `failure_analysis.md`. The complete run-to-checkpoint mapping is recorded in `reproducibility/manifests/task1_llm_anushka_manifest.json`.

### Task 2 — Yelp Polarity sentiment classification

Location: `task2_sentiment/member_anushka/`

Three classifiers were trained from scratch on `fancyzhx/yelp_polarity`: a padding-aware learned-embedding mean-pooling baseline, a multi-kernel 1-D CNN with kernel sizes 3, 4, and 5, and a learned-embedding one-layer bidirectional GRU.

The full run used a Tesla T4 GPU, seed `3963`, vocabulary size 30,000, maximum sequence length 200, batch size 64, and 10 epochs. Test accuracy was 0.9350 for the baseline, 0.9426 for the CNN, and **0.9476 for the BiGRU**, which was also the strongest model by macro-F1 (0.9476). The notebook also produces confidence intervals, ROC/PR metrics, calibration metrics, McNemar tests, slice robustness results, and a 20-example error review.

Key artifacts include `src/Part2_MemberA_Yelp_Sentiment.ipynb`, `src/config.json`, `src/vocabulary.json`, three checkpoints, `logs/`, evaluation outputs, `outputs/error_review_20.csv`, `outputs/error_review_20_template.csv`, `metrics_report.csv`, `results.md`, and `failure_analysis.md`. The source notebook, checkpoints, logs, and metrics are mapped in `reproducibility/manifests/task2_sentiment_anushka_manifest.json`.

### Task 3 — CycleGAN image translation

Location: `task3_gan/member_anushka/`

The implementation trains an unpaired CycleGAN for both translation directions between Monet paintings and photographs. It uses 9-residual-block generators with 64 base channels, 70×70 PatchGAN discriminators, least-squares adversarial loss, cycle-consistency and identity losses, 256×256 training crops, a 1,000-photo holdout, and a 25-epoch constant-learning-rate phase followed by 25 epochs of linear decay. No pretrained model is used for training or image generation; fixed pretrained networks are used only for evaluation metrics.

The final 50-epoch run achieved course-script FID/MiFID values of 98.75/0.4150 for Monet→Photo and 95.02/0.4045 for Photo→Monet; the submitted file’s averaged values are **96.886 FID** and **0.4097 MiFID**. Key artifacts include `src/Part3_CycleGAN.ipynb`, `src/config.json`, `src/Dockerfile`, `checkpoints/`, `outputs/full_run_20261001_222745/`, `metrics_report.csv`, `full_metrics_report.csv`, `submission.csv`, `evaluate_local.py`, `results.md`, and `failure_analysis.md`. The final run and checkpoint-to-results mapping are recorded in `reproducibility/manifests/task3_gan_anushka_manifest.json`.

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

