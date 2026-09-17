# llm-sentiment-cyclegan-lab

DATA266 Lab 1 (Fall 2026) — two independently-built model suites per member, compared as a team:

1. **Task 1 — GPT-style LLM from scratch** on TinyStories (character-level, hand-written attention).
2. **Task 2 — Yelp Polarity sentiment classification** (3 models per member: baseline + 2 experiments, learned embeddings only).
3. **Task 3 — CycleGAN style transfer** between Monet paintings and photos (Kaggle-scored).

Team members: **Nikhil** ([@Nikhil-Khaneja](https://github.com/Nikhil-Khaneja)) and **Anushka**.

See [`CONTEXT.md`](./CONTEXT.md) for the full workflow: branching model, folder ownership, and how to set up and submit your work.

## Repo layout

```
task1_llm/
  data/                    ← shared TinyStories dataset (not committed — see CONTEXT.md)
  member_nikhil/
  member_anushka/
task2_sentiment/
  data/                    ← shared Yelp Polarity dataset (not committed)
  member_nikhil/
  member_anushka/
task3_gan/
  data/monet_jpg/, photo_jpg/   ← shared Monet/Photo images (not committed)
  member_nikhil/
  member_anushka/
reproducibility/
  manifests/               ← env/package versions per run
  raw_logs/                ← unedited training logs (evidence trail)
report/
  DATA266_Lab1_Report_Team_[Team Number].pdf   ← combined final report (added at the end)
docs/assignment/           ← original lab spec PDFs for reference
```

Each `member_<name>/` folder under a task contains:

```
src/                ← your code (executed .ipynb with outputs visible)
data_processed/     ← your own preprocessing output (if applicable)
checkpoints/        ← your trained model weights
outputs/            ← generated results (samples, predictions, plots, confusion matrices)
metrics_report.csv  ← every required metric for this task, in one file
failure_analysis.md ← required failure/error case write-up
results.md          ← architecture + hyperparameter justification
```

## Setup

```bash
git clone https://github.com/Nikhil-Khaneja/llm-sentiment-cyclegan-lab.git
cd llm-sentiment-cyclegan-lab
git checkout <your-branch>   # nikhil or anushka
```

Model weights are tracked with [Git LFS](https://git-lfs.com/). Run `git lfs install` once per machine before committing any `.pt`/`.pth`/`.ckpt` file.

Raw datasets are not committed (see `.gitignore`) — download them into the relevant `data/` folder per the links in `docs/assignment/`.
