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

Raw datasets are not committed (see `.gitignore`). Download them with the shared scripts:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python task1_llm/data/download_tinystories.py   # -> task1_llm/data/tinystories_raw.txt (40k stories)
python task2_sentiment/data/download_yelp.py    # -> task2_sentiment/data/{train,test}.csv
```

## Reproduce a run (smoke test = one command)

After the setup above, from the repo root:

```bash
python task1_llm/member_nikhil/src/train.py --config task1_llm/member_nikhil/src/configs/smoke.yaml
```

This trains Nikhil's Task 1 GPT for one epoch on 2,000 sequences (~15 s on an Apple M-series GPU) and writes the log, manifest, outputs, and checkpoint to the same places a full run does. Device is picked automatically (CUDA > Apple MPS > CPU).

| Run | Command (`--config ...`) | Where results land |
|---|---|---|
| Task 1 full (Nikhil) | `task1_llm/member_nikhil/src/configs/gpt_nikhil.yaml` | `task1_llm/member_nikhil/{metrics_report.csv,outputs/<run_id>/,checkpoints/}` |
| Task 2 smoke (Nikhil) | `task2_sentiment/member_nikhil/src/configs/smoke.yaml` | `task2_sentiment/member_nikhil/outputs/<run_id>/` |
| Task 2 full (Nikhil) | `task2_sentiment/member_nikhil/src/configs/sentiment_nikhil.yaml` | `task2_sentiment/member_nikhil/{metrics_report.csv,outputs/<run_id>/,checkpoints/}` |
| Task 3 smoke (Nikhil) | `task3_gan/member_nikhil/src/configs/smoke.yaml` | `task3_gan/member_nikhil/outputs/<run_id>/` |
| Task 3 full (Nikhil) | `task3_gan/member_nikhil/src/configs/cyclegan_nikhil.yaml` | `task3_gan/member_nikhil/{full_metrics_report.csv,images.zip,outputs/,checkpoints/}` |

Use the `train.py` inside the same folder as the config. Task 3 needs the Kaggle images first (`task3_gan/data/README.md`); an interrupted Task 3 run continues with `--resume <checkpoint>`, and `task3_gan/member_nikhil/evaluate_local.py --ckpt <checkpoint> [--submission]` re-scores any checkpoint. Every run also writes an unedited raw log to `reproducibility/raw_logs/<member>/` and a manifest (package versions, hardware, git commit, config, checkpoint ↔ result mapping) to `reproducibility/manifests/<member>/`. The executed notebooks in each `src/` folder run the same `train.run(config)` and show the outputs inline.
