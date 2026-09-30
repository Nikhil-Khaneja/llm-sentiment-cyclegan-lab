# GPU machine handoff (Nikhil)

Context for a Claude Code session running on the CUDA GPU box. Read this, `CONTEXT.md` and `README.md` before doing anything.

## Where things stand (as of 2026-09-29, branch `nikhil`, commit 9e2de31)

| Task | Status | What's left |
|---|---|---|
| Task 1: char-level GPT (TinyStories) | **Full run done** on Apple M5 (MPS): `gpt_nikhil_4L4H_20260925-1737`. Checkpoints are in LFS; the metrics are in `task1_llm/member_nikhil/metrics_report.csv`. | Nothing to train. **Don't re-run** unless Nikhil asks. |
| Task 2: Yelp sentiment (A1 mean-pool, A2 CNN k345, A3 BiGRU) | Only the smoke tests have run. | **Full run** with `sentiment_nikhil.yaml` (all 560k train, 8 epochs × 3 models). |
| Task 3: CycleGAN Monet↔Photo | Smoke test plus a `quick` pilot on public monet2photo. | Download the **Kaggle data**, then do the **full run** with `cyclegan_nikhil.yaml` (50 epochs), then produce `images.zip` for the Kaggle submission. |

`results.md` / `failure_analysis.md` are still templates. Nikhil writes those himself (viva). Don't fill them in unless he asks.

## 1. Setup

```bash
git clone https://github.com/Nikhil-Khaneja/llm-sentiment-cyclegan-lab.git   # if not already cloned
cd llm-sentiment-cyclegan-lab
git checkout nikhil && git pull origin nikhil
git lfs install && git lfs pull

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

`torch==2.8.0` from PyPI ships with CUDA on Linux. If `cuda.is_available()` is False, install the wheel that matches the driver (`nvidia-smi`), e.g. `pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126`. Keep the same torch/torchvision versions.

Device selection is automatic (`device: auto` → CUDA > MPS > CPU). You shouldn't need to edit any code for the GPU.

## 2. Data (not committed; each machine downloads its own)

```bash
python task1_llm/data/download_tinystories.py     # only needed for smoke-testing Task 1
python task2_sentiment/data/download_yelp.py      # -> task2_sentiment/data/{train,test}.csv
```

For Task 3 you need the class Kaggle competition data. See `task3_gan/data/README.md`. **Ask Nikhil for the competition slug and to set up `~/.kaggle/kaggle.json`.** Don't guess the slug. When it's done you should see about 300 files in `task3_gan/data/monet_jpg/` and about 7k in `task3_gan/data/photo_jpg/`.

## 3. Runs, in order

Run long jobs inside `tmux` (or `nohup ... &`) so an SSH/VS Code disconnect doesn't kill them.

1. **Smoke tests on CUDA** (a few minutes total). In each new manifest, confirm the device reported is CUDA:
   ```bash
   python task2_sentiment/member_nikhil/src/train.py --config task2_sentiment/member_nikhil/src/configs/smoke.yaml
   python task3_gan/member_nikhil/src/train.py      --config task3_gan/member_nikhil/src/configs/smoke.yaml
   ```
2. **Task 2 full run:**
   ```bash
   python task2_sentiment/member_nikhil/src/train.py --config task2_sentiment/member_nikhil/src/configs/sentiment_nikhil.yaml
   ```
3. **Task 3 full run** (the longest one; checkpoints are saved every 5 epochs):
   ```bash
   python task3_gan/member_nikhil/src/train.py --config task3_gan/member_nikhil/src/configs/cyclegan_nikhil.yaml
   # if interrupted:
   python task3_gan/member_nikhil/src/train.py --config task3_gan/member_nikhil/src/configs/cyclegan_nikhil.yaml --resume <checkpoint>
   # re-score any checkpoint / rebuild the submission:
   python task3_gan/member_nikhil/evaluate_local.py --ckpt <checkpoint> --submission
   ```

Steps 2 and 3 can run at the same time if GPU memory allows. Check with `nvidia-smi`.

## 4. Rules (from the lab spec; don't break these)

- **Don't change the hyperparameters or architecture in the configs** without asking Nikhil first. They were chosen deliberately to differ from Anushka's models. Lowering `num_workers` to fix a dataloader problem is fine; tell him if you do it.
- Only touch `task*/member_nikhil/**` and `reproducibility/*/nikhil/`. Never touch `member_anushka/`. Never commit to `main`.
- No pretrained models or embeddings anywhere. The Kaggle images must be the CycleGAN's direct output.
- No hard-coded absolute paths or secrets in committed files. Runs are config-driven only.
- Raw logs in `reproducibility/raw_logs/nikhil/` are evidence. Commit them **unedited**.

## 5. After each run: commit and push to `nikhil`

```bash
git add task2_sentiment/member_nikhil reproducibility/raw_logs/nikhil reproducibility/manifests/nikhil
git commit -m "Task 2 (Nikhil): full run on <GPU name>"
git push origin nikhil
```

Checkpoints (`*.pt`) go through LFS automatically (see `.gitattributes`). Don't commit datasets, `images.zip`, or `submission.csv`; they're gitignored. `images.zip` gets uploaded to Kaggle by hand.

## 6. Report back to Nikhil

For each run, report the GPU model, wall-clock time, peak memory, the headline metrics (Task 2: accuracy/F1 per model; Task 3: FID/KID etc. from `full_metrics_report.csv`), anything that looked wrong (NaN loss, collapse, OOM), and the commit hash you pushed.
