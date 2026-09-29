# Task 1 — Character-level GPT from scratch (Anushka)

A decoder-only GPT trained from scratch on **TinyStories** at the character level. Multi-head causal self-attention, LayerNorm, the feed-forward network, residual connections, embeddings and the LM head are all written by hand in PyTorch. **No `nn.Transformer`, `nn.MultiheadAttention` or other prebuilt attention module is used.**

| | |
|---|---|
| Final run | `20260929_225852` |
| Model | 6 blocks · 6 heads · d_model 384 · d_ff 1536 · context 512 characters · **10,926,443 parameters** |
| Data | 100,000 train / 10,000 validation sequences of 512 characters (vocabulary: 107 characters) |
| Training | 20 epochs · AdamW · 800-step warmup + cosine decay · bf16 mixed precision |
| Validation | **loss 0.565 · perplexity 1.76 · 0.815 bits/char · 82.0% top-1 next-character accuracy** |
| Hardware | NVIDIA GeForce RTX 5090 (32 GB), SJSU GPU lab · 42.9 min · 14.8 GB peak memory |

The architecture, the reason for each hyperparameter, the full metrics and the comparison with the baseline run are in **[results.md](results.md)**. The three failure cases are in **[failure_analysis.md](failure_analysis.md)**.

## Folder contents

```
member_anushka/
├── README.md                  ← this file
├── results.md                 ← architecture, hyperparameter justification, all metrics, baseline comparison, hardware
├── failure_analysis.md        ← 3 failure cases with verbatim generated snippets
├── metrics_report.csv         ← every Task 1 metric for the final run (one row per metric)
├── src/
│   ├── Part1_LLM.ipynb        ← full pipeline, executed with outputs visible
│   ├── config.json            ← hyperparameters of the final run (written by the notebook)
│   └── Dockerfile             ← lab PyTorch image + torch 2.8/CUDA 12.8 (needed for RTX 50xx GPUs)
├── data_processed/
│   └── character_vocabulary.json   ← char_to_idx / idx_to_char for the final run
├── checkpoints/               ← Git LFS
│   ├── best_model.pt          ← final run, epoch 20 (source of every reported number)
│   ├── last_model.pt          ← final run, last epoch (also 20)
│   └── baseline_seq256_best_model.pt   ← baseline run 20260929_213922
└── outputs/
    ├── metrics.json                  ← same numbers as metrics_report.csv
    ├── training_loss_curve.png       ← train/val loss per epoch + step-level loss and gradient norm
    ├── generated_samples.txt         ← greedy, T=0.5, T=1.0, T=1.5 samples
    ├── generated_sample_greedy.txt
    ├── history_20260929_225852.json  ← per-epoch / per-step history of the final run
    ├── history_20260929_213922.json  ← baseline run history
    ├── history_20260929_213834.json  ← smoke-test history
    └── baseline_run_20260929_213922_seq256/   ← baseline outputs, kept for comparison (see its README)
```

## Runs

| Run ID | What | Result | Raw log |
|---|---|---|---|
| `20260929_213834` | Smoke test (1 epoch × 20 batches) | Pipeline check only | `smoke_training_log_20260929_213834.txt` |
| `20260929_213922` | Baseline: context 256, 15 epochs | val loss 0.642 | `full_training_log_20260929_213922.txt` |
| `20260929_220320` | Context 512 attempt, **interrupted** (lab machine rebooted after epoch 6) | Not used | `full_training_log_20260929_220320.txt` |
| `20260929_225852` | **Final**: context 512, 20 epochs | **val loss 0.565** | `full_training_log_20260929_225852.txt` |

Raw logs are in `reproducibility/raw_logs/task1_llm_anushka/` and are unedited. The environment manifest (Python, torch, CUDA, GPU, run → checkpoint → results mapping) and the full `pip freeze` are in `reproducibility/manifests/task1_llm_anushka_*`.

## Data

- **Source:** [`roneneldan/TinyStories`](https://huggingface.co/datasets/roneneldan/TinyStories), train split, streamed and shuffled with seed 3963 (buffer size 10,000).
- **Raw corpus:** `task1_llm/data/tinystories_corpus_seed3963.txt` (56,520,969 bytes, SHA-256 `190af27595f2ecc71fa1ccdd36a2f0aed1c23d9dc1cc66039b36c9ecf055838c`). It is committed so the exact training text is available. If the file is missing, the notebook downloads the same corpus again automatically.
- **Preprocessing:** character-level tokenization → non-overlapping windows of 513 characters (input = first 512, target = shifted by one) → the first 100,000 windows are used for training and the next 10,000 for validation.

## How to reproduce

Requirements: Docker with an NVIDIA GPU, plus the lab image `pytorch:latest` (`docker pull gdevakumar/pytorch && docker tag gdevakumar/pytorch pytorch`). Run from the repo root:

```bash
# 1. Build the environment (adds torch 2.8 + CUDA 12.8, datasets, matplotlib, nbconvert)
docker build -t pytorch-task1 task1_llm/member_anushka/src

# 2. Smoke test: 1 epoch x 20 batches, takes a few minutes
docker run --rm --gpus all --shm-size=8g -e TASK1_SMOKE=1 -v "<repo_root>:/app" -w /app/task1_llm/member_anushka/src pytorch-task1 jupyter nbconvert --to notebook --execute --inplace Part1_LLM.ipynb

# 3. Full run (about 43 min on an RTX 5090): same command without  -e TASK1_SMOKE=1
```

Replace `<repo_root>` with the path of your clone. Each run writes its outputs to this folder and a new raw log to `reproducibility/raw_logs/task1_llm_anushka/`. It needs no other paths or credentials.

## Load the trained model

```python
import torch
ckpt = torch.load('task1_llm/member_anushka/checkpoints/best_model.pt', map_location='cpu', weights_only=False)
# keys: model_state_dict, config, char_to_idx, idx_to_char, history, epoch, val_loss, run_id
```

Build `GPTFromScratch(len(ckpt['char_to_idx']), ckpt['config'])` from the notebook, load `model_state_dict`, and generate with `generate_text(...)`.

## References

- Vaswani et al., *Attention Is All You Need*, NeurIPS 2017.
- Eldan & Li, *TinyStories: How Small Can Language Models Be and Still Speak Coherent English?*, 2023.
