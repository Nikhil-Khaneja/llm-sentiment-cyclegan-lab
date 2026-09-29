# Task 1 Results (Anushka)

Character-level GPT trained from scratch on TinyStories. No prebuilt Transformer or attention modules are used. Attention, causal masking, blocks, embeddings and the LM head are all written by hand in [src/Part1_LLM.ipynb](src/Part1_LLM.ipynb).

- **Final run:** `20260929_225852` (sequence length 512, 20 epochs)
- **Checkpoint for every reported number:** `checkpoints/best_model.pt` (epoch 20, the lowest validation loss). `checkpoints/last_model.pt` is the same epoch.
- **Raw log:** `reproducibility/raw_logs/task1_llm_anushka/full_training_log_20260929_225852.txt`
- **Manifest:** `reproducibility/manifests/task1_llm_anushka_manifest.json`
- **Baseline for comparison:** run `20260929_213922` (sequence length 256, 15 epochs). Its files are in [outputs/baseline_run_20260929_213922_seq256/](outputs/baseline_run_20260929_213922_seq256/) and its checkpoint is `checkpoints/baseline_seq256_best_model.pt`.

## Data preprocessing
- TinyStories stories are streamed from `roneneldan/TinyStories` (train split, shuffled with seed 3963) until 56.4M characters are collected. They are joined with `\n\n` and cached in `task1_llm/data/`, which is gitignored and re-downloaded with the same seed.
- The text is tokenized at the character level. `char_to_idx` / `idx_to_char` are built from the sorted set of characters (vocabulary size 107). They are saved in `data_processed/character_vocabulary.json`.
- The encoded stream is cut into non-overlapping windows of 513 characters. Input = first 512 characters, target = the same window shifted by one.
- Split: the first **100,000** windows are used for training and the next **10,000** for validation. The two sets are contiguous and disjoint, so no validation text appears in training.

## Architecture
Decoder-only, pre-LayerNorm GPT:

```
token_embedding(107→384) + position_embedding(512→384)
→ 6 × [ x + MHA(LN(x)) ;  x + FFN(LN(x)) ]
→ final LayerNorm → Linear(384→107) language-model head
```

- **Multi-head causal self-attention (manual).** A single fused `Linear(d, 3d)` produces Q, K and V, which are split into 6 heads of size 64. Scores are `QKᵀ/√64`. An upper-triangular boolean mask (`torch.triu(..., diagonal=1)`) sets future positions to `-inf` before the softmax, so position *t* can only attend to positions ≤ *t*. Dropout is applied to the attention weights, then an output projection.
- **Feed-forward network:** `Linear(384→1536) → GELU → Linear(1536→384) → Dropout` (4× expansion).
- **Residual connections** around both sub-layers, with **pre-norm** LayerNorm (LN is applied before each sub-layer). This keeps a clean identity path for the gradients and is more stable than post-norm at this depth.
- **Learnable embeddings** for both tokens and absolute positions (512 positions).
- **Parameter count:** 10,926,443

## Hyperparameters (`src/config.json`)

| Hyperparameter | Value | Why |
|---|---|---|
| sequence_length | 512 (baseline: 256) | 100K windows × 512 gives 51.2M training characters, twice the baseline, while keeping the required 100K / 10K sequence counts. The context covers about 100 words, so names and objects from earlier in a story are still visible. |
| n_blocks / n_heads / d_model / d_ff | 6 / 6 / 384 / 1536 | About 10.9M parameters. Big enough to learn spelling and grammar. The final gap (0.040) shows it still generalises. |
| dropout | 0.15 | Each character is seen 20 times, so some regularisation is needed. Val tracking train closely shows this is enough. |
| batch_size | 128 (65,536 characters per step) | Gives stable gradient estimates. Peak memory 14.8 GB of 32 GB. |
| optimizer | AdamW, β=(0.9, 0.95) | β₂=0.95 reacts faster to gradient-scale changes. This is standard for transformer LMs. |
| learning_rate | 6e-4, cosine decay down to 3e-5 | Peak LR is suited to a model of about 10M parameters. The cosine tail gives the slow final improvement in epochs 15–20. |
| warmup_steps | 800 (about 1 epoch) | Early Adam statistics are noisy, and warmup avoids large updates at random initialisation. |
| weight_decay | 0.1, applied only to Linear weight matrices | Regularises the projections. Biases, LayerNorm gains and embeddings are not decayed. |
| gradient_clip_norm | 1.0 | Guards against spikes. The pre-clip norm stays around 0.16–0.34 after warmup. |
| epochs | 20 (the brief requires at least 10; baseline 15) | The baseline's val loss was still falling at its last epoch, so this run trains longer. |
| precision | bf16 autocast for matmuls; softmax and loss in fp32 | About 2× throughput on the RTX 5090. bf16 needs no loss scaling. |
| seed | 3963 | For reproducibility. The interrupted run (below) matched this run's epoch 1–6 losses exactly. |

## Results (best checkpoint, epoch 20; metrics computed in fp32 over the full train and val sets)

| Metric | Train | Validation |
|---|---|---|
| Cross-entropy loss (nats/char) | 0.5252 | 0.5651 |
| Perplexity | 1.691 | 1.760 |
| Bits per character | 0.758 | 0.815 |
| Top-1 next-character accuracy | 83.11% | 82.01% |

| Metric | Value |
|---|---|
| Generalization gap (val − train loss) | 0.0399 |
| Distinct-1 / 2 / 3 (T=1.0 sample, character n-grams) | 0.107 / 0.489 / 0.778 |
| Repeated 4-gram rate (T=1.0 sample) | 0.102 |
| Gradient norm (pre-clip): mean / max | 0.201 / 2.26 (max is at step 1) |
| NaN / non-finite losses | 0 |
| Loss spikes | None (see the step-level plot) |
| Parameter count | 10,926,443 |
| Training throughput | about 399K tokens/s |
| Generation throughput (batch 1, no KV cache) | 597 tok/s greedy, 610 tok/s sampled |
| Peak GPU memory | 14,805 MB |
| Total training time | 2,572.7 s (about 42.9 min, 20 epochs × about 128 s) |

![Loss curves](outputs/training_loss_curve.png)

**Reading the curves:** in epochs 1–17 the validation loss is *below* the training loss. That is because the training loss is averaged over each epoch while the weights are still improving, and dropout is on during training but off during evaluation. The curves meet at about epoch 18, and the final gap is small (0.040 nats). Val loss was still decreasing at epoch 20, so the model is not overfitting.

Generated samples (greedy, T=0.5, T=1.0, T=1.5) are in `outputs/generated_samples.txt`.

## Comparison with the baseline run

| | Baseline `20260929_213922` | Final `20260929_225852` |
|---|---|---|
| sequence_length / epochs | 256 / 15 | 512 / 20 |
| Training characters per epoch | 25.6M | 51.2M |
| Parameters | 10,825,063 | 10,926,443 (larger position table) |
| Val loss / perplexity | 0.6419 / 1.900 | **0.5651 / 1.760** |
| Val bits per character | 0.926 | **0.815** |
| Val top-1 accuracy | 79.67% | **82.01%** |
| Generalization gap | 0.058 | **0.040** |
| Repeated 4-gram rate (T=1.0) | 0.187 | **0.102** |
| Distinct-3 (T=1.0) | 0.699 | **0.778** |
| Training time / peak memory | 11.1 min / 5.1 GB | 42.9 min / 14.8 GB |

All other hyperparameters are identical, so the gain comes from the longer context, twice as much training text, and more epochs. The generalization gap also shrank, which fits the idea that more distinct data matters more than a bigger model at this scale. The two runs use different validation windows (the window length changed), so the comparison is indicative rather than exact. Both validation sets are unseen TinyStories text drawn with the same seed.

**Interrupted run:** a first attempt at this config (`20260929_220320`) was killed after epoch 6 when the lab machine rebooted. Its raw log is kept unedited in `reproducibility/raw_logs/task1_llm_anushka/`. Epochs 1–6 of that log match the final run exactly.

## Failure analysis
See [failure_analysis.md](failure_analysis.md). The three failure types are: repetition / circular actions, loss of coherence with speaker and entity confusion, and broken spelling and grammar at high temperature.

## Hardware
- GPU: **NVIDIA GeForce RTX 5090** (32 GB), SJSU GPU lab
- Environment: lab Docker image `pytorch:latest` plus [src/Dockerfile](src/Dockerfile), which upgrades to torch 2.8.0+cu128. The lab image's torch 2.1 has no sm_120 kernels for the RTX 50xx series.
- Python 3.10.13, CUDA 12.8. Full package list: `reproducibility/manifests/task1_llm_anushka_requirements.txt`

## How to reproduce (from the repo root)
```
docker build -t pytorch-task1 task1_llm/member_anushka/src
# smoke test (1 epoch x 20 batches, a few minutes including data download)
docker run --rm --gpus all --shm-size=8g -e TASK1_SMOKE=1 -v "<repo_root>:/app" -w /app/task1_llm/member_anushka/src pytorch-task1 jupyter nbconvert --to notebook --execute --inplace Part1_LLM.ipynb
# full run: drop  -e TASK1_SMOKE=1
```
