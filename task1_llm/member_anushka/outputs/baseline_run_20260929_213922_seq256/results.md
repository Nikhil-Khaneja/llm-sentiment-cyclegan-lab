# Task 1 Results (Anushka)

Character-level GPT trained from scratch on TinyStories. No prebuilt Transformer or attention modules are used. Attention, causal masking, blocks, embeddings and the LM head are all written by hand in [src/Part1_LLM.ipynb](src/Part1_LLM.ipynb).

- Run ID: `20260929_213922`
- Checkpoint for all reported numbers: `checkpoints/best_model.pt` (epoch 15, the lowest validation loss)
- Raw log: `reproducibility/raw_logs/task1_llm_anushka/full_training_log_20260929_213922.txt`
- Manifest: `reproducibility/manifests/task1_llm_anushka_manifest.json`

## Data preprocessing
- TinyStories stories are streamed from `roneneldan/TinyStories` (train split, shuffled with seed 3963) until 28.27M characters are collected. They are joined with `\n\n` and cached in `task1_llm/data/`.
- The text is tokenized at the character level. `char_to_idx` / `idx_to_char` are built from the sorted set of characters (vocabulary size 103). They are saved in `data_processed/character_vocabulary.json`.
- The encoded stream is cut into non-overlapping windows of 257 characters. Input = first 256 characters, target = the same window shifted by one.
- Split: the first **100,000** windows are used for training and the next **10,000** for validation. The two sets are contiguous and disjoint, so no validation text appears in training.

## Architecture
Decoder-only, pre-LayerNorm GPT:

```
token_embedding(103→384) + position_embedding(256→384)
→ 6 × [ x + MHA(LN(x)) ;  x + FFN(LN(x)) ]
→ final LayerNorm → Linear(384→103) language-model head
```

- **Multi-head causal self-attention (manual).** A single fused `Linear(d, 3d)` produces Q, K and V, which are split into 6 heads of size 64. Scores are `QKᵀ/√64`. An upper-triangular boolean mask (`torch.triu(..., diagonal=1)`) sets future positions to `-inf` before the softmax, so position *t* can only attend to positions ≤ *t*. Dropout is applied to the attention weights, then an output projection.
- **Feed-forward network:** `Linear(384→1536) → GELU → Linear(1536→384) → Dropout` (4× expansion).
- **Residual connections** around both sub-layers, with **pre-norm** LayerNorm (LN is applied before each sub-layer). This keeps a clean identity path for the gradients and is more stable than post-norm at this depth.
- **Learnable embeddings** for both tokens and absolute positions (256 positions).
- **Parameter count:** 10,825,063

## Hyperparameters (`src/config.json`)

| Hyperparameter | Value | Why |
|---|---|---|
| sequence_length | 256 | Context of about 2–3 sentences, so the model can keep track of names and events. 100K windows × 256 gives 25.6M training characters. |
| n_blocks / n_heads / d_model / d_ff | 6 / 6 / 384 / 1536 | About 10.8M parameters. At about 2.4 tokens per parameter per epoch over 15 epochs, this is big enough to learn spelling and grammar and still generalise (final gap 0.058). |
| dropout | 0.15 | Each character is seen 15 times, so some regularisation is needed. Val tracking train closely shows this is enough. |
| batch_size | 128 (32,768 characters per step) | Gives stable gradient estimates. Fits easily in 32 GB (peak about 5 GB). |
| optimizer | AdamW, β=(0.9, 0.95) | β₂=0.95 reacts faster to gradient-scale changes. This is standard for transformer LMs. |
| learning_rate | 6e-4, cosine decay down to 3e-5 | Peak LR is suited to a model of about 10M parameters. The cosine tail gives the final loss drop seen in epochs 11–15. |
| warmup_steps | 800 (about 1 epoch) | Early Adam statistics are noisy, and warmup avoids large updates at random initialisation. |
| weight_decay | 0.1, applied only to Linear weight matrices | Regularises the projections. Biases, LayerNorm gains and embeddings are not decayed. |
| gradient_clip_norm | 1.0 | Guards against spikes. After warmup, clipping only triggered in the first few hundred steps. |
| epochs | 15 (the brief requires at least 10) | Val loss was still improving slightly at epoch 15. |
| precision | bf16 autocast for matmuls; softmax and loss in fp32 | About 2× throughput on the RTX 5090. bf16 needs no loss scaling. |
| seed | 3963 | For reproducibility. |

## Results (best checkpoint, epoch 15; metrics computed in fp32 over the full train and val sets)

| Metric | Train | Validation |
|---|---|---|
| Cross-entropy loss (nats/char) | 0.5836 | 0.6419 |
| Perplexity | 1.792 | 1.900 |
| Bits per character | 0.842 | 0.926 |
| Top-1 next-character accuracy | 81.23% | 79.67% |

| Metric | Value |
|---|---|
| Generalization gap (val − train loss) | 0.0583 |
| Distinct-1 / 2 / 3 (T=1.0 sample, character n-grams) | 0.091 / 0.448 / 0.699 |
| Repeated 4-gram rate (T=1.0 sample) | 0.187 |
| Gradient norm (pre-clip): mean / max | 0.229 / 2.41 (max is at step 1) |
| NaN / non-finite losses | 0 |
| Loss spikes | None (see the step-level plot) |
| Parameter count | 10,825,063 |
| Training throughput | about 584K tokens/s |
| Generation throughput (batch 1, no KV cache) | 411 tok/s greedy, 547 tok/s sampled |
| Peak GPU memory | 5,110 MB |
| Total training time | 664.7 s (about 11.1 min, 15 epochs × about 44 s) |

![Loss curves](outputs/training_loss_curve.png)

**Reading the curves:** in epochs 1–10 the validation loss is *below* the training loss. That is because the training loss is averaged over each epoch while the weights are still improving, and dropout is on during training but off during evaluation. The curves cross at about epoch 12, and the final gap is small (0.058 nats). This means mild, controlled fitting of the training set, not overfitting: val loss was still decreasing at epoch 15.

Generated samples (greedy, T=0.5, T=1.0, T=1.5) are in `outputs/generated_samples.txt`.

## Failure analysis
See [failure_analysis.md](failure_analysis.md). The three failure types are: phrase repetition, semantic incoherence / entity drift, and broken grammar at high temperature.

## Hardware
- GPU: **NVIDIA GeForce RTX 5090** (32 GB), SJSU GPU lab
- Environment: lab Docker image `pytorch:latest` plus [src/Dockerfile](src/Dockerfile), which upgrades to torch 2.8.0+cu128. The lab image's torch 2.1 has no sm_120 kernels for the RTX 50xx series.
- Python 3.10.13, CUDA 12.8. Full package list: `reproducibility/manifests/task1_llm_anushka_requirements.txt`

## How to reproduce (from the repo root)
```
docker build -t pytorch-task1 task1_llm/member_anushka/src
# smoke test (1 epoch x 20 batches, about 1 min including data download)
docker run --rm --gpus all --shm-size=8g -e TASK1_SMOKE=1 -v "<repo_root>:/app" -w /app/task1_llm/member_anushka/src pytorch-task1 jupyter nbconvert --to notebook --execute --inplace Part1_LLM.ipynb
# full run: drop  -e TASK1_SMOKE=1
```
