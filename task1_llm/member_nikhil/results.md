# Results — Task 1 — LLM from Scratch

**Member:** Nikhil (Member B)

## Architecture
What was built (facts from `src/model.py` and `src/configs/gpt_nikhil_B.yaml`):

- Character-level causal Transformer written from scratch; no `nn.Transformer` or `MultiheadAttention`.
- Token and learned position embeddings, then 2 pre-LayerNorm blocks. Each block has hand-written multi-head self-attention with a causal mask (8 heads, head dimension 32), a feed-forward network (256 → 1,024 → 256) and residual connections, followed by a language-model head (256 → 80).
- Vocabulary of 80 characters from my own `char_to_idx` / `idx_to_char`; context length 128 characters; dropout 0.10.
- 1,653,840 parameters.

**[TODO – Nikhil]** Why this design (2 blocks × 8 heads instead of a deeper, narrower one)? What did you expect it to change, and why did you pick these sizes?

## Hyperparameters
| Setting | Value |
|---|---|
| Data | TinyStories, 208,918 stories; story-level validation hold-out; 100,000 training and 10,000 validation windows |
| Optimizer | AdamW, weight decay 0.01 |
| Learning rate | 2.5e-4, 500 warm-up steps, cosine decay to 1e-5 |
| Batch size / epochs | 64 / 12 (minimum required: 10) |
| Gradient clipping | 1.0 |
| Seed | 1337 |
| Generation | temperature 0.9 and a greedy sample per prompt, up to 400 new characters |
| Hardware | NVIDIA GeForce RTX 5090 |

**[TODO – Nikhil]** How did you arrive at the final values (learning rate, epochs, dropout)? Did you try alternatives?

## Metrics
Full list in `metrics_report.csv` (run `gpt_nikhil_2L8H_B_20261001-0025`). Headline numbers:

| Metric | Value |
|---|---|
| Train / validation cross-entropy | 0.8242 / 0.8440 |
| Validation perplexity | 2.326 |
| Validation bits per character | 1.218 |
| Train / validation top-1 accuracy | 73.69% / 73.20% |
| Generalization gap | 0.0198 |
| Distinct-1/2/3 (temperature 0.9) | 0.313 / 0.771 / 0.921 |
| Repeated 4-gram rate (temperature 0.9 / greedy) | 0.000 / 0.260 |
| Mean / max gradient norm | 0.757 / 6.71 |
| Non-finite steps / loss spikes | 0 / 0 |
| Parameters | 1,653,840 |
| Training / generation tokens per second | 759,303 / 314 |
| Peak accelerator memory / training time | 541 MB / 211 s |

Loss curves: `outputs/gpt_nikhil_2L8H_B_20261001-0025/loss_curves.png`. Gradient norm and learning-rate schedule: `outputs/gpt_nikhil_2L8H_B_20261001-0025/grad_norm_and_lr.png`.

## Notes
- Evidence: raw log and manifest in `reproducibility/raw_logs/nikhil/` and `reproducibility/manifests/nikhil/` (run `task1_gpt_nikhil_2L8H_B_20261001-0025`), best and last checkpoints in `checkpoints/`, executed notebook `src/task1_gpt_nikhil.ipynb`.
- Comparison with the other member is in the team report.
