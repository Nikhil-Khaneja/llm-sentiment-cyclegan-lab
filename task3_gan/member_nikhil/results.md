# Results — Task 3 — CycleGAN Style Transfer

**Member:** Nikhil (Member B)

## Architecture
What was built (facts from `src/models.py` and `src/configs/cyclegan_nikhil_B.yaml`). Domain A = Monet paintings, B = photographs.

- Two ResNet generators (A→B and B→A): 7×7 stem, two stride-2 downsampling convolutions, 9 residual blocks, two transposed convolutions, tanh output, instance normalization, reflection padding; 32 base channels, 2,850,563 parameters each.
- Two 70×70 PatchGAN discriminators, 64 base channels, 2,764,737 parameters each. Total 11,230,600 parameters.
- Losses: least-squares GAN, cycle-consistency L1 (weight 10), identity L1 (weight 5). A history pool of 50 generated images feeds the discriminators.

**[TODO – Nikhil]** Why 9 blocks with 32 channels (compared with fewer, wider blocks)? What did you expect from this capacity trade-off?

## Hyperparameters
| Setting | Value |
|---|---|
| Training resolution | 128×128 (resize to 143, random crop, horizontal flip) |
| Optimizer | Adam, lr 2e-4, β = (0.5, 0.999) |
| Schedule | 25 constant epochs + 25 linear-decay epochs (50 total) |
| Batch size | 1 |
| Held-out data | 1,000 photos never used in training (evaluation and human audit); all 300 Monet paintings used for training |
| Checkpoints | every 5 epochs |
| Seed | 7 |
| Hardware | NVIDIA GeForce RTX 5090 |

**[TODO – Nikhil]** How did you choose these values, and what did you observe when training?

## Metrics
Full list in `full_metrics_report.csv` (run `cyclegan_nikhil_r9c32_B_20261001-0022`); evaluation uses 300 Monet and 1,000 held-out photos.

| Direction | FID | KID | Precision | Recall | Density | Coverage | Cycle L1 | LPIPS | Content cosine |
|---|---|---|---|---|---|---|---|---|---|
| A→B (Monet → photo) | 88.71 | 0.0272 | 0.753 | 0.356 | 0.727 | 0.448 | 0.0477 | 0.388 | 0.698 |
| B→A (photo → Monet) | 89.87 | 0.0175 | 0.571 | 0.623 | 0.554 | 0.920 | 0.0551 | 0.344 | 0.689 |

FID of the untranslated inputs against the target domain (the do-nothing reference): 126.01.

Efficiency and stability: training time 27,580 s, 22.70 images/s, peak accelerator memory 464 MB, 0 non-finite steps. Loss curves: `outputs/cyclegan_nikhil_r9c32_B_20261001-0022/loss_curves.png`; per-epoch values in `train_history.csv`.

Leaderboard files (course evaluation script, first 300 real vs 300 generated images, both directions averaged): FID 108.032, MiFID 0.4129 in `submission.csv` (per-direction values in `submission_course_script.csv`). Submitted to Kaggle on 6 Oct 2026 under team `PairProgramming_Team_42`: public score −54.2222. The team's best score is Anushka's entry, −48.6479 (rank 21 on 6 Oct 2026).

## Notes
- Evidence: raw log and manifest under `reproducibility/*/nikhil/` (run `task3_cyclegan_nikhil_r9c32_B_20261001-0022`), checkpoints in `checkpoints/`, predictions in `outputs/pred_A2B/` and `outputs/pred_B2A/`, executed notebook `src/task3_cyclegan_nikhil.ipynb`.
- `evaluate_local.py` is the course script (writes `submission.csv`); `evaluate_full.py` computes the other metrics (writes `full_metrics_report.csv`). `kaggle_score.py` is an unofficial estimate that does not match the course script.
- Human audit: blinded sheets are in `outputs/cyclegan_nikhil_r9c32_B_20261001-0022/human_audit/`; ratings and agreement are still to be added.
