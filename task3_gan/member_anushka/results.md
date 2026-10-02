# Results — Task 3 — CycleGAN Style Transfer

**Member:** anushka

<!-- Draft written from the evidence of run 20261001_222745. To be rewritten in my own words before submission. -->

Run `20261001_222745`, 50 epochs, final checkpoint `checkpoints/cyclegan_20261001_222745_epoch050.pt`.
Domain A = Monet, domain B = photo. `A2B` = Monet -> Photo, `B2A` = Photo -> Monet.

## Architecture

- **Generators (x2, 11.38 M parameters each):** 7x7 stem with 64 channels, two stride-2 convolutions (to 256 channels at 64x64), 9 residual blocks, two transposed convolutions back to 256x256, 7x7 output with tanh. Instance normalization and reflection padding throughout.
- **Discriminators (x2, 2.76 M parameters each):** 70x70 PatchGAN with 64 base channels. It scores overlapping patches instead of the whole image, so it judges local texture, which is what separates a painting from a photo.
- **Losses:** least-squares adversarial loss, cycle-consistency L1 (weight 10), identity L1 (weight 5).
- No pretrained network is used in training or in producing the translated images. InceptionV3 and AlexNet-LPIPS are used only to measure the outputs.

## Hyperparameters

| Setting | Value | Reason |
|---|---|---|
| Residual blocks | 9 | The CycleGAN paper uses 9 blocks for 256x256 images; more blocks give a larger receptive field at the bottleneck. |
| Generator base channels | 64 | Paper setting. The team plan had 32 for me; I raised it for capacity at 256px. |
| Image size | resize to 286, random crop 256, random flip | The dataset and the course evaluation script work at 256x256. Training at 128 would mean every submitted image is upsampled before scoring. |
| Batch size | 1 | Paper setting; instance normalization works per image. |
| Adversarial loss | LSGAN (MSE) | More stable gradients than the log loss when the discriminator is confident. |
| Cycle weight / identity weight | 10 / 5 | Paper values for Monet <-> photo; the identity term keeps the colour palette of the input. |
| Optimizer | Adam, lr 2e-4, betas (0.5, 0.999) | Paper values. |
| Schedule | 25 epochs constant, 25 epochs linear decay | Team plan; the decay phase steadies the generators at the end. |
| History pool | 50 generated images | The discriminators see older fakes as well, which reduces oscillation. |
| Weight init | N(0, 0.02) | Paper value. |
| Photo holdout | 1,000 photos, seed 3963 | Test set for Photo -> Monet; never trained on. |
| torch.compile | on, training step only | Same model and arithmetic, about 1.4x faster steps on this machine (14 vs 10 steps/s). |

**Difference from my teammate:** Nikhil uses 6 residual blocks, batch size 2 and trains at 128px; I use 9 blocks, batch size 1 and train at 256px.

**Settings that are not in the team plan:** 256px training (plan: 128), 64 generator channels (plan: 32), history pool 50, weight init N(0, 0.02), resize-then-crop ratio, 1,000-photo holdout, discriminator width 64, torch.compile.

## Metrics

All rows are in [full_metrics_report.csv](full_metrics_report.csv) (same rows in `metrics_report.csv`). Raw log: `reproducibility/raw_logs/task3_gan_anushka/full_training_log_20261001_222745.txt`.

| Metric | Monet -> Photo (A2B) | Photo -> Monet (B2A) |
|---|---|---|
| FID (300 vs 1,000 images) | 81.16 | 88.58 |
| FID of untranslated inputs (baseline) | 122.71 | 122.71 |
| KID (mean ± std) | 0.0163 ± 0.0007 | 0.0143 ± 0.0010 |
| Precision / recall | 0.740 / 0.469 | 0.504 / 0.643 |
| Density / coverage | 0.915 / 0.554 | 0.448 / 0.897 |
| Cycle-reconstruction L1 ([0, 1] scale) | 0.0345 | 0.0467 |
| Identity L1 | 0.0253 | 0.0348 |
| LPIPS input vs translation | 0.338 | 0.385 |
| LPIPS input vs reconstruction | 0.233 | 0.174 |
| Content cosine, input vs translation | 0.792 | 0.741 |
| Course script FID | 98.75 | 95.02 |
| Course script MiFID | 0.4150 | 0.4045 |

- **Kaggle submission file (course script, average of both directions):** FID 96.886, MiFID 0.4097.
- **Kaggle public / private score and rank:** pending.
- **Human audit:** pending (two raters).
- **Cost:** 28.29 M parameters in total, 6 h 15 min of training (22,507 s), 26.8 images/s, peak GPU memory 9,974 MB in training and 11,898 MB in evaluation, peak process RAM 5,738 MB, RTX 4090.

The two FID rows differ because the course script compares 300 against 300 images with torchvision's InceptionV3, while the notebook's FID uses the standard FID InceptionV3 weights and all 1,000 held-out photos.

## Training behaviour

Evidence: [loss_curves.png](outputs/full_run_20261001_222745/loss_curves.png), [fid_by_checkpoint.png](outputs/full_run_20261001_222745/fid_by_checkpoint.png), per-epoch sample grids in `outputs/full_run_20261001_222745/samples/`.

- **Stability:** 0 non-finite steps in 301,900. The mean generator gradient norm stays near 21 after epoch 10 and the mean discriminator norm falls from 22.6 to 6.4. The largest single-step generator norm is 636 (epoch 3).
- **Convergence:** cycle L1 falls in every epoch (A: 0.221 -> 0.058, B: 0.239 -> 0.068 on the [-1, 1] scale) and identity L1 falls with it.
- **Discriminators pull ahead:** D_A loss falls from 0.239 to 0.053 and D_B from 0.239 to 0.121, while the generators' adversarial losses rise (B2A: 0.42 -> 0.82, A2B: 0.43 -> 0.59). The Monet discriminator sees only 300 paintings, so it is the stronger case.
- **Quality still improved:** FID per checkpoint goes from 103.0 / 107.7 (A2B / B2A) at epoch 5 to 81.2 / 88.6 at epoch 50. Epoch 50 is the best checkpoint in both directions. B2A is not monotonic (it rises at epochs 25 and 35).

## Cycle-consistency check

On 64 images per direction ([cycle_check_A2B.jpg](outputs/full_run_20261001_222745/cycle_check_A2B.jpg), [cycle_check_B2A.jpg](outputs/full_run_20261001_222745/cycle_check_B2A.jpg)), mean L1 on the [0, 1] scale:

| | Own reconstruction | Translation | Reconstruction of a different image |
|---|---|---|---|
| A2B | 0.034 | 0.136 | 0.240 |
| B2A | 0.045 | 0.154 | 0.269 |

The round trip returns to the specific input (about 6 to 7 times closer than to another image), and the translation itself changes the image about 3 to 4 times more than the round-trip error, so the generators are not just copying their input.

## Notes

- Monet -> Photo is evaluated on the 300 training paintings because there are no other Monet images.
- Two earlier full runs exist only as raw logs: `20261001_202102` (128px, stopped at epoch 1 on another machine) and `20261001_220908` (256px without torch.compile, stopped in epoch 1 because it would have taken 8.5 hours).
- Shortcomings are in [failure_analysis.md](failure_analysis.md).
