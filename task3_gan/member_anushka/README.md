# Task 3 — CycleGAN Monet <-> Photo (Anushka)

A CycleGAN trained from scratch for unpaired image translation between Monet paintings (domain A) and photos (domain B). Two ResNet generators and two 70x70 PatchGAN discriminators, trained with least-squares adversarial loss, cycle-consistency loss and identity loss. **No pretrained model is used for training or for producing the translated images.** Pretrained InceptionV3 and AlexNet-LPIPS are used only to compute the evaluation metrics.

The architecture, the reason for each hyperparameter and the full metrics are in **[results.md](results.md)**. The failure cases are in **[failure_analysis.md](failure_analysis.md)**.

## Folder contents

```
member_anushka/
├── README.md                  ← this file
├── results.md                 ← architecture, hyperparameter justification, metrics, hardware
├── failure_analysis.md        ← failure cases with the images
├── full_metrics_report.csv    ← every Task 3 metric, both directions (one row per metric)
├── metrics_report.csv         ← same rows (per-member file name used in every task)
├── submission.csv             ← FID / MiFID from the course script (generated, gitignored)
├── evaluate_local.py          ← course evaluation script with repo-relative paths
├── src/
│   ├── Part3_CycleGAN.ipynb   ← full pipeline, executed with outputs visible
│   ├── config.json            ← every hyperparameter of the run
│   └── Dockerfile             ← PyTorch 2.8 / CUDA 12.8 + dependencies
├── checkpoints/               ← Git LFS
└── outputs/
    ├── pred_A2B/              ← Monet -> Photo translations of the 300 Monet paintings
    ├── pred_B2A/              ← Photo -> Monet translations of the 1,000 held-out photos
    ├── full_run_<run_id>/     ← loss curves, per-epoch sample grids, cycle check, failure candidates,
    │                             per-image scores, human audit sheets, history
    └── smoke_run_<run_id>/    ← same files for a smoke test
```

Raw logs are in `reproducibility/raw_logs/task3_gan_anushka/` and are unedited. The environment manifest and the full `pip freeze` are in `reproducibility/manifests/task3_gan_anushka_*`.

## Data

- `task3_gan/data/monet_jpg/` (300 images) and `task3_gan/data/photo_jpg/` (7,038 images), 256x256 JPEG. Not committed; unzip the course `dataset.zip` so that these two folders exist.
- Split: photos are shuffled with seed 3963 and 1,000 are held out for evaluation (listed in `outputs/<run>/photo_holdout_files.txt`). All 300 Monet paintings are used for training.
- Preprocessing: training images are resized to 286x286, then randomly cropped to 256x256 and randomly flipped at every step. Evaluation images are used at 256x256, the native size of the dataset.

## How to reproduce

Requirements: Docker with an NVIDIA GPU. Run from the repo root:

```bash
# 1. Build the environment
docker build -t pytorch-task3 task3_gan/member_anushka/src

# 2. Smoke test: 2 epochs on 96 photos / 32 paintings plus the full evaluation, about 2 minutes
docker run --rm --gpus all --shm-size=8g -e TASK3_SMOKE=1 -v "<repo_root>:/app" -w /app/task3_gan/member_anushka/src pytorch-task3 jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 --output /tmp/smoke.ipynb Part3_CycleGAN.ipynb

# 3. Full run (50 epochs): same command without  -e TASK3_SMOKE=1  and with  --inplace  instead of  --output /tmp/smoke.ipynb
```

Replace `<repo_root>` with the path of your clone. A smoke test writes only to `outputs/smoke_run_<run_id>/` and its own raw log, so it never overwrites the results of the full run.

To continue an interrupted run, or to re-run only the evaluation of a finished run (for example after the human audit sheets are filled in), add `-e TASK3_RESUME=<checkpoint file name>`.

## Load the trained generators

```python
import torch
ckpt = torch.load('task3_gan/member_anushka/checkpoints/generators_<run_id>_epoch050.pt', map_location='cpu')
# keys: run_id, epoch, config, G_A2B (Monet -> Photo), G_B2A (Photo -> Monet)
```

Build `ResNetGenerator(config['generator_base_channels'], config['generator_residual_blocks'])` from the notebook and load the state dict. Inputs and outputs are in [-1, 1].

## References

- Zhu, Park, Isola, Efros, *Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks*, ICCV 2017.
