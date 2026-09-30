"""Local estimate of the Kaggle score (FID, MiFID) for a folder or zip of 256x256 JPGs, and writes
`submission.csv` (ID, FID, MiFID). The official evaluation script was not provided, so this follows
the competition's Evaluation text:
  FID   = Frechet distance between the generated-image Inception features and the reference Monet
          statistics in real_stats.npz (mu_real, sigma_real). Features use the extractor that
          reproduces those stats (see kaggle_inception), not the pytorch-fid weights that evaluate.py
          uses for the report metrics, so the two FIDs are not directly comparable.
  MiFID = mean cosine distance between generated and real Monet features, both subsampled to the same
          size. Each generated image is compared with its nearest real image (min cosine distance).
          The all-pairs mean is also printed as a cross-check.
Final leaderboard score = (FID + MiFID) / 2.

Usage:
    python task3_gan/member_nikhil/kaggle_score.py --images task3_gan/member_nikhil/images.zip [--id 1]
"""
import argparse
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from scipy import linalg

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
from runtime import REPO_ROOT, pick_device  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--images", required=True, help="images.zip or a folder of jpgs")
ap.add_argument("--stats", default=str(REPO_ROOT / "task3_gan/data/real_stats.npz"))
ap.add_argument("--out", default=str(HERE / "submission.csv"))
ap.add_argument("--id", type=int, default=1)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()


def load_images(path):
    p = Path(path)
    if p.suffix == ".zip":
        z = zipfile.ZipFile(p)
        for n in sorted(z.namelist()):
            if n.lower().endswith((".jpg", ".jpeg")):
                yield Image.open(io.BytesIO(z.read(n))).convert("RGB")
    else:
        for f in sorted(p.glob("*.jp*g")):
            yield Image.open(f).convert("RGB")


def kaggle_inception(device):
    """The extractor that reproduces real_stats.npz: torchvision ImageNet Inception-v3, fc removed,
    bilinear resize to 299, mean=std=0.5 (checked: real Monet images give per-image cosine 0.9999 vs
    feats_real). Fixed measuring instrument only, never used to generate images."""
    m = torchvision.models.inception_v3(weights=torchvision.models.Inception_V3_Weights.IMAGENET1K_V1,
                                        transform_input=False)
    m.fc = torch.nn.Identity()
    return m.to(device).eval()


def features(imgs, device, bs=50):
    net, out, buf = kaggle_inception(device), [], []

    def flush():
        x = F.interpolate(torch.stack(buf).to(device), size=(299, 299), mode="bilinear", align_corners=False)
        with torch.no_grad():
            out.append(net((x - 0.5) / 0.5).cpu().double().numpy())
        buf.clear()

    for im in imgs:
        assert im.size == (256, 256), f"expected 256x256, got {im.size}"
        buf.append(torch.from_numpy(np.asarray(im, dtype=np.float32) / 255).permute(2, 0, 1))
        if len(buf) == bs:
            flush()
    if buf:
        flush()
    return np.concatenate(out)


device = pick_device("auto")
gen = features(load_images(a.images), device)
ref = np.load(a.stats)

# FID against the reference Gaussian statistics
mu_g, sig_g = gen.mean(0), np.cov(gen, rowvar=False)
diff = mu_g - ref["mu_real"]
covmean, _ = linalg.sqrtm(sig_g.dot(ref["sigma_real"]), disp=False)
if not np.isfinite(covmean).all():
    off = np.eye(sig_g.shape[0]) * 1e-6
    covmean = linalg.sqrtm((sig_g + off).dot(ref["sigma_real"] + off))
covmean = covmean.real
fid = float(diff.dot(diff) + np.trace(sig_g) + np.trace(ref["sigma_real"]) - 2 * np.trace(covmean))

# MiFID proxy: cosine distances on equal-sized subsamples
rng = np.random.default_rng(a.seed)
real = ref["feats_real"]
n = min(len(gen), len(real))
g = gen[rng.choice(len(gen), n, replace=False)]
r = real[rng.choice(len(real), n, replace=False)]
g /= np.linalg.norm(g, axis=1, keepdims=True)
r /= np.linalg.norm(r, axis=1, keepdims=True)
dist = 1 - g @ r.T
mifid = float(dist.min(1).mean())
print(f"images={len(gen)}  FID={fid:.4f}  MiFID(nearest-real cos dist)={mifid:.4f}  "
      f"MiFID(all-pairs mean, cross-check)={float(dist.mean()):.4f}  score=(FID+MiFID)/2={(fid + mifid) / 2:.4f}")

Path(a.out).write_text(f"ID,FID,MiFID\n{a.id},{fid:.3f},{mifid:.3f}\n", encoding="utf-8")
print(f"wrote {a.out}")
