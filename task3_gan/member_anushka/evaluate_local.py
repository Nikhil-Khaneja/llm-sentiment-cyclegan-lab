"""Course evaluation script (Part3_Evaluation_Script.ipynb) as a command-line file.

The metric code is the course script unchanged; only the folder configuration was replaced by
repo-relative defaults and command-line arguments. It writes submission.csv (FID and MiFID,
averaged over both directions).

    A = Monet, B = Photo
    FID_A2B, MiFID_A2B: real photos vs generated photos (pred_A2B)
    FID_B2A, MiFID_B2A: real Monet  vs generated Monet  (pred_B2A)

Run from anywhere:
    python task3_gan/member_anushka/evaluate_local.py
"""
import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.linalg
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from scipy.spatial.distance import cosine
from tqdm import tqdm

MEMBER_DIR = Path(__file__).resolve().parent
DATA_DIR = MEMBER_DIR.parent / "data"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def list_images(folder):
    exts = (".jpg", ".jpeg", ".png")
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(folder, f"*{ext}")))
        paths.extend(glob.glob(os.path.join(folder, f"*{ext.upper()}")))
    paths = sorted(list(set(paths)))
    return paths


def take_n(paths, n):
    if n is None:
        return paths
    return paths[:min(n, len(paths))]


# Inception feature extractor
def get_inception_model():
    inception = models.inception_v3(
        weights=models.Inception_V3_Weights.IMAGENET1K_V1,
        transform_input=False
    )
    inception.fc = nn.Identity()
    inception.to(device)
    inception.eval()
    return inception


INCEPTION_TF = T.Compose([
    T.Resize(299),
    T.CenterCrop(299),
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


def load_batch(paths):
    imgs = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        imgs.append(INCEPTION_TF(img))
    return torch.stack(imgs, dim=0)


@torch.no_grad()
def get_activations(model, image_paths, batch_size=32):
    feats = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Inception activations"):
        batch_paths = image_paths[i:i+batch_size]
        x = load_batch(batch_paths).to(device)
        f = model(x).detach().cpu().numpy()
        feats.append(f)
    return np.concatenate(feats, axis=0)


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    diff = mu1 - mu2
    return float(diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean))


def calculate_fid_mifid(real_paths, gen_paths, batch_size=32, subsample_to_match=True):
    # For fair comparison, match counts
    real_paths = sorted(real_paths)
    gen_paths = sorted(gen_paths)

    if subsample_to_match:
        n = min(len(real_paths), len(gen_paths))
        real_paths = real_paths[:n]
        gen_paths = gen_paths[:n]

    model = get_inception_model()

    real_act = get_activations(model, real_paths, batch_size=batch_size)
    gen_act = get_activations(model, gen_paths, batch_size=batch_size)

    mu_r, sig_r = real_act.mean(axis=0), np.cov(real_act, rowvar=False)
    mu_g, sig_g = gen_act.mean(axis=0), np.cov(gen_act, rowvar=False)

    fid = frechet_distance(mu_r, sig_r, mu_g, sig_g)

    # mean cosine distance between feature vectors,
    # paired by index after subsampling/matching
    m = min(len(real_act), len(gen_act))
    cos_dists = [cosine(real_act[i], gen_act[i]) for i in range(m)]
    mifid = float(np.mean(cos_dists))

    return fid, mifid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-monet", default=str(DATA_DIR / "monet_jpg"))             # real domain A
    ap.add_argument("--real-photo", default=str(DATA_DIR / "photo_jpg"))             # real domain B
    ap.add_argument("--pred-a2b", default=str(MEMBER_DIR / "outputs" / "pred_A2B"))  # Monet -> Photo
    ap.add_argument("--pred-b2a", default=str(MEMBER_DIR / "outputs" / "pred_B2A"))  # Photo -> Monet
    ap.add_argument("--n-eval", type=int, default=300, help="images per set (course default 300)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", default=str(MEMBER_DIR / "submission.csv"))
    args = ap.parse_args()
    print("Device:", device)

    for d in [args.real_monet, args.real_photo, args.pred_a2b, args.pred_b2a]:
        assert os.path.isdir(d), f"Missing folder: {d}"

    real_monet = take_n(list_images(args.real_monet), args.n_eval)
    real_photo = take_n(list_images(args.real_photo), args.n_eval)
    gen_a2b = take_n(list_images(args.pred_a2b), args.n_eval)  # Monet->Photo (generated photos)
    gen_b2a = take_n(list_images(args.pred_b2a), args.n_eval)  # Photo->Monet (generated monet)

    print("\nCounts (after N_EVAL cap):")
    print("Real Monet:", len(real_monet), " | Gen Monet (B2A):", len(gen_b2a))
    print("Real Photo:", len(real_photo), " | Gen Photo (A2B):", len(gen_a2b))

    print("\n Evaluating Photo -> Monet (B2A) ")
    # Ground truth = real Monet; Generated = pred_B2A (generated Monet-like)
    fid_B2A, mifid_B2A = calculate_fid_mifid(real_monet, gen_b2a, batch_size=args.batch_size)
    print(f"[Photo->Monet] FID={fid_B2A:.3f}  MiFID={mifid_B2A:.4f}")

    print("\n Evaluating Monet -> Photo (A2B) ")
    # Ground truth = real Photo; Generated = pred_A2B (generated Photo-like)
    fid_A2B, mifid_A2B = calculate_fid_mifid(real_photo, gen_a2b, batch_size=args.batch_size)
    print(f"[Monet->Photo] FID={fid_A2B:.3f}  MiFID={mifid_A2B:.4f}")

    sub_fid = (fid_A2B + fid_B2A) / 2
    sub_mifid = (mifid_A2B + mifid_B2A) / 2

    submission = pd.DataFrame([{
        "ID": 1,
        "FID": float(sub_fid),
        "MiFID": float(sub_mifid)}])

    submission.to_csv(args.out, index=False)
    print(submission)
    print(f"DIRECTIONAL fid_A2B={fid_A2B} mifid_A2B={mifid_A2B} fid_B2A={fid_B2A} mifid_B2A={mifid_B2A}")


if __name__ == "__main__":
    main()
