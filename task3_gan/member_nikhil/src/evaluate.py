"""Evaluate a CycleGAN checkpoint: every Task 3 metric, both directions.

Pretrained networks appear ONLY here, as fixed measuring instruments (InceptionV3 for FID/KID/
precision-recall/content cosine, AlexNet-LPIPS). They never touch training or the Kaggle images.

A = Monet, B = Photo.  A2B: Monet -> Photo (compared with real photos).
                        B2A: Photo -> Monet (compared with real Monets; this is the Kaggle direction).
"""
import csv
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import linalg
from torch.utils.data import DataLoader

import data as D
from models import ResnetGenerator
from runtime import REPO_ROOT, PeakMemory, pick_device, rel

MEMBER_DIR = Path(__file__).resolve().parents[1]


# ---------------- feature-space metrics ----------------

class InceptionFeatures:
    """2048-d pool3 features from the standard FID InceptionV3 (pytorch-fid weights)."""

    def __init__(self, device):
        from pytorch_fid.inception import InceptionV3
        self.device = device
        self.net = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]], resize_input=True, normalize_input=True).to(device).eval()

    @torch.no_grad()
    def __call__(self, x):  # x in [-1, 1]
        return self.net((x + 1) / 2)[0].squeeze(-1).squeeze(-1).cpu().double().numpy()


def fid(f1, f2):
    mu1, mu2 = f1.mean(0), f2.mean(0)
    s1, s2 = np.cov(f1, rowvar=False), np.cov(f2, rowvar=False)
    covmean, _ = linalg.sqrtm(s1.dot(s2), disp=False)
    if not np.isfinite(covmean).all():
        off = np.eye(s1.shape[0]) * 1e-6
        covmean = linalg.sqrtm((s1 + off).dot(s2 + off))
    covmean = covmean.real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


# NumPy 2.0 + macOS Accelerate emits spurious divide/overflow warnings from matmul even when every
# value is finite (checked against einsum), so those warnings are silenced for the two matmul helpers.
@np.errstate(divide="ignore", over="ignore", invalid="ignore")
def kid(f1, f2, n_subsets=100, subset_size=1000, seed=0):
    """Unbiased MMD^2 with the cubic polynomial kernel k(x,y) = (x.y/d + 1)^3, averaged over subsets."""
    rng = np.random.default_rng(seed)
    d = f1.shape[1]
    m = min(subset_size, len(f1), len(f2))
    vals = []
    for _ in range(n_subsets):
        x = f1[rng.choice(len(f1), m, replace=False)]
        y = f2[rng.choice(len(f2), m, replace=False)]
        kxx, kyy, kxy = (x @ x.T / d + 1) ** 3, (y @ y.T / d + 1) ** 3, (x @ y.T / d + 1) ** 3
        vals.append((kxx.sum() - np.trace(kxx)) / (m * (m - 1)) + (kyy.sum() - np.trace(kyy)) / (m * (m - 1)) - 2 * kxy.mean())
    return float(np.mean(vals)), float(np.std(vals))


@np.errstate(divide="ignore", over="ignore", invalid="ignore")
def _pdist(a, b):
    return np.sqrt(np.maximum((a ** 2).sum(1)[:, None] + (b ** 2).sum(1)[None] - 2 * a @ b.T, 0))


def prdc(real, fake, k=5):
    """Precision/recall (Kynkaanniemi et al. 2019) and density/coverage (Naeem et al. 2020)
    using k-NN balls in Inception feature space."""
    r_radii = np.sort(_pdist(real, real), axis=1)[:, k]
    f_radii = np.sort(_pdist(fake, fake), axis=1)[:, k]
    d_rf = _pdist(real, fake)
    precision = (d_rf < r_radii[:, None]).any(0).mean()
    recall = (d_rf < f_radii[None, :]).any(1).mean()
    density = (1.0 / k) * (d_rf < r_radii[:, None]).sum(0).mean()
    coverage = (d_rf.min(1) < r_radii).mean()
    return {"precision": float(precision), "recall": float(recall), "density": float(density), "coverage": float(coverage)}


# ---------------- helpers ----------------

def load_generators(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location="cpu")
    m = ck["config"]["model"]
    gens = {}
    for k in ["G_AB", "G_BA"]:
        g = ResnetGenerator(m["ngf"], m["n_blocks"])
        g.load_state_dict(ck["nets"][k])
        gens[k] = g.to(device).eval()
    return ck, gens


def translate_folder(files, G, G_back, size, device, bs, feats, lpips_fn, save_dir=None):
    """Translate `files`, return input/translation/reconstruction features + per-image scores."""
    dl = DataLoader(D.Folder(files, D.eval_transform(size)), batch_size=bs, shuffle=False, num_workers=0)
    f_in, f_out, cyc, lp_tr, lp_rec, n, t_gen = [], [], [], [], [], 0, 0.0
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for x, idx in dl:
            x = x.to(device)
            t0 = time.perf_counter()
            y = G(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elif device.type == "mps":
                torch.mps.synchronize()
            t_gen += time.perf_counter() - t0
            rec = G_back(y)
            f_in.append(feats(x)); f_out.append(feats(y))
            cyc.append(((rec - x).abs().mean((1, 2, 3)) / 2).cpu())   # /2: L1 on the [0,1] pixel scale
            lp_tr.append(lpips_fn(x, y).flatten().cpu())
            lp_rec.append(lpips_fn(x, rec).flatten().cpu())
            if save_dir:
                for img, i in zip(D.to_uint8(y), idx.tolist()):
                    Image.fromarray(img).save(save_dir / f"{Path(files[i]).stem}.jpg", quality=95)
            n += len(x)
    f_in, f_out = np.concatenate(f_in), np.concatenate(f_out)
    cos = (f_in * f_out).sum(1) / (np.linalg.norm(f_in, axis=1) * np.linalg.norm(f_out, axis=1) + 1e-8)
    return {"f_in": f_in, "f_out": f_out, "cycle_l1": torch.cat(cyc).numpy(), "lpips_translation": torch.cat(lp_tr).numpy(),
            "lpips_reconstruction": torch.cat(lp_rec).numpy(), "content_cosine": cos, "gen_images_per_sec": n / t_gen}


def real_features(files, size, device, bs, feats):
    dl = DataLoader(D.Folder(files, D.eval_transform(size)), batch_size=bs, shuffle=False)
    with torch.no_grad():
        return np.concatenate([feats(x.to(device)) for x, _ in dl])


def human_audit(files_b, files_a, gens, size, device, out_dir, n=30, seed=0):
    """30 fixed samples (15 per direction), shuffled and blinded to a sample id, plus a blank
    rating sheet per rater. Score 1-5: style (looks like target domain), content (input
    structure kept), artifacts (5 = none)."""
    rng = np.random.default_rng(seed)
    picks = [("B2A", f) for f in rng.choice(files_b, n // 2, replace=False)] + \
            [("A2B", f) for f in rng.choice(files_a, n - n // 2, replace=False)]
    order = rng.permutation(len(picks))
    audit = out_dir / "human_audit"
    (audit / "images").mkdir(parents=True, exist_ok=True)
    tf = D.eval_transform(size)
    key = []
    for sid, j in enumerate(order, 1):
        direction, f = picks[j]
        x = tf(Image.open(f).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            y = gens["G_BA" if direction == "B2A" else "G_AB"](x)
        pair = np.concatenate([D.to_uint8(x)[0], np.full((size, 8, 3), 255, np.uint8), D.to_uint8(y)[0]], axis=1)
        Image.fromarray(pair).save(audit / "images" / f"sample_{sid:02d}.png")
        key.append({"sample_id": sid, "direction": direction, "source_file": Path(f).name})
    with (audit / "answer_key_do_not_show_raters.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(key[0]))
        w.writeheader(); w.writerows(key)
    for r in ["rater1", "rater2"]:
        with (audit / f"ratings_{r}.csv").open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["sample_id", "style_1to5", "content_1to5", "artifacts_1to5", "comment"])
            w.writerows([[k["sample_id"], "", "", "", ""] for k in key])


def run(ckpt_path, log=print, write_submission=None):
    ckpt_path = Path(ckpt_path)
    device = pick_device("auto")
    ck, gens = load_generators(ckpt_path, device)
    cfg = ck["config"]
    e, size = cfg["eval"], cfg["data"]["image_size"]
    out_dir = MEMBER_DIR / "outputs" / ck["run_id"]
    import train  # local import: reuse the exact same split logic
    sp = train.data_splits(cfg)
    mem = PeakMemory(device)
    import lpips
    lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    feats = InceptionFeatures(device)

    log(f"eval: {rel(ckpt_path)} on monet={len(sp['a_eval'])} photo={len(sp['b_eval'])}")
    real_a = real_features(sp["a_all"], size, device, e["batch_size"], feats)   # reference Monet set
    real_b = real_features(sp["b_eval"], size, device, e["batch_size"], feats)  # reference Photo set
    pred_root = MEMBER_DIR / "outputs"
    res = {
        "A2B": translate_folder(sp["a_eval"], gens["G_AB"], gens["G_BA"], size, device, e["batch_size"], feats, lpips_fn,
                                pred_root / "pred_A2B" if e["save_predictions"] else None),
        "B2A": translate_folder(sp["b_eval"], gens["G_BA"], gens["G_AB"], size, device, e["batch_size"], feats, lpips_fn,
                                pred_root / "pred_B2A" if e["save_predictions"] else None),
    }
    mem.sample()
    ref = {"A2B": real_b, "B2A": real_a}
    rows = []
    for direction, r in res.items():
        k_mean, k_std = kid(r["f_out"], ref[direction], subset_size=e["kid_subset_size"])
        pr = prdc(ref[direction], r["f_out"], k=e["prdc_k"])
        vals = {
            "FID": fid(r["f_out"], ref[direction]),
            "KID_mean": k_mean, "KID_std": k_std,
            **{f"gen_{k}": v for k, v in pr.items()},
            "cycle_reconstruction_L1": float(r["cycle_l1"].mean()),
            "LPIPS_input_vs_translation": float(r["lpips_translation"].mean()),
            "LPIPS_input_vs_reconstruction": float(r["lpips_reconstruction"].mean()),
            "content_cosine_input_vs_translation": float(r["content_cosine"].mean()),
            "inference_images_per_sec": r["gen_images_per_sec"],
            "n_translated": int(len(r["f_out"])), "n_reference": int(len(ref[direction])),
        }
        rows += [{"direction": direction, "metric": k, "value": v} for k, v in vals.items()]
        log(f"eval {direction}: " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in vals.items()))
    # FID of the untranslated inputs vs the target domain: the "do nothing" reference point.
    rows.append({"direction": "B2A", "metric": "FID_identity_baseline_photos_vs_monet", "value": fid(res["B2A"]["f_in"], real_a)})
    rows.append({"direction": "A2B", "metric": "FID_identity_baseline_monet_vs_photos", "value": fid(res["A2B"]["f_in"], real_b)})

    # training-side metrics (loss values, stability, cost)
    hist = ck["hist"]
    last = hist[-1]
    ts_path = out_dir / "train_stats.json"
    ts = json.loads(ts_path.read_text()) if ts_path.exists() else {}
    for k in ["cycle_A", "cycle_B", "identity_A", "identity_B", "G_adv_A2B", "G_adv_B2A", "D_A", "D_B", "G_total"]:
        rows.append({"direction": "train", "metric": f"final_epoch_{k}", "value": last[k]})
    gnG, gnD = [h["grad_norm_G"] for h in hist], [h["grad_norm_D"] for h in hist]
    rows += [{"direction": "train", "metric": k, "value": v} for k, v in {
        "grad_norm_G_mean": float(np.mean(gnG)), "grad_norm_G_max_epoch_mean": float(np.max(gnG)),
        "grad_norm_D_mean": float(np.mean(gnD)), "grad_norm_D_max_epoch_mean": float(np.max(gnD)),
        "nan_inf_steps": ck["nan_steps"], "epochs": ck["epoch"],
        "train_time_sec": ck["train_time"], "train_images_per_sec": ts.get("train_images_per_sec"),
        "params_G_AB": ts.get("parameter_count", {}).get("G_AB"), "params_G_BA": ts.get("parameter_count", {}).get("G_BA"),
        "params_D_A": ts.get("parameter_count", {}).get("D_A"), "params_D_B": ts.get("parameter_count", {}).get("D_B"),
        "peak_accelerator_mem_mb_train": ts.get("peak_accelerator_mem_mb"),
        "peak_accelerator_mem_mb_eval": mem.report_mb()["peak_accelerator_mem_mb"],
        "hardware": ts.get("hardware"), "checkpoint": rel(ckpt_path),
        "human_audit_score": "fill in: python src/audit_agreement.py",
        "kaggle_public_score": "fill in after submission", "kaggle_rank": "fill in after submission",
    }.items()]

    for path in [out_dir / "full_metrics_report.csv"] + ([MEMBER_DIR / "full_metrics_report.csv"] if cfg.get("write_metrics_report") else []):
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["direction", "metric", "value"])
            w.writeheader(); w.writerows(rows)
    human_audit(sp["b_eval"], sp["a_eval"], gens, size, device, out_dir, n=e["audit_samples"], seed=cfg["seed"])
    log(f"eval done -> {rel(out_dir / 'full_metrics_report.csv')}; human audit sheet -> {rel(out_dir / 'human_audit')}")

    if write_submission if write_submission is not None else e.get("make_submission"):
        make_submission(gens["G_BA"], sp["b_all"], cfg, device, log)
    return rows


def make_submission(G_BA, photo_files, cfg, device, log):
    """Photo -> Monet for every photo, saved as 256x256 JPEGs in images.zip (Kaggle
    'I'm Something of a Painter Myself' format). Direct generator output, no editing."""
    s = cfg["submission"]
    out_zip = MEMBER_DIR / s["zip_name"]
    dl = DataLoader(D.Folder(photo_files, D.eval_transform(s["infer_size"])), batch_size=cfg["eval"]["batch_size"])
    n = 0
    with zipfile.ZipFile(out_zip, "w") as zf, torch.no_grad():
        for x, idx in dl:
            y = G_BA(x.to(device))
            if s["infer_size"] != s["output_size"]:
                y = torch.nn.functional.interpolate(y, size=s["output_size"], mode="bicubic", align_corners=False)
            for img, i in zip(D.to_uint8(y), idx.tolist()):
                from io import BytesIO
                buf = BytesIO()
                Image.fromarray(img).save(buf, format="JPEG", quality=95)
                zf.writestr(f"{Path(photo_files[i]).stem}.jpg", buf.getvalue())
                n += 1
    log(f"submission: {n} images -> {rel(out_zip)}")
