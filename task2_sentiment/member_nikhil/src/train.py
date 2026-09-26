"""Train + evaluate all three of Nikhil's Yelp Polarity models from one YAML config.

Smoke test (from repo root, ~2 min on a laptop):
    python task2_sentiment/member_nikhil/src/train.py --config task2_sentiment/member_nikhil/src/configs/smoke.yaml
Full run:
    python task2_sentiment/member_nikhil/src/train.py --config task2_sentiment/member_nikhil/src/configs/sentiment_nikhil.yaml
"""
import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics as M  # noqa: E402
import models  # noqa: E402
import prep  # noqa: E402
from runtime import (REPO_ROOT, PeakMemory, RawLog, hardware_name, pick_device, rel,  # noqa: E402
                     set_seed, write_manifest)

MEMBER_DIR = Path(__file__).resolve().parents[1]
MEMBER = MEMBER_DIR.name.replace("member_", "")


def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def batches(X, L, y, bs, device, shuffle, bucket=50):
    """Mini-batches trimmed to their longest review. When shuffling, reviews are grouped into
    length buckets (sort within random pools of bs*bucket, then shuffle batch order) so each
    batch carries little padding - a large speed-up for the GRU and CNN."""
    if shuffle:
        order = torch.randperm(len(y))
        pools = [order[i:i + bs * bucket] for i in range(0, len(y), bs * bucket)]
        order = torch.cat([p[torch.argsort(L[p], descending=True)] for p in pools])
        starts = torch.randperm((len(y) + bs - 1) // bs) * bs
    else:
        order, starts = torch.arange(len(y)), torch.arange(0, len(y), bs)
    for i in starts.tolist():
        idx = order[i:i + bs]
        lens = L[idx]
        T = int(lens.max())                       # trim padding to the longest review in the batch
        yield X[idx, :T].to(device), lens, y[idx].to(device)


@torch.no_grad()
def predict(model, X, L, y, bs, device):
    model.eval()
    probs, loss_sum = [], 0.0
    for xb, lb, yb in batches(X, L, y, bs, device, shuffle=False):
        logit = model(xb, lb)
        loss_sum += F.binary_cross_entropy_with_logits(logit, yb, reduction="sum").item()
        probs.append(torch.sigmoid(logit).float().cpu())
    model.train()
    return torch.cat(probs).numpy(), loss_sum / len(y)


def make_optimizer(model, spec):
    o = spec["optim"]
    if o["name"] == "adam":
        return torch.optim.Adam(model.parameters(), lr=o["lr"])
    return torch.optim.AdamW(model.parameters(), lr=o["lr"], weight_decay=o.get("weight_decay", 0.01))


def train_one(spec, splits, vocab_size, cfg, device, log, run_id, out_dir):
    name = spec["name"]
    set_seed(cfg["seed"])
    if device.type == "mps":
        torch.mps.empty_cache()
    model = models.build(spec, vocab_size).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = make_optimizer(model, spec)
    o = spec["optim"]
    log(f"[{name}] arch={spec['arch']} params={n_params:,} spec={json.dumps(spec)}")

    Xtr, Ltr, ytr = (torch.from_numpy(a) for a in splits["train"])
    Xva, Lva, yva = (torch.from_numpy(a) for a in splits["val"])
    Xtr, Xva = Xtr.long(), Xva.long()
    mem = PeakMemory(device)
    hist, best_val, train_time, seen = [], float("inf"), 0.0, 0
    ckpt = MEMBER_DIR / "checkpoints" / f"{run_id}_{name}.pt"
    ckpt.parent.mkdir(exist_ok=True)

    for epoch in range(1, o["epochs"] + 1):
        sync(device)
        t0 = time.perf_counter()
        tot, n, step = 0.0, 0, 0
        for xb, lb, yb in batches(Xtr, Ltr, ytr, o["batch_size"], device, shuffle=True):
            loss = F.binary_cross_entropy_with_logits(model(xb, lb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), o["grad_clip"])
            opt.step()
            tot += loss.item() * len(yb)
            n += len(yb)
            step += 1
            if step % cfg["log_every"] == 0:
                mem.sample()
                log(f"[{name}] ep {epoch} step {step} loss {loss.item():.4f}")
        sync(device)
        train_time += time.perf_counter() - t0
        seen += n
        p_val, val_loss = predict(model, Xva, Lva, yva, cfg["eval_batch_size"], device)
        val_acc = float(((p_val >= 0.5) == yva.numpy()).mean())
        hist.append({"epoch": epoch, "train_loss": tot / n, "val_loss": val_loss, "val_acc": val_acc})
        log(f"[{name}] epoch {epoch} train_loss={tot / n:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(), "spec": spec, "epoch": epoch, "val_loss": val_loss,
                        "vocab_size": vocab_size, "run_id": run_id}, ckpt)

    best = torch.load(ckpt, map_location=device)
    model.load_state_dict(best["model"])
    log(f"[{name}] best epoch {best['epoch']} (val_loss {best['val_loss']:.4f}) -> {rel(ckpt)}")
    Xte, Lte, yte = (torch.from_numpy(a) for a in splits["test"])
    p_val, _ = predict(model, Xva, Lva, yva, cfg["eval_batch_size"], device)
    p_test, _ = predict(model, Xte.long(), Lte, yte, cfg["eval_batch_size"], device)
    m = mem.report_mb()
    info = {"model": name, "arch": spec["arch"], "parameter_count": n_params, "best_epoch": best["epoch"],
            "train_time_sec": train_time, "train_examples_per_sec": seen / train_time,
            "peak_accelerator_mem_mb": m["peak_accelerator_mem_mb"], "peak_process_rss_mb": m["peak_process_rss_mb"],
            "hardware": hardware_name(device), "checkpoint": rel(ckpt),
            "val_macro_f1": f1_score(yva.numpy().astype(int), (p_val >= 0.5).astype(int), average="macro")}
    pd.DataFrame(hist).to_csv(out_dir / f"{name}_history.csv", index=False)
    return p_test, info, hist


def plots(results, y_test, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, roc_curve

    k = len(results)
    fig, ax = plt.subplots(1, k, figsize=(4.2 * k, 3.6))
    for a, (name, r) in zip(np.atleast_1d(ax), results.items()):
        ConfusionMatrixDisplay.from_predictions(y_test.astype(int), (r["p"] >= 0.5).astype(int),
                                                display_labels=["neg", "pos"], ax=a, colorbar=False, values_format="d")
        a.set_title(name)
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrices.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for name, r in results.items():
        h = pd.DataFrame(r["hist"])
        ax[0].plot(h["epoch"], h["train_loss"], "o--", label=f"{name} train")
        ax[0].plot(h["epoch"], h["val_loss"], "o-", label=f"{name} val")
        fpr, tpr, _ = roc_curve(y_test, r["p"])
        ax[1].plot(fpr, tpr, label=name)
        # reliability diagram
        bins = np.linspace(0, 1, 11)
        idx = np.clip(np.digitize(r["p"], bins[1:-1]), 0, 9)
        xs = [r["p"][idx == b].mean() for b in range(10) if (idx == b).any()]
        ys = [y_test[idx == b].mean() for b in range(10) if (idx == b).any()]
        ax[2].plot(xs, ys, "o-", label=name)
    ax[0].set(xlabel="epoch", ylabel="BCE loss", title="Training curves")
    ax[1].plot([0, 1], [0, 1], "k:", lw=0.8)
    ax[1].set(xlabel="FPR", ylabel="TPR", title="ROC (test)")
    ax[2].plot([0, 1], [0, 1], "k:", lw=0.8)
    ax[2].set(xlabel="predicted P(positive)", ylabel="observed positive rate", title="Reliability (test)")
    for a in ax:
        a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "curves_roc_reliability.png", dpi=120)
    plt.close(fig)


def error_candidates(y, p, meta, slices, n=5):
    """Pick the 20 errors for manual review: 5 confident FP, 5 confident FN, 5 near-threshold,
    5 from the worst-performing slice. error_type / proposed_fix are left for the human reviewer."""
    df = meta.copy()
    df["label"], df["p_pos"] = y.astype(int), p
    df["pred"] = (p >= 0.5).astype(int)
    err = df[df["pred"] != df["label"]]
    fp = err[err["label"] == 0].nlargest(n, "p_pos").assign(category="confident_false_positive")
    fn = err[err["label"] == 1].nsmallest(n, "p_pos").assign(category="confident_false_negative")
    used = set(fp.index) | set(fn.index)
    rest = err.drop(index=list(used))
    near = rest.loc[(rest["p_pos"] - 0.5).abs().sort_values().index[:n]].assign(category="near_threshold")
    used |= set(near.index)
    sl = pd.DataFrame(slices)
    worst = sl[sl["n"] >= 100].sort_values("error_rate", ascending=False).iloc[0]
    col, val = worst["slice"], worst["value"]
    in_slice = err.drop(index=list(used))
    in_slice = in_slice[in_slice[col].astype(str) == val]
    sl_pick = in_slice.loc[(in_slice["p_pos"] - 0.5).abs().sort_values(ascending=False).index[:n]]
    sl_pick = sl_pick.assign(category=f"slice:{col}={val}")
    out = pd.concat([fp, fn, near, sl_pick])
    out["text"] = out["text"].str.replace("\\n", " ", regex=False).str.slice(0, 1200)
    out["error_type"], out["proposed_fix"] = "", ""
    cols = ["category", "label", "pred", "p_pos", "length_bucket", "has_negation", "has_contrast",
            "text", "clean", "error_type", "proposed_fix"]
    return out.reset_index().rename(columns={"index": "test_row"})[["test_row"] + cols]


def run(config_path):
    config_path = Path(config_path)
    cfg = yaml.safe_load(config_path.read_text())
    run_id = f"{cfg['run_name']}_{datetime.now().strftime('%Y%m%d-%H%M')}"
    log = RawLog(REPO_ROOT / "reproducibility" / "raw_logs" / MEMBER / f"task2_{run_id}.log")
    out_dir = MEMBER_DIR / "outputs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg["seed"])
    device = pick_device(cfg.get("device", "auto"))
    log(f"run_id={run_id} config={rel(config_path)} device={device} hardware={hardware_name(device)}")
    log(f"config: {json.dumps(cfg)}")

    t0 = time.perf_counter()
    splits, test_meta, vocab = prep.prepare(cfg, REPO_ROOT, MEMBER_DIR, log)
    log(f"preprocessing took {time.perf_counter() - t0:.1f}s")
    y_test = splits["test"][2]
    tags = test_meta[["length_bucket", "has_negation", "has_contrast"]]

    results, rows, slice_rows = {}, [], []
    for spec in cfg["models"]:
        p, info, hist = train_one(spec, splits, len(vocab), cfg, device, log, run_id, out_dir)
        rep = M.full_report(y_test, p, n_boot=cfg["bootstrap"], seed=cfg["seed"])
        results[spec["name"]] = {"p": p, "hist": hist, "info": info}
        rows.append({**info, **{f"test_{k}": v for k, v in rep.items()}})
        for s in M.slice_report(y_test, p, tags):
            slice_rows.append({"model": spec["name"], **s})
        pd.DataFrame({"label": y_test.astype(int), "p_pos": p}).to_csv(out_dir / f"{spec['name']}_test_predictions.csv", index_label="test_row")
        log(f"[{spec['name']}] test: " + " ".join(f"{k}={v:.4f}" for k, v in rep.items()))

    base = cfg["models"][0]["name"]
    mc_rows = []
    for spec in cfg["models"][1:]:
        r = M.mcnemar(y_test, results[base]["p"] >= 0.5, results[spec["name"]]["p"] >= 0.5)
        mc_rows.append({"baseline": base, "experiment": spec["name"], **r})
        log(f"mcnemar {base} vs {spec['name']}: {r}")

    report = pd.DataFrame(rows)
    report.to_csv(out_dir / "metrics.csv", index=False)
    pd.DataFrame(slice_rows).to_csv(out_dir / "slice_metrics.csv", index=False)
    pd.DataFrame(mc_rows).to_csv(out_dir / "mcnemar.csv", index=False)
    plots(results, y_test, out_dir)

    # Strongest model = best validation macro-F1 (chosen without looking at test).
    strongest = max(results, key=lambda k: results[k]["info"]["val_macro_f1"])
    sl = [s for s in slice_rows if s["model"] == strongest]
    errs = error_candidates(y_test, results[strongest]["p"], test_meta, sl)
    errs.to_csv(out_dir / f"error_review_candidates_{strongest}.csv", index=False)
    log(f"strongest model by val macro-F1: {strongest}; wrote {len(errs)} error candidates")

    if cfg.get("write_metrics_report"):
        # One file: per-model metrics, then McNemar and slice tables stacked below with a 'section' column.
        full = pd.concat([report.assign(section="per_model"),
                          pd.DataFrame(mc_rows).assign(section="mcnemar_vs_baseline"),
                          pd.DataFrame(slice_rows).assign(section="slice_robustness")], ignore_index=True)
        cols = ["section"] + [c for c in full.columns if c != "section"]
        full[cols].to_csv(MEMBER_DIR / "metrics_report.csv", index=False)

    write_manifest(REPO_ROOT / "reproducibility" / "manifests" / MEMBER / f"task2_{run_id}.json", cfg, device, {
        "task": "task2_sentiment", "member": MEMBER, "run_id": run_id,
        "config_file": rel(config_path), "raw_log": rel(log.path), "outputs_dir": rel(out_dir),
        "checkpoints": {name: r["info"]["checkpoint"] for name, r in results.items()},
        "strongest_model_for_error_review": strongest,
        "headline": {name: {"test_accuracy": float(report.loc[report.model == name, "test_accuracy"].iloc[0]),
                            "test_f1_macro": float(report.loc[report.model == name, "test_f1_macro"].iloc[0])}
                     for name in results},
    })
    log(f"done in {time.perf_counter() - t0:.0f}s: outputs -> {rel(out_dir)}")
    return {"run_id": run_id, "out_dir": out_dir, "strongest": strongest}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    run(ap.parse_args().config)
