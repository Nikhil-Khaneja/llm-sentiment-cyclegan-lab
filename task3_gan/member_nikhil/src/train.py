"""Train Nikhil's CycleGAN (Monet <-> Photo) from a YAML config, then run the full evaluation.

Smoke test (from repo root):
    python task3_gan/member_nikhil/src/train.py --config task3_gan/member_nikhil/src/configs/smoke.yaml
Full run (GPU):
    python task3_gan/member_nikhil/src/train.py --config task3_gan/member_nikhil/src/configs/cyclegan_nikhil.yaml
Resume an interrupted run from its latest checkpoint (same run_id, same log file):
    python task3_gan/member_nikhil/src/train.py --config ... --resume task3_gan/member_nikhil/checkpoints/<run_id>_epoch010.pt
"""
import argparse
import csv
import itertools
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data as D  # noqa: E402
from models import ImagePool, PatchDiscriminator, ResnetGenerator, count_params, init_weights  # noqa: E402
from runtime import (REPO_ROOT, PeakMemory, RawLog, hardware_name, pick_device, rel,  # noqa: E402
                     set_seed, write_manifest)

MEMBER_DIR = Path(__file__).resolve().parents[1]
MEMBER = MEMBER_DIR.name.replace("member_", "")


def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def grad_norm(params):
    norms = [p.grad.detach().norm() for p in params if p.grad is not None]
    return torch.norm(torch.stack(norms)).item() if norms else 0.0


def lr_lambda(n_const, n_decay):
    """Constant for n_const epochs, then linear decay to 0 over n_decay epochs."""
    return lambda ep: 1.0 - max(0, ep - n_const) / float(n_decay + 1)


def build(cfg, device):
    m = cfg["model"]
    nets = {
        "G_AB": ResnetGenerator(m["ngf"], m["n_blocks"]),   # Monet -> Photo
        "G_BA": ResnetGenerator(m["ngf"], m["n_blocks"]),   # Photo -> Monet
        "D_A": PatchDiscriminator(m["ndf"]),                # real Monet vs fake Monet
        "D_B": PatchDiscriminator(m["ndf"]),                # real Photo vs fake Photo
    }
    for n in nets.values():
        n.apply(init_weights)
        n.to(device)
    return nets


def data_splits(cfg):
    d = cfg["data"]
    root = REPO_ROOT / d["root"]
    a = D.list_images(root / d["monet_dir"])
    b = D.list_images(root / d["photo_dir"])
    if d.get("max_images"):
        a, b = a[:d["max_images"]], b[:d["max_images"]]
    a_tr, a_ev = D.split_domain(a, d["monet_holdout"], cfg["seed"])
    b_tr, b_ev = D.split_domain(b, d["photo_holdout"], cfg["seed"])
    # Monet has only ~300 images; if none are held out, evaluate Monet->Photo on the training Monets.
    return {"a_train": a_tr, "b_train": b_tr, "a_eval": a_ev or a_tr, "b_eval": b_ev, "a_all": a, "b_all": b}


def save_grid(nets, fixed_a, fixed_b, path):
    import torchvision.utils as vu
    for n in nets.values():
        n.eval()
    with torch.no_grad():
        fb, fa = nets["G_AB"](fixed_a), nets["G_BA"](fixed_b)
        rows = [fixed_a, fb, nets["G_BA"](fb), fixed_b, fa, nets["G_AB"](fa)]
    for n in nets.values():
        n.train()
    grid = vu.make_grid(torch.cat(rows), nrow=len(fixed_a), normalize=True, value_range=(-1, 1))
    vu.save_image(grid, path)


def run(config_path, resume=None, skip_eval=False):
    config_path = Path(config_path)
    cfg = yaml.safe_load(config_path.read_text())
    ckpt_dir = MEMBER_DIR / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    state = torch.load(resume, map_location="cpu") if resume else None
    run_id = state["run_id"] if state else f"{cfg['run_name']}_{datetime.now().strftime('%Y%m%d-%H%M')}"
    log = RawLog(REPO_ROOT / "reproducibility" / "raw_logs" / MEMBER / f"task3_{run_id}.log")
    out_dir = MEMBER_DIR / "outputs" / run_id
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)

    set_seed(cfg["seed"] + (state["epoch"] if state else 0))
    device = pick_device(cfg.get("device", "auto"))
    log(f"{'RESUME from ' + rel(resume) if resume else 'START'} run_id={run_id} config={rel(config_path)} "
        f"device={device} hardware={hardware_name(device)}")
    log(f"config: {json.dumps(cfg)}")

    sp = data_splits(cfg)
    log(f"data: monet train/eval={len(sp['a_train'])}/{len(sp['a_eval'])} photo train/eval={len(sp['b_train'])}/{len(sp['b_eval'])}")
    d, t = cfg["data"], cfg["train"]
    ds = D.Unpaired(sp["a_train"], sp["b_train"], D.train_transform(d["image_size"], d["load_size"]), d.get("epoch_size"))
    dl = DataLoader(ds, batch_size=t["batch_size"], shuffle=True, num_workers=t["num_workers"], drop_last=True,
                    pin_memory=device.type == "cuda", persistent_workers=t["num_workers"] > 0)

    nets = build(cfg, device)
    params = {k: count_params(v) for k, v in nets.items()}
    log(f"params: {params} total={sum(params.values()):,}")
    opt_G = torch.optim.Adam(itertools.chain(nets["G_AB"].parameters(), nets["G_BA"].parameters()),
                             lr=t["lr"], betas=tuple(t["betas"]))
    opt_D = torch.optim.Adam(itertools.chain(nets["D_A"].parameters(), nets["D_B"].parameters()),
                             lr=t["lr"], betas=tuple(t["betas"]))
    sched = [torch.optim.lr_scheduler.LambdaLR(o, lr_lambda(t["epochs_constant"], t["epochs_decay"])) for o in (opt_G, opt_D)]
    start_epoch, hist, train_time, nan_steps = 1, [], 0.0, 0
    if state:
        for k, n in nets.items():
            n.load_state_dict(state["nets"][k])
        opt_G.load_state_dict(state["opt_G"]); opt_D.load_state_dict(state["opt_D"])
        for s, sd in zip(sched, state["sched"]):
            s.load_state_dict(sd)
        start_epoch, hist, train_time, nan_steps = state["epoch"] + 1, state["hist"], state["train_time"], state["nan_steps"]

    mse, l1 = nn.MSELoss(), nn.L1Loss()   # LSGAN adversarial loss; L1 for cycle / identity
    pool_A, pool_B = ImagePool(t["pool_size"]), ImagePool(t["pool_size"])
    fixed = D.Folder(sp["a_eval"][:4], D.eval_transform(d["image_size"])), D.Folder(sp["b_eval"][:4], D.eval_transform(d["image_size"]))
    fixed_a = torch.stack([fixed[0][i][0] for i in range(len(fixed[0]))]).to(device)
    fixed_b = torch.stack([fixed[1][i][0] for i in range(len(fixed[1]))]).to(device)
    mem = PeakMemory(device)
    n_epochs = t["epochs_constant"] + t["epochs_decay"]
    lam_c, lam_i = t["lambda_cycle"], t["lambda_identity"]

    for epoch in range(start_epoch, n_epochs + 1):
        sums, n_it = {}, 0
        sync(device)
        t0 = time.perf_counter()
        for real_a, real_b in dl:
            real_a, real_b = real_a.to(device, non_blocking=True), real_b.to(device, non_blocking=True)

            # ---- generators ----
            fake_b, fake_a = nets["G_AB"](real_a), nets["G_BA"](real_b)
            rec_a, rec_b = nets["G_BA"](fake_b), nets["G_AB"](fake_a)
            pred_fb, pred_fa = nets["D_B"](fake_b), nets["D_A"](fake_a)
            adv_ab = mse(pred_fb, torch.ones_like(pred_fb))
            adv_ba = mse(pred_fa, torch.ones_like(pred_fa))
            cyc_a, cyc_b = l1(rec_a, real_a), l1(rec_b, real_b)
            if lam_i > 0:  # G_BA(real Monet) should stay Monet; G_AB(real Photo) should stay Photo
                idt_a, idt_b = l1(nets["G_BA"](real_a), real_a), l1(nets["G_AB"](real_b), real_b)
            else:
                idt_a = idt_b = torch.zeros((), device=device)
            loss_G = adv_ab + adv_ba + lam_c * (cyc_a + cyc_b) + lam_i * (idt_a + idt_b)
            opt_G.zero_grad(set_to_none=True)
            loss_G.backward()
            gn_G = grad_norm(itertools.chain(nets["G_AB"].parameters(), nets["G_BA"].parameters()))

            # ---- discriminators (fakes drawn from the history pool) ----
            def d_loss(Dn, real, fake):
                pr, pf = Dn(real), Dn(fake.detach())
                return 0.5 * (mse(pr, torch.ones_like(pr)) + mse(pf, torch.zeros_like(pf)))

            loss_DA = d_loss(nets["D_A"], real_a, pool_A.query(fake_a))
            loss_DB = d_loss(nets["D_B"], real_b, pool_B.query(fake_b))
            opt_D.zero_grad(set_to_none=True)
            (loss_DA + loss_DB).backward()
            gn_D = grad_norm(itertools.chain(nets["D_A"].parameters(), nets["D_B"].parameters()))

            vals = {"G_total": loss_G.item(), "G_adv_A2B": adv_ab.item(), "G_adv_B2A": adv_ba.item(),
                    "cycle_A": cyc_a.item(), "cycle_B": cyc_b.item(), "identity_A": idt_a.item(),
                    "identity_B": idt_b.item(), "D_A": loss_DA.item(), "D_B": loss_DB.item(),
                    "grad_norm_G": gn_G, "grad_norm_D": gn_D}
            if not all(math.isfinite(v) for v in vals.values()):
                nan_steps += 1
                log(f"epoch {epoch} iter {n_it}: non-finite values {vals}; skipping update")
                opt_G.zero_grad(set_to_none=True)
                continue
            opt_G.step()
            opt_D.step()
            for k, v in vals.items():
                sums[k] = sums.get(k, 0.0) + v
            n_it += 1
            if n_it % cfg["log_every"] == 0:
                mem.sample()
                log(f"epoch {epoch} iter {n_it}/{len(dl)} " + " ".join(f"{k}={v:.4f}" for k, v in vals.items()))
        sync(device)
        ep_time = time.perf_counter() - t0
        train_time += ep_time
        for s in sched:
            s.step()
        row = {"epoch": epoch, **{k: v / max(n_it, 1) for k, v in sums.items()},
               "lr": opt_G.param_groups[0]["lr"], "epoch_time_sec": ep_time,
               "train_images_per_sec": 2 * n_it * t["batch_size"] / ep_time}
        hist.append(row)
        log("epoch_summary " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
        save_grid(nets, fixed_a, fixed_b, out_dir / "samples" / f"epoch{epoch:03d}.png")

        if epoch % t["checkpoint_every"] == 0 or epoch == n_epochs:
            path = ckpt_dir / f"{run_id}_epoch{epoch:03d}.pt"
            torch.save({"run_id": run_id, "epoch": epoch, "config": cfg,
                        "nets": {k: n.state_dict() for k, n in nets.items()},
                        "opt_G": opt_G.state_dict(), "opt_D": opt_D.state_dict(),
                        "sched": [s.state_dict() for s in sched], "hist": hist,
                        "train_time": train_time, "nan_steps": nan_steps}, path)
            log(f"checkpoint -> {rel(path)}")

    final_ckpt = ckpt_dir / f"{run_id}_epoch{n_epochs:03d}.pt"
    with (out_dir / "train_history.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(hist[0].keys()))
        w.writeheader()
        w.writerows(hist)
    save_loss_plots(hist, out_dir)
    train_stats = {"parameter_count": params, "train_time_sec": train_time, "nan_inf_steps": nan_steps,
                   "train_images_per_sec": float(np.mean([h["train_images_per_sec"] for h in hist])),
                   **mem.report_mb(), "hardware": hardware_name(device)}
    (out_dir / "train_stats.json").write_text(json.dumps(train_stats, indent=1))
    log(f"training done: {json.dumps(train_stats)}")

    if not skip_eval:
        import evaluate
        evaluate.run(final_ckpt, log=log)

    write_manifest(REPO_ROOT / "reproducibility" / "manifests" / MEMBER / f"task3_{run_id}.json", cfg, device, {
        "task": "task3_gan", "member": MEMBER, "run_id": run_id, "config_file": rel(config_path),
        "raw_log": rel(log.path), "outputs_dir": rel(out_dir),
        "checkpoints": {"final (used for all reported metrics + Kaggle)": rel(final_ckpt),
                        "intermediate": [rel(p) for p in sorted(ckpt_dir.glob(f"{run_id}_epoch*.pt"))]},
        "train_stats": train_stats,
    })
    log(f"done: outputs -> {rel(out_dir)}")
    return {"run_id": run_id, "out_dir": out_dir, "checkpoint": final_ckpt}


def save_loss_plots(hist, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = [h["epoch"] for h in hist]
    fig, ax = plt.subplots(1, 4, figsize=(20, 4))
    for k in ["G_adv_A2B", "G_adv_B2A"]:
        ax[0].plot(ep, [h[k] for h in hist], "o-", label=k)
    for k in ["D_A", "D_B"]:
        ax[0].plot(ep, [h[k] for h in hist], "s--", label=k)
    ax[0].set(title="Adversarial losses (LSGAN)", xlabel="epoch")
    for k in ["cycle_A", "cycle_B", "identity_A", "identity_B"]:
        ax[1].plot(ep, [h[k] for h in hist], "o-", label=k)
    ax[1].set(title="Cycle / identity L1 (unweighted)", xlabel="epoch")
    for k in ["grad_norm_G", "grad_norm_D"]:
        ax[2].plot(ep, [h[k] for h in hist], "o-", label=k)
    ax[2].set(title="Mean gradient L2 norm", xlabel="epoch", yscale="log")
    ax[3].plot(ep, [h["G_total"] for h in hist], "o-", label="G_total")
    ax[3].plot(ep, [h["lr"] for h in hist], "k:", label="lr (after epoch)")
    ax[3].set(title="Total generator loss", xlabel="epoch")
    for a in ax:
        a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "loss_curves.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", help="checkpoint to continue from")
    ap.add_argument("--skip-eval", action="store_true")
    a = ap.parse_args()
    run(a.config, a.resume, a.skip_eval)
