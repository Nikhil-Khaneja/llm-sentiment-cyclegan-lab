"""Train + evaluate the Task 1 character-level GPT from a YAML config.

Smoke test (from repo root, ~1 min):
    python task1_llm/member_nikhil/src/train.py --config task1_llm/member_nikhil/src/configs/smoke.yaml
Full run:
    python task1_llm/member_nikhil/src/train.py --config task1_llm/member_nikhil/src/configs/gpt_nikhil_B.yaml
"""
import argparse
import csv
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data as D  # noqa: E402
from gen_metrics import distinct_n, repeated_4gram_rate  # noqa: E402
from model import GPT, count_params  # noqa: E402
from runtime import (REPO_ROOT, PeakMemory, RawLog, hardware_name, pick_device, rel,  # noqa: E402
                     set_seed, write_manifest)

MEMBER_DIR = Path(__file__).resolve().parents[1]
MEMBER = MEMBER_DIR.name.replace("member_", "")


def lr_at(step, total_steps, lr, min_lr, warmup):
    """Linear warm-up to `lr`, then cosine decay to `min_lr` at the final step."""
    if step < warmup:
        return lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total_steps - warmup)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * min(1.0, progress)))


def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def evaluate(model, seqs, batch_size, device):
    """Mean token-level cross-entropy and top-1 next-char accuracy over `seqs`."""
    model.eval()
    tot_loss, tot_correct, tot_tok = 0.0, 0, 0
    for i in range(0, len(seqs), batch_size):
        b = seqs[i:i + batch_size].to(device).long()
        x, y = b[:, :-1], b[:, 1:]
        logits, loss = model(x, y)
        n = y.numel()
        tot_loss += loss.item() * n
        tot_correct += (logits.argmax(-1) == y).sum().item()
        tot_tok += n
    model.train()
    return tot_loss / tot_tok, tot_correct / tot_tok


def build_optimizer(model, lr, wd):
    # Weight decay on matrices/embeddings only; biases and LayerNorm gains are not decayed.
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    return torch.optim.AdamW([{"params": decay, "weight_decay": wd},
                              {"params": no_decay, "weight_decay": 0.0}], lr=lr, betas=(0.9, 0.95))


def generate_samples(model, cfg, char_to_idx, idx_to_char, device):
    g = cfg["generate"]
    model.eval()
    out, gen_tokens, gen_time = [], 0, 0.0
    for prompt in g["prompts"]:
        ctx = torch.tensor([D.encode(prompt, char_to_idx)], device=device)
        modes = [("greedy", None)] + [("temperature", g["temperature"])] * g["samples_per_prompt"]
        for mode, temp in modes:
            sync(device)
            t0 = time.perf_counter()
            ids = model.generate(ctx, g["max_new_tokens"], temperature=temp or 1.0, greedy=(mode == "greedy"))
            sync(device)
            gen_time += time.perf_counter() - t0
            gen_tokens += g["max_new_tokens"]
            text = D.decode(ids[0].tolist(), idx_to_char)
            out.append({"prompt": prompt, "mode": mode, "temperature": temp, "text": text,
                        "continuation": text[len(prompt):]})
    model.train()
    return out, gen_tokens / gen_time


def save_plots(hist, epochs_rows, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = np.array([h["step"] for h in hist])
    loss = np.array([h["loss"] for h in hist])
    k = min(100, len(loss))
    smooth = np.convolve(loss, np.ones(k) / k, mode="valid")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(steps, loss, alpha=0.25, lw=0.6, label="train (per step)")
    ax[0].plot(steps[k - 1:], smooth, lw=1.5, label=f"train ({k}-step avg)")
    ep_steps = [r["step"] for r in epochs_rows]
    ax[0].plot(ep_steps, [r["val_loss"] for r in epochs_rows], "o-", label="validation")
    ax[0].set(xlabel="step", ylabel="cross-entropy (nats/char)", title="Loss")
    ax[0].legend()
    ax[1].plot([r["epoch"] for r in epochs_rows], [r["train_eval_loss"] for r in epochs_rows], "o-", label="train (eval mode)")
    ax[1].plot([r["epoch"] for r in epochs_rows], [r["val_loss"] for r in epochs_rows], "o-", label="validation")
    ax[1].set(xlabel="epoch", ylabel="cross-entropy", title="Per-epoch train vs validation")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "loss_curves.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 3.5))
    ax[0].plot(steps, [h["grad_norm"] for h in hist], lw=0.5)
    ax[0].set(xlabel="step", ylabel="grad L2 norm (pre-clip)", title="Gradient norm", yscale="log")
    ax[1].plot(steps, [h["lr"] for h in hist])
    ax[1].set(xlabel="step", ylabel="learning rate", title="LR schedule (warm-up + cosine)")
    fig.tight_layout()
    fig.savefig(out_dir / "grad_norm_and_lr.png", dpi=130)
    plt.close(fig)


def run(config_path):
    config_path = Path(config_path)
    cfg = yaml.safe_load(config_path.read_text())
    run_id = f"{cfg['run_name']}_{datetime.now().strftime('%Y%m%d-%H%M')}"
    log = RawLog(REPO_ROOT / "reproducibility" / "raw_logs" / MEMBER / f"task1_{run_id}.log")
    out_dir = MEMBER_DIR / "outputs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = MEMBER_DIR / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    set_seed(cfg["seed"])
    device = pick_device(cfg.get("device", "auto"))
    log(f"run_id={run_id} config={rel(config_path)} device={device} hardware={hardware_name(device)}")
    log(f"config: {json.dumps(cfg)}")

    # ---- data ----
    train_np, val_np, c2i, i2c = D.prepare(cfg, REPO_ROOT, MEMBER_DIR / cfg["data"]["processed_dir"], log)
    train = torch.from_numpy(train_np)
    val = torch.from_numpy(val_np)
    rng = np.random.default_rng(cfg["seed"])
    train_eval = train[rng.choice(len(train), size=min(len(val), len(train)), replace=False)]

    # ---- model ----
    m = cfg["model"]
    model = GPT(len(c2i), cfg["data"]["block_size"], m["d_model"], m["n_heads"], m["n_layers"], m["d_ff"], m["dropout"]).to(device)
    n_params = count_params(model)
    log(f"model: {m} params={n_params:,}")

    t = cfg["train"]
    opt = build_optimizer(model, t["lr"], t["weight_decay"])
    steps_per_epoch = math.ceil(len(train) / t["batch_size"])
    total_steps = steps_per_epoch * t["epochs"]
    log(f"steps/epoch={steps_per_epoch} total_steps={total_steps}")

    mem = PeakMemory(device)
    hist, epoch_rows = [], []
    ema, spikes, nan_steps = None, 0, 0
    best_val, best_path = float("inf"), ckpt_dir / f"{run_id}_best.pt"
    step, train_time, train_tokens = 0, 0.0, 0
    t_start = time.perf_counter()

    for epoch in range(1, t["epochs"] + 1):
        perm = torch.randperm(len(train))
        ep_loss, ep_n = 0.0, 0
        sync(device)
        t_ep = time.perf_counter()
        for i in range(0, len(train), t["batch_size"]):
            b = train[perm[i:i + t["batch_size"]]].to(device).long()
            x, y = b[:, :-1], b[:, 1:]
            lr = lr_at(step, total_steps, t["lr"], t["min_lr"], t["warmup_steps"])
            for g in opt.param_groups:
                g["lr"] = lr

            _, loss = model(x, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"]).item()
            lv = loss.item()
            if not (math.isfinite(lv) and math.isfinite(gnorm)):
                nan_steps += 1
                log(f"step {step}: non-finite loss/grad (loss={lv}, grad={gnorm}); skipping update")
                step += 1
                continue
            opt.step()

            # Spike = step loss more than `spike_ratio` x the running (EMA) loss, after warm-up.
            if ema is not None and step > t["warmup_steps"] and lv > t["spike_ratio"] * ema:
                spikes += 1
                log(f"step {step}: loss spike {lv:.4f} vs ema {ema:.4f}")
            ema = lv if ema is None else 0.98 * ema + 0.02 * lv
            hist.append({"step": step, "epoch": epoch, "loss": lv, "lr": lr, "grad_norm": gnorm})
            ep_loss += lv * len(b)
            ep_n += len(b)
            train_tokens += x.numel()
            if step % t["log_every"] == 0:
                mem.sample()
                log(f"ep {epoch} step {step}/{total_steps} loss {lv:.4f} lr {lr:.2e} grad {gnorm:.3f}")
            step += 1
        sync(device)
        train_time += time.perf_counter() - t_ep

        val_loss, val_acc = evaluate(model, val, t["eval_batch_size"], device)
        tr_loss, tr_acc = evaluate(model, train_eval, t["eval_batch_size"], device)
        row = {"epoch": epoch, "step": step, "train_loss_running": ep_loss / max(ep_n, 1),
               "train_eval_loss": tr_loss, "train_eval_acc": tr_acc, "val_loss": val_loss, "val_acc": val_acc,
               "val_ppl": math.exp(val_loss), "val_bpc": val_loss / math.log(2), "lr_end": lr}
        epoch_rows.append(row)
        log("epoch " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(), "config": cfg, "char_to_idx": c2i, "epoch": epoch,
                        "val_loss": val_loss, "run_id": run_id}, best_path)
            log(f"saved best checkpoint -> {rel(best_path)}")

    total_time = time.perf_counter() - t_start
    last_path = ckpt_dir / f"{run_id}_last.pt"
    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "config": cfg, "char_to_idx": c2i,
                "epoch": t["epochs"], "step": step, "run_id": run_id}, last_path)
    log(f"saved last checkpoint -> {rel(last_path)}")

    # ---- final evaluation with the best checkpoint ----
    best = torch.load(best_path, map_location=device)
    model.load_state_dict(best["model"])
    val_loss, val_acc = evaluate(model, val, t["eval_batch_size"], device)
    tr_loss, tr_acc = evaluate(model, train_eval, t["eval_batch_size"], device)

    samples, gen_tps = generate_samples(model, cfg, c2i, i2c, device)
    temp_txt = [s["continuation"] for s in samples if s["mode"] == "temperature"]
    greedy_txt = [s["continuation"] for s in samples if s["mode"] == "greedy"]
    mem_rep = mem.report_mb()
    gn = np.array([h["grad_norm"] for h in hist])

    temp = cfg["generate"]["temperature"]
    metrics = [
        ("train_cross_entropy", tr_loss, "best ckpt, eval mode, 10k-seq train subset (nats/char)"),
        ("train_cross_entropy_running_last_epoch", epoch_rows[-1]["train_loss_running"], "mean step loss over final epoch, dropout on"),
        ("val_cross_entropy", val_loss, "best ckpt, 10k val sequences (nats/char)"),
        ("val_perplexity", math.exp(val_loss), "exp(val CE)"),
        ("val_bits_per_char", val_loss / math.log(2), "val CE / ln 2"),
        ("generalization_gap", val_loss - tr_loss, "val CE - train CE (eval mode)"),
        ("val_top1_next_char_accuracy", val_acc, ""),
        ("train_top1_next_char_accuracy", tr_acc, ""),
        (f"distinct_1_temp{temp}", distinct_n(temp_txt, 1), "word-level, pooled over temperature samples"),
        (f"distinct_2_temp{temp}", distinct_n(temp_txt, 2), ""),
        (f"distinct_3_temp{temp}", distinct_n(temp_txt, 3), ""),
        (f"repeated_4gram_rate_temp{temp}", repeated_4gram_rate(temp_txt), "within-sample repeated word 4-grams"),
        ("distinct_1_greedy", distinct_n(greedy_txt, 1), ""),
        ("distinct_2_greedy", distinct_n(greedy_txt, 2), ""),
        ("distinct_3_greedy", distinct_n(greedy_txt, 3), ""),
        ("repeated_4gram_rate_greedy", repeated_4gram_rate(greedy_txt), ""),
        ("grad_norm_mean", float(gn.mean()), "pre-clip L2 norm over all steps"),
        ("grad_norm_max", float(gn.max()), ""),
        ("grad_norm_p99", float(np.percentile(gn, 99)), ""),
        ("grad_clip_fraction", float((gn > t["grad_clip"]).mean()), f"share of steps with norm > {t['grad_clip']}"),
        ("loss_spike_count", spikes, f"step loss > {t['spike_ratio']}x EMA after warm-up"),
        ("nan_inf_steps", nan_steps, ""),
        ("parameter_count", n_params, ""),
        ("train_tokens_per_sec", train_tokens / train_time, "training steps only, excludes eval"),
        ("generation_tokens_per_sec", gen_tps, "batch 1, full forward per token (no KV cache)"),
        ("peak_accelerator_memory_mb", mem_rep["peak_accelerator_mem_mb"], str(device)),
        ("peak_process_rss_mb", mem_rep["peak_process_rss_mb"], ""),
        ("total_training_time_sec", total_time, "train + per-epoch eval"),
        ("pure_training_time_sec", train_time, ""),
        ("epochs", t["epochs"], ""),
        ("best_epoch", best["epoch"], ""),
        ("hardware", hardware_name(device), ""),
        ("checkpoint", rel(best_path), ""),
        ("run_id", run_id, ""),
    ]
    for k, v, _ in metrics:
        log(f"metric {k} = {v}")

    # ---- artifacts ----
    with (out_dir / "metrics.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value", "notes"])
        w.writerows(metrics)
    if cfg.get("write_metrics_report"):
        (MEMBER_DIR / "metrics_report.csv").write_text((out_dir / "metrics.csv").read_text())
    with (out_dir / "train_history.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(hist[0].keys()))
        w.writeheader()
        w.writerows(hist)
    with (out_dir / "epoch_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(epoch_rows[0].keys()))
        w.writeheader()
        w.writerows(epoch_rows)
    (out_dir / "samples.json").write_text(json.dumps(samples, indent=1))
    with (out_dir / "samples.txt").open("w") as f:
        for s in samples:
            tag = "greedy" if s["mode"] == "greedy" else f"temperature={s['temperature']}"
            f.write(f"===== [{tag}] prompt: {s['prompt']!r}\n{s['text']}\n\n")
    save_plots(hist, epoch_rows, out_dir)

    write_manifest(REPO_ROOT / "reproducibility" / "manifests" / MEMBER / f"task1_{run_id}.json", cfg, device, {
        "task": "task1_llm", "member": MEMBER, "run_id": run_id,
        "config_file": rel(config_path),
        "raw_log": rel(log.path),
        "outputs_dir": rel(out_dir),
        "checkpoints": {"best (used for all reported metrics)": rel(best_path), "last": rel(last_path)},
        "processed_data": rel(MEMBER_DIR / cfg["data"]["processed_dir"]),
        "headline": {k: v for k, v, _ in metrics[:8]},
    })
    log(f"done: outputs -> {rel(out_dir)}")
    return {"run_id": run_id, "out_dir": out_dir, "metrics": {k: v for k, v, _ in metrics}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    run(ap.parse_args().config)
