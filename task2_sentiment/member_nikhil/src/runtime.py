"""Run bookkeeping: device choice, seeding, raw logs, peak memory, and manifests."""
import json
import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

try:  # POSIX only; Windows falls back to psutil
    import resource
except ImportError:
    resource = None

REPO_ROOT = Path(__file__).resolve().parents[3]


def pick_device(pref="auto"):
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def hardware_name(device):
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if platform.system() == "Darwin":
        try:
            chip = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        except Exception:
            chip = platform.processor()
        return f"{chip} ({device.type})"
    return f"{platform.processor() or platform.machine()} ({device.type})"


class RawLog:
    """Append-only log of everything a run prints. Never edit these files afterwards."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = path.open("a", encoding="utf-8")
        self.path = path

    def __call__(self, msg):
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()


class PeakMemory:
    """Tracks peak accelerator memory (sampled) plus peak process RSS."""

    def __init__(self, device):
        self.device = device
        self.peak_accel = 0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

    def sample(self):
        if self.device.type == "mps":
            self.peak_accel = max(self.peak_accel, torch.mps.driver_allocated_memory())
        elif self.device.type == "cuda":
            self.peak_accel = max(self.peak_accel, torch.cuda.max_memory_allocated(self.device))

    def report_mb(self):
        self.sample()
        if resource is not None:
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            rss_mb = rss / 2**20 if sys.platform == "darwin" else rss / 2**10  # bytes on macOS, KB on Linux
        else:
            import psutil  # peak working set on Windows
            rss_mb = psutil.Process().memory_info().peak_wset / 2**20
        return {"peak_accelerator_mem_mb": round(self.peak_accel / 2**20, 1), "peak_process_rss_mb": round(rss_mb, 1)}


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def write_manifest(path: Path, cfg: dict, device, extra: dict):
    import importlib.metadata as md

    pkgs = {}
    for name in ["torch", "numpy", "pandas", "matplotlib", "scikit-learn", "scipy", "datasets", "nltk", "pyyaml"]:
        try:
            pkgs[name] = md.version(name)
        except md.PackageNotFoundError:
            pass
    manifest = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "git_commit_at_launch": git_commit(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hardware": hardware_name(device),
        "device": str(device),
        "packages": pkgs,
        "config": cfg,
        **extra,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, default=str))


def rel(p: Path) -> str:
    """Repo-relative path string, so nothing personal ends up in committed files."""
    try:
        return str(Path(p).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.elapsed = time.perf_counter() - self.t0
