"""Re-evaluate a saved CycleGAN checkpoint locally (all Task 3 metrics) and optionally rebuild
the Kaggle submission. Training already calls this automatically; use it to re-score an
intermediate checkpoint or to regenerate outputs after cloning the repo.

    python task3_gan/member_nikhil/evaluate_local.py --ckpt task3_gan/member_nikhil/checkpoints/<run_id>_epoch050.pt [--submission]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import evaluate  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--submission", action="store_true", help="also write the Kaggle images.zip")
a = ap.parse_args()
evaluate.run(a.ckpt, write_submission=a.submission)
