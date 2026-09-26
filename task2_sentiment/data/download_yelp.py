"""Download Yelp Review Polarity (shared raw data for Task 2).

Writes `train.csv` (560,000 rows) and `test.csv` (38,000 rows) next to this script with
columns `label` (0 = negative, 1 = positive) and `text`. Each member does their own
cleaning / tokenisation / vocabulary from these files.

Usage (from repo root):
    python task2_sentiment/data/download_yelp.py
"""
import os
from pathlib import Path

from datasets import load_dataset

OUT = Path(__file__).resolve().parent


def main():
    ds = load_dataset("fancyzhx/yelp_polarity")
    for split in ["train", "test"]:
        path = OUT / f"{split}.csv"
        ds[split].to_pandas()[["label", "text"]].to_csv(path, index=False)
        print(f"{split}: {len(ds[split]):,} rows -> {path}", flush=True)


if __name__ == "__main__":
    main()
    os._exit(0)
