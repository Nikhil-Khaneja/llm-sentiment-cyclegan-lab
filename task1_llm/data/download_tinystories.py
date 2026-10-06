"""Download a fixed subset of TinyStories (shared raw data for Task 1).

Streams the first N stories of the HuggingFace `roneneldan/TinyStories` train split and
writes them to `tinystories_raw.txt` next to this script, one story per block separated by
a blank line. Each member builds their own char vocab / train-val split from this file.

Usage (from repo root):
    python task1_llm/data/download_tinystories.py            # default 40,000 stories
    python task1_llm/data/download_tinystories.py --n 5000   # smaller, for smoke tests
"""
import argparse
import os
from pathlib import Path

from datasets import load_dataset

OUT = Path(__file__).resolve().parent / "tinystories_raw.txt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40_000, help="number of stories to keep")
    args = ap.parse_args()

    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    n_chars = 0
    with OUT.open("w", encoding="utf-8") as f:
        for i, row in enumerate(ds):
            if i >= args.n:
                break
            story = row["text"].strip()
            f.write(story + "\n\n")
            n_chars += len(story) + 2
    print(f"wrote {i} stories, {n_chars:,} chars -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
    # `datasets` streaming leaves background fetch threads alive that block interpreter
    # shutdown; the file is already closed at this point, so exit hard.
    os._exit(0)
