"""Task 1 data pipeline: clean TinyStories text, build a character vocabulary, and cut
fixed-length (input, target) sequences for next-character prediction.

Split is done at the *story* level before chunking so no story contributes text to both
the train and validation sets.
"""
import json
import random
from pathlib import Path

import numpy as np

# TinyStories contains UTF-8 text that was mis-decoded as cp1252 ("mojibake"), e.g. the
# apostrophe in "don't" appears as "donâ€™t". Map the common sequences back to ASCII.
MOJIBAKE = {
    "â€™": "'", "â€˜": "'", "â€œ": '"', "â€\x9d": '"', "â€": '"',
    "â€”": "-", "â€“": "-", "â€¦": "...", "Ã©": "e", "Â": "",
}
SMART = {"’": "'", "‘": "'", "“": '"', "”": '"', "—": "-",
         "–": "-", "…": "...", "\xa0": " ", "\t": " "}
ALLOWED = set(chr(c) for c in range(32, 127)) | {"\n"}


def clean_text(text: str) -> str:
    for bad, good in {**MOJIBAKE, **SMART}.items():
        text = text.replace(bad, good)
    # Anything still outside printable ASCII is rare noise; drop it rather than give it a
    # vocabulary slot the model will almost never see.
    return "".join(ch for ch in text if ch in ALLOWED)


def load_stories(raw_path: Path):
    text = clean_text(raw_path.read_text(encoding="utf-8"))
    # Stories were written separated by a blank line, but stories themselves also contain
    # blank lines between paragraphs. Paragraph-level units are fine for chunking.
    return [s.strip() for s in text.split("\n\n") if s.strip()]


def build_vocab(text: str):
    chars = sorted(set(text))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for ch, i in char_to_idx.items()}
    return char_to_idx, idx_to_char


def encode(s, char_to_idx):
    return [char_to_idx[c] for c in s]


def decode(ids, idx_to_char):
    return "".join(idx_to_char[int(i)] for i in ids)


def chunk(text: str, block: int, n: int, rng: random.Random):
    """Cut non-overlapping windows of block+1 chars; x = w[:-1], y = w[1:]."""
    w = block + 1
    starts = list(range(0, len(text) - w, w))
    if len(starts) < n:
        raise ValueError(f"only {len(starts)} windows available, need {n}; download more stories")
    rng.shuffle(starts)
    return [text[s:s + w] for s in starts[:n]]


def prepare(cfg: dict, repo_root: Path, out_dir: Path, log=print):
    d = cfg["data"]
    rng = random.Random(cfg["seed"])
    stories = load_stories(repo_root / d["raw_path"])
    rng.shuffle(stories)
    n_val_stories = int(len(stories) * d["val_story_frac"])
    val_text = "\n".join(stories[:n_val_stories])
    train_text = "\n".join(stories[n_val_stories:])

    char_to_idx, idx_to_char = build_vocab(train_text + val_text)
    train_w = chunk(train_text, d["block_size"], d["n_train"], rng)
    val_w = chunk(val_text, d["block_size"], d["n_val"], rng)
    train = np.array([encode(s, char_to_idx) for s in train_w], dtype=np.uint8)
    val = np.array([encode(s, char_to_idx) for s in val_w], dtype=np.uint8)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "sequences.npz", train=train, val=val)
    (out_dir / "vocab.json").write_text(json.dumps({"char_to_idx": char_to_idx}, indent=1))
    stats = {
        "stories": len(stories), "train_chars_pool": len(train_text), "val_chars_pool": len(val_text),
        "vocab_size": len(char_to_idx), "train_seqs": len(train), "val_seqs": len(val),
        "block_size": d["block_size"],
    }
    (out_dir / "data_stats.json").write_text(json.dumps(stats, indent=1))
    log(f"data: {stats}")
    return train, val, char_to_idx, idx_to_char
