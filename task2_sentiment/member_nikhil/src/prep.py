"""Task 2 preprocessing: EDA, cleaning, stopword removal, stemming, split, vocabulary, encoding.

Heavy intermediate files (cleaned text, encoded arrays) are cached in data_processed/ and
git-ignored; the vocabulary, split sizes and EDA plots are small and committed.
"""
import html
import json
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split

PAD, UNK = 0, 1
# Words that flip or intensify sentiment. They are in the generic stopword list, but dropping
# them turns "not good" into "good", so they are kept when keep_negations is on.
NEGATION_KEEP = {"not", "no", "nor", "never", "nothing", "none", "nobody", "nowhere", "neither",
                 "cannot", "against", "very", "too", "few", "less", "least", "more", "most",
                 "but", "however", "although", "though", "only"}
NEGATION_WORDS = {"not", "no", "never", "nothing", "none", "nobody", "cannot", "nt"}
CONTRAST_WORDS = {"but", "however", "although", "though"}

_stem = lru_cache(maxsize=500_000)(PorterStemmer().stem)
_NON_ALPHA = re.compile(r"[^a-z\s]")
_WS = re.compile(r"\s+")


def clean(text: str, stopwords: frozenset, stem: bool):
    text = html.unescape(text).replace("\\n", " ").replace("\\\"", '"').lower()
    text = re.sub(r"n't\b", " not", text)          # don't -> do not, keeps the negation token
    text = _NON_ALPHA.sub(" ", text)               # punctuation, digits, special characters
    toks = [t for t in _WS.split(text) if t and t not in stopwords]
    if stem:
        toks = [_stem(t) for t in toks]
    return toks


def stopword_set(keep_negations: bool):
    sw = set(ENGLISH_STOP_WORDS)
    if keep_negations:
        sw -= NEGATION_KEEP
    return frozenset(sw)


def eda(df_train, df_test, out_dir: Path, log):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {}
    for name, df in [("train", df_train), ("test", df_test)]:
        wc = df["text"].fillna("").str.split().str.len()
        stats[name] = {
            "rows": int(len(df)),
            "missing_text": int(df["text"].isna().sum()),
            "empty_text": int((df["text"].fillna("").str.strip() == "").sum()),
            "duplicate_text": int(df["text"].duplicated().sum()),
            "class_counts": {int(k): int(v) for k, v in df["label"].value_counts().sort_index().items()},
            "positive_share": float(df["label"].mean()),
            "words_mean": float(wc.mean()), "words_median": float(wc.median()),
            "words_p95": float(wc.quantile(0.95)), "words_max": int(wc.max()),
        }
    stats["train_test_text_overlap"] = int(df_test["text"].isin(set(df_train["text"])).sum())

    wc = df_train["text"].str.split().str.len()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for lab, name in [(0, "negative"), (1, "positive")]:
        ax[0].hist(wc[df_train["label"] == lab].clip(upper=1000), bins=80, alpha=0.6, label=name)
    ax[0].set(xlabel="words per review (clipped at 1000)", ylabel="reviews", title="Review length by class (train)")
    ax[0].legend()
    counts = df_train["label"].value_counts().sort_index()
    ax[1].bar(["negative (0)", "positive (1)"], counts.values)
    ax[1].set(title="Class distribution (train)", ylabel="reviews")
    fig.tight_layout()
    fig.savefig(out_dir / "eda_length_and_class.png", dpi=120)
    plt.close(fig)
    (out_dir / "eda_stats.json").write_text(json.dumps(stats, indent=1))
    log(f"eda: {json.dumps(stats)}")
    return stats


def build_vocab(token_lists, size, min_freq):
    cnt = Counter(t for toks in token_lists for t in toks)
    words = [w for w, c in cnt.most_common(size - 2) if c >= min_freq]
    return {"<pad>": PAD, "<unk>": UNK, **{w: i + 2 for i, w in enumerate(words)}}, cnt


def encode(token_lists, vocab, max_len):
    arr = np.zeros((len(token_lists), max_len), dtype=np.int32)
    lengths = np.zeros(len(token_lists), dtype=np.int32)
    for i, toks in enumerate(token_lists):
        ids = [vocab.get(t, UNK) for t in toks[:max_len]]  # truncate long reviews from the end
        arr[i, :len(ids)] = ids
        lengths[i] = max(len(ids), 1)                       # empty -> one <pad>, never length 0
    return arr, lengths


def slice_tags(raw_texts, token_lists):
    """Per-review slice labels used for robustness metrics and slice-specific error picks."""
    wc = np.array([len(t.split()) for t in raw_texts])
    length = pd.cut(wc, bins=[0, 50, 100, 200, 10**6], labels=["len<=50", "len51-100", "len101-200", "len>200"], include_lowest=True)
    neg = np.array([bool(NEGATION_WORDS & set(t)) for t in token_lists])
    contrast = np.array([bool(CONTRAST_WORDS & set(t)) for t in token_lists])
    return pd.DataFrame({"length_bucket": length.astype(str), "has_negation": neg, "has_contrast": contrast})


def prepare(cfg, repo_root: Path, member_dir: Path, log):
    d = cfg["data"]
    cache = member_dir / d["processed_dir"]
    cache.mkdir(parents=True, exist_ok=True)
    raw_dir = repo_root / d["raw_dir"]
    tr = pd.read_csv(raw_dir / "train.csv")
    te = pd.read_csv(raw_dir / "test.csv")
    eda_stats = eda(tr, te, member_dir / "outputs" / "eda", log)

    # Missing / malformed rows: drop NaN or blank text, invalid labels, and exact duplicates.
    def sanitize(df, name):
        n0 = len(df)
        df = df.dropna(subset=["text", "label"])
        df = df[df["text"].str.strip() != ""]
        df = df[df["label"].isin([0, 1])]
        df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
        log(f"{name}: dropped {n0 - len(df)} missing/malformed/duplicate rows ({len(df):,} left)")
        return df

    tr, te = sanitize(tr, "train"), sanitize(te, "test")
    if d.get("max_train_samples"):
        tr = tr.sample(n=min(d["max_train_samples"], len(tr)), random_state=cfg["seed"]).reset_index(drop=True)
    if d.get("max_test_samples"):
        te = te.sample(n=min(d["max_test_samples"], len(te)), random_state=cfg["seed"]).reset_index(drop=True)

    sw = stopword_set(d["keep_negations"])
    log(f"cleaning {len(tr) + len(te):,} reviews (stem={d['stem']}, keep_negations={d['keep_negations']})")
    tr_tok = [clean(t, sw, d["stem"]) for t in tr["text"]]
    te_tok = [clean(t, sw, d["stem"]) for t in te["text"]]

    idx_tr, idx_val = train_test_split(np.arange(len(tr)), test_size=d["val_frac"], stratify=tr["label"], random_state=cfg["seed"])
    vocab, cnt = build_vocab([tr_tok[i] for i in idx_tr], d["vocab_size"], d["min_freq"])
    lens = np.array([len(t) for t in tr_tok])
    log(f"vocab={len(vocab):,} (of {len(cnt):,} distinct train tokens); tokens/review mean={lens.mean():.1f} "
        f"p95={np.percentile(lens, 95):.0f}; truncating at max_len={d['max_len']} "
        f"({(lens > d['max_len']).mean():.2%} of reviews truncated)")

    X_all, L_all = encode(tr_tok, vocab, d["max_len"])
    X_te, L_te = encode(te_tok, vocab, d["max_len"])
    y_all = tr["label"].to_numpy(np.float32)
    splits = {
        "train": (X_all[idx_tr], L_all[idx_tr], y_all[idx_tr]),
        "val": (X_all[idx_val], L_all[idx_val], y_all[idx_val]),
        "test": (X_te, L_te, te["label"].to_numpy(np.float32)),
    }
    test_meta = slice_tags(te["text"].tolist(), te_tok)
    test_meta["text"] = te["text"].values
    test_meta["clean"] = [" ".join(t) for t in te_tok]

    (cache / "vocab.json").write_text(json.dumps(vocab))
    split_info = {k: int(len(v[2])) for k, v in splits.items()}
    unk_rate = float((X_te == UNK).sum() / max((X_te > 0).sum(), 1))
    (cache / "split_stats.json").write_text(json.dumps({**split_info, "vocab_size": len(vocab), "test_unk_rate": unk_rate,
                                                        "eda": eda_stats}, indent=1))
    log(f"splits: {split_info} test_unk_rate={unk_rate:.3%}")
    return splits, test_meta, vocab
