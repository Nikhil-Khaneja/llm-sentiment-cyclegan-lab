"""Evaluation metrics for binary sentiment predictions (y in {0,1}, p = P(positive))."""
import numpy as np
from scipy.stats import binomtest, chi2
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_recall_fscore_support, roc_auc_score)


def ece(y, p, n_bins=15):
    """Expected calibration error on the predicted-class confidence, equal-width bins."""
    pred = (p >= 0.5).astype(int)
    conf = np.where(pred == 1, p, 1 - p)
    correct = (pred == y).astype(float)
    bins = np.linspace(0.5, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, bins[1:-1]), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        sel = idx == b
        if sel.any():
            total += sel.mean() * abs(correct[sel].mean() - conf[sel].mean())
    return float(total)


def _fast_scores(y, pred):
    """acc, macro-F1, MCC from confusion counts; y/pred are [R, N] arrays of resamples."""
    tp = ((pred == 1) & (y == 1)).sum(1).astype(float)
    tn = ((pred == 0) & (y == 0)).sum(1).astype(float)
    fp = ((pred == 1) & (y == 0)).sum(1).astype(float)
    fn = ((pred == 0) & (y == 1)).sum(1).astype(float)
    n = tp + tn + fp + fn
    acc = (tp + tn) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        f1_pos = np.nan_to_num(2 * tp / (2 * tp + fp + fn))
        f1_neg = np.nan_to_num(2 * tn / (2 * tn + fn + fp))
        mcc = np.nan_to_num((tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return acc, (f1_pos + f1_neg) / 2, mcc


def bootstrap_ci(y, pred, n_boot=1000, seed=0, chunk=100):
    rng = np.random.default_rng(seed)
    out = {"accuracy": [], "macro_f1": [], "mcc": []}
    for s in range(0, n_boot, chunk):
        idx = rng.integers(0, len(y), size=(min(chunk, n_boot - s), len(y)))
        a, f, m = _fast_scores(y[idx], pred[idx])
        out["accuracy"].append(a); out["macro_f1"].append(f); out["mcc"].append(m)
    return {k: (float(np.percentile(np.concatenate(v), 2.5)), float(np.percentile(np.concatenate(v), 97.5)))
            for k, v in out.items()}


def mcnemar(y, pred_a, pred_b):
    """Paired test on the reviews where exactly one model is right.
    b = A right & B wrong, c = A wrong & B right."""
    ra, rb = pred_a == y, pred_b == y
    b, c = int((ra & ~rb).sum()), int((~ra & rb).sum())
    stat = (abs(b - c) - 1) ** 2 / (b + c) if b + c else 0.0
    return {"b_base_right_exp_wrong": b, "c_base_wrong_exp_right": c,
            "chi2_cc": float(stat), "p_chi2": float(chi2.sf(stat, 1)),
            "p_exact": float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0}


def full_report(y, p, n_boot=1000, seed=0):
    y = y.astype(int)
    pred = (p >= 0.5).astype(int)
    r = {"accuracy": accuracy_score(y, pred)}
    for avg in ["macro", "micro", "weighted"]:
        pr, rc, f1, _ = precision_recall_fscore_support(y, pred, average=avg, zero_division=0)
        r[f"precision_{avg}"], r[f"recall_{avg}"], r[f"f1_{avg}"] = pr, rc, f1
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    r.update({"cm_tn": int(tn), "cm_fp": int(fp), "cm_fn": int(fn), "cm_tp": int(tp)})
    r["roc_auc"] = roc_auc_score(y, p)
    r["pr_auc"] = average_precision_score(y, p)
    r["mcc"] = matthews_corrcoef(y, pred)
    r["brier"] = brier_score_loss(y, p)
    r["ece_15bin"] = ece(y, p)
    for k, (lo, hi) in bootstrap_ci(y, pred, n_boot, seed).items():
        r[f"{k}_ci95_low"], r[f"{k}_ci95_high"] = lo, hi
    return {k: float(v) for k, v in r.items()}


def slice_report(y, p, tags):
    """Macro-F1 and error rate for every slice column in `tags` (a DataFrame)."""
    y = y.astype(int)
    pred = (p >= 0.5).astype(int)
    rows = []
    for col in tags.columns:
        for val in sorted(tags[col].unique(), key=str):
            sel = (tags[col] == val).to_numpy()
            rows.append({"slice": col, "value": str(val), "n": int(sel.sum()),
                         "macro_f1": float(f1_score(y[sel], pred[sel], average="macro", zero_division=0)),
                         "error_rate": float((pred[sel] != y[sel]).mean())})
    return rows
