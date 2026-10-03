"""Shared decision rule and the challenge metric (macro F0.5 per S1 entity).

Decision rule applied to model probabilities:
  1. one-parent: each S2/S3 record is kept only for the S1 record where its
     probability is highest (the training data shows no record has two parents),
  2. keep a pair if its probability >= t,
  3. top-1 floor: if an S1 record kept nothing, still keep its best remaining pair
     when that probability >= t1 (an empty list scores 0 for the 94% of S1 records
     that do have matches, so a moderately likely top-1 is worth including).
"""
import numpy as np
import pandas as pd


def one_parent(df):
    """Boolean mask: row is the highest-probability S1 for its (src, c_row) candidate."""
    ckey = df["src"].to_numpy().astype(np.int64) * 10**7 + df["c_row"].to_numpy()
    prob = df["prob"].to_numpy()
    order = np.lexsort((-prob, ckey))
    first = np.r_[True, ckey[order][1:] != ckey[order][:-1]]
    keep = np.zeros(len(df), bool)
    keep[order[first]] = True
    return keep


def decide(df, t, t1=None, parent_mask=None):
    """Boolean mask of predicted matches for rows of df (needs s1_row, src, c_row, prob)."""
    base = one_parent(df) if parent_mask is None else parent_mask
    prob = df["prob"].to_numpy()
    sel = base & (prob >= t)
    if t1 is not None and t1 < t:
        s1 = df["s1_row"].to_numpy()
        cand = base & ~sel & (prob >= t1)
        has = np.zeros(s1.max() + 1, bool)
        has[s1[sel]] = True
        cand &= ~has[s1]
        idx = np.flatnonzero(cand)
        if len(idx):
            order = idx[np.lexsort((-prob[idx], s1[idx]))]
            first = np.r_[True, s1[order][1:] != s1[order][:-1]]
            sel[order[first]] = True
    return sel


def macro_f05(s1_eval, true_count, pred_s1, pred_is_true):
    """Macro F0.5 over the S1 rows in s1_eval.
    true_count[s1_row] = number of ground-truth matches (full ground truth, not just candidates);
    pred_s1 / pred_is_true = S1 row and correctness of every predicted pair."""
    n = len(true_count)
    npred = np.bincount(pred_s1, minlength=n)[s1_eval]
    tp = np.bincount(pred_s1, weights=pred_is_true.astype(np.float64), minlength=n)[s1_eval]
    T = true_count[s1_eval]
    prec = np.divide(tp, npred, out=np.zeros(len(tp)), where=npred > 0)
    rec = np.divide(tp, T, out=np.zeros(len(tp)), where=T > 0)
    denom = 0.25 * prec + rec
    f = np.divide(1.25 * prec * rec, denom, out=np.zeros(len(tp)), where=denom > 0)
    f = np.where((T == 0) & (npred == 0), 1.0, f)
    return {"f05": float(f.mean()),
            "precision_micro": float(tp.sum() / max(npred.sum(), 1)),
            "recall_micro": float(tp.sum() / max(T.sum(), 1)),
            "empty_pred_frac": float((npred == 0).mean()),
            "singleton_frac": float((T == 0).mean()),
            "per_entity": f}
