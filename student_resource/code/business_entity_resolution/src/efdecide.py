"""Per-record expected-F0.5 decision rule (instead of one global threshold), optionally on the
averaged probabilities of several runs that share one candidate set.

The metric scores every S1 record separately:  F0.5 = 1.25 * TP / (0.25 * T + m)  for m
predicted and T true matches (1.0 when both are 0). Given calibrated probabilities p_i of the
candidates of one S1 record, the expected score of keeping its top-m pairs is approximated by
    EF(m) = 1.25 * sum_{i<=m} p_i / (0.25 * E[T] + m),     E[T] = sum of p over all its candidates
and the expected score of an empty list by
    EF(0) = s * prod_i (1 - p_i)                            (probability of a singleton)
The record keeps its top m* = argmax_m EF(m) pairs (with pairs sorted by p, the best set is
always a top-m set). Pairs that the one-parent rule gives to another S1 record are never kept
but still count in E[T] and EF(0). Unlike one threshold this adapts per record: a lone 0.6
candidate is kept (EF 0.65 vs 0.40 for an empty list), a 0.6 candidate next to two 0.99 ones is
not (0.88 vs 0.94).

Probabilities are recalibrated as p -> sigmoid(a * logit(p) + b); a, b, s (and the run weights
when several tags are given) are tuned on validation fold C with the exact macro F0.5, next to
the usual threshold rule. The best rule overall is applied to the test pairs. If the best is
the plain threshold rule of a single run (i.e. nothing better than that run's own output),
nothing is written.

Input : per tag, models/lgb_<tag>.txt + decision_<tag>.json, feat/<tag>/train.parquet (fold-C
        rows), pred/<tag>_test.parquet (written by predict.py)
Output: OUT_DIR/matching_results.tsv, candidate_pairs.tsv, history/<out-tag>_<time>/, validator
Usage:  python src/efdecide.py --tags labse2k --out-tag labse2k_ef
        python src/efdecide.py --tags labse2k,labse2kb --out-tag labse2k_ens_ef
"""
import argparse
import itertools
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from common import load_ids
from decision import decide, macro_f05, one_parent
from ensemble import KEYS, aligned, best_threshold, val_probs
from predict import id_lists, write_tsv
from train_match import true_counts

A_GRID = [0.75, 1.0, 1.25, 1.5, 2.0]
B_GRID = [-1.5, -1.0, -0.5, 0.0, 0.5]
S_GRID = [0.8, 1.0, 1.25]


class Groups:
    """Rows sorted by (S1 row, probability desc) once; calibration is monotonic, so the order
    never changes during the grid search."""

    def __init__(self, s1, prob, parent):
        self.order = np.lexsort((-prob, s1))
        s = s1[self.order]
        self.n = len(s)
        self.starts = np.r_[0, np.flatnonzero(s[1:] != s[:-1]) + 1] if self.n else np.zeros(0, np.int64)
        self.gid = np.repeat(np.arange(len(self.starts)), np.diff(np.r_[self.starts, self.n]))
        self.parent = parent[self.order]
        cc = np.cumsum(self.parent, dtype=np.int64)
        off = np.r_[0, cc[self.starts[1:] - 1]] if len(self.starts) else np.zeros(0, np.int64)
        self.cc = cc - off[self.gid]                    # eligible pairs up to and including this row

    def select(self, prob, a, b, s):
        """Boolean mask (original row order) of the pairs kept by the expected-F0.5 rule."""
        p = np.clip(prob[self.order].astype(np.float64), 1e-7, 1 - 1e-7)
        if a != 1.0 or b != 0.0:
            p = 1.0 / (1.0 + np.exp(-(a * np.log(p / (1 - p)) + b)))
        st, g = self.starts, self.gid
        et = np.add.reduceat(p, st)
        ef0 = s * np.exp(np.add.reduceat(np.log1p(-np.minimum(p, 1 - 1e-7)), st))
        cq = np.cumsum(np.where(self.parent, p, 0.0))
        cq -= np.r_[0.0, cq[st[1:] - 1]][g]
        ef = np.where(self.parent, 1.25 * cq / (0.25 * et[g] + np.maximum(self.cc, 1)), -1.0)
        best = np.maximum.reduceat(ef, st)
        big = np.iinfo(np.int64).max
        m_best = np.minimum.reduceat(np.where(self.parent & (ef >= best[g] - 1e-12), self.cc, big), st)
        mstar = np.where(best > ef0, m_best, 0)
        keep_sorted = self.parent & (self.cc <= mstar[g])
        keep = np.zeros(self.n, bool)
        keep[self.order] = keep_sorted
        return keep


def weight_grid(k):
    if k == 1:
        return [(1.0,)]
    combos = [tuple(float(i == j) for j in range(k)) for i in range(k)]
    combos.append(tuple([1.0 / k] * k))
    for w in itertools.product(np.arange(0, 1.01, 0.25), repeat=k):
        w = tuple(float(x) for x in w)
        if abs(sum(w) - 1) < 1e-6 and w not in combos:
            combos.append(w)
    return combos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.environ.get("DATA_DIR"))
    ap.add_argument("--work-dir", default=os.environ.get("WORK_DIR"))
    ap.add_argument("--out-dir", default=os.environ.get("OUT_DIR"))
    ap.add_argument("--tags", required=True, help="comma list of run tags sharing one candidate set")
    ap.add_argument("--out-tag", default="ef")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    ap.add_argument("--min-gain", type=float, default=0.0002,
                    help="write the test output only if validation F0.5 beats the reference by this much")
    ap.add_argument("--rule", choices=["auto", "expected-F"], default="auto",
                    help="expected-F: always apply the best expected-F setting (no min-gain check)")
    args = ap.parse_args()
    tags = args.tags.split(",")
    t0 = time.time()

    # ------------------------------------------------------------ validation
    vals = [val_probs(args.work_dir, t, args.threads) for t in tags]
    base, P = aligned(vals)
    lab = vals[0]["label"].to_numpy()
    del vals
    tc, _ = true_counts(args.data_dir, args.work_dir)
    s1_eval = np.union1d(base["s1_row"].unique(),
                         np.load(os.path.join(args.work_dir, "feat", tags[0], "val_s1_rows.npy")))
    s1v = base["s1_row"].to_numpy()
    print(f"validation: {len(base):,} pairs, {len(s1_eval):,} S1 entities, runs {tags} ({time.time() - t0:.0f}s)")

    rows = []
    for w in weight_grid(len(tags)):
        base["prob"] = (P @ np.array(w, np.float32)).astype(np.float32)
        prob = base["prob"].to_numpy()
        parent = one_parent(base)
        t, t1, f, pr, rc = best_threshold(base.assign(label=lab), s1_eval, tc, parent)
        rows.append({"weights": w, "rule": "threshold", "t": t, "t1": t1, "a": None, "b": None, "s": None,
                     "f05": f, "precision": pr, "recall": rc})
        thr_f, ef_best = f, 0.0
        grp = Groups(s1v, prob, parent)
        for a, b, s in itertools.product(A_GRID, B_GRID, S_GRID):
            sel = grp.select(prob, a, b, s)
            m = macro_f05(s1_eval, tc, s1v[sel], lab[sel])
            rows.append({"weights": w, "rule": "expected-F", "t": None, "t1": None, "a": a, "b": b, "s": s,
                         "f05": m["f05"], "precision": m["precision_micro"], "recall": m["recall_micro"]})
            ef_best = max(ef_best, m["f05"])
        del grp
        print(f"  weights {w}: threshold rule (t={t}, t1={t1}) {thr_f:.4f} | best expected-F rule {ef_best:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    res = pd.DataFrame(rows).sort_values("f05", ascending=False).reset_index(drop=True)
    pd.set_option("display.width", 220)
    print("\nvalidation macro F0.5, best 15 settings:")
    print(res.head(15).to_string(index=False))
    ref = res[(res["rule"] == "threshold") & res["weights"].map(lambda w: w == (1.0,) + (0.0,) * (len(tags) - 1))]
    ref_f = float(ref["f05"].iloc[0])
    best = res[res["rule"] == "expected-F"].iloc[0] if args.rule == "expected-F" else res.iloc[0]
    print(f"\nreference ({tags[0]} alone, threshold rule): {ref_f:.4f}")
    print(f"BEST {best['rule']} weights {dict(zip(tags, best['weights']))} "
          f"{'t=%s t1=%s' % (best['t'], best['t1']) if best['rule'] == 'threshold' else 'a=%s b=%s s=%s' % (best['a'], best['b'], best['s'])}: "
          f"macro F0.5 {best['f05']:.4f} | precision {best['precision']:.4f} | recall {best['recall']:.4f} "
          f"(gain {best['f05'] - ref_f:+.4f})")
    if args.rule == "auto" and best["f05"] - ref_f < args.min_gain:
        print(f"gain below {args.min_gain}: keep {tags[0]}'s existing output. Nothing written.")
        return

    # ------------------------------------------------------------ test
    tests = [pd.read_parquet(os.path.join(args.work_dir, "pred", f"{tg}_test.parquet"), columns=KEYS + ["prob"])
             for tg in tags]
    tb, TP = aligned(tests)
    del tests
    tb["prob"] = (TP @ np.array(best["weights"], np.float32)).astype(np.float32)
    del TP
    parent = one_parent(tb)
    if best["rule"] == "threshold":
        sel = decide(tb, float(best["t"]), None if pd.isna(best["t1"]) else float(best["t1"]), parent)
    else:
        grp = Groups(tb["s1_row"].to_numpy(), tb["prob"].to_numpy(), parent)
        sel = grp.select(tb["prob"].to_numpy(), float(best["a"]), float(best["b"]), float(best["s"]))
        del grp
    print(f"test: {len(tb):,} pairs, {sel.sum():,} matches ({time.time() - t0:.0f}s)")

    s1 = load_ids(args.data_dir, "test", "source1")
    ids = {s: load_ids(args.data_dir, "test", f"source{s}")["entity_id"].to_numpy() for s in (2, 3)}
    src, crow, s1r = tb["src"].to_numpy(), tb["c_row"].to_numpy(), tb["s1_row"].to_numpy()
    cid = np.where(src == 2, ids[2][np.where(src == 2, crow, 0)], ids[3][np.where(src == 3, crow, 0)])
    n_s1 = len(s1)
    cands = id_lists(s1r, cid, n_s1)
    matches = id_lists(s1r[sel], cid[sel], n_s1, order_key=-tb["prob"].to_numpy()[sel])
    n_match = np.array([0 if m == "" else m.count(",") + 1 for m in matches])
    print(f"S1 entities {n_s1:,} | with >=1 match {np.mean(n_match > 0):.2%} | mean matches {n_match.mean():.2f}")
    for c in sorted(s1["country"].unique()):
        m = s1["country"].to_numpy() == c
        print(f"  {c:>8s}: {m.sum():>9,} S1 | empty {np.mean(n_match[m] == 0):6.2%} | mean {n_match[m].mean():.2f}")

    os.makedirs(args.out_dir, exist_ok=True)
    mpath = os.path.join(args.out_dir, "matching_results.tsv")
    cpath = os.path.join(args.out_dir, "candidate_pairs.tsv")
    write_tsv(mpath, "matched_entity_ids", s1["entity_id"], matches)
    write_tsv(cpath, "candidate_entity_ids", s1["entity_id"], cands)
    hist = os.path.join(args.out_dir, "history", f"{args.out_tag}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(hist, exist_ok=True)
    for p in (mpath, cpath):
        shutil.copy2(p, hist)
    info = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in best.to_dict().items()}
    info.update(tags=tags, weights=list(best["weights"]), reference_f05=ref_f)
    with open(os.path.join(hist, "decision.json"), "w") as f:
        json.dump(info, f, indent=2, default=float)
    tb.assign(selected=sel.astype(np.int8)).to_parquet(
        os.path.join(args.work_dir, "pred", f"{args.out_tag}_test.parquet"), index=False)
    print(f"wrote {mpath}\n      copy in {hist}")

    root = os.path.dirname(os.path.abspath(args.data_dir))
    validator = os.path.join(root, "utils", "validate_submission.py")
    r = subprocess.run([sys.executable, validator, "--matching", mpath, "--candidate", cpath,
                        "--test-dir", os.path.join(args.data_dir, "test")], capture_output=True, text=True)
    print("\n--- validate_submission.py ---\n" + r.stdout[-600:] + r.stderr[-600:])
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
