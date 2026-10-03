"""Ensemble of LightGBM runs that share one candidate set: average their probabilities,
tune the decision rule on validation fold C, and write the test submission.

The runs must use the same candidate tag and cut-off (e.g. labse2g, labse2h, labse2i: all
built from cand/labse2 with k_fwd 15 / k_rev 5), so their pair lists are identical; this is
checked. Validation probabilities are recomputed from each run's model on its own fold-C
rows; test probabilities are read from WORK_DIR/pred/<tag>_test.parquet (written by predict.py).

Weights: every run alone, the plain average, and a grid over weight vectors (step 0.25).
For each, the threshold is tuned for macro F0.5 with the one-parent rule, exactly as in
train_match.py; the best combination is applied to the test set.

Output: OUT_DIR/matching_results.tsv, candidate_pairs.tsv, history/<out-tag>_<time>/, validator.
Usage:  python src/ensemble.py --tags labse2g,labse2h,labse2i --out-tag ens3
"""
import argparse
import itertools
import json
import os
import shutil
import subprocess
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from common import load_ids
from decision import decide, macro_f05, one_parent
from predict import id_lists, write_tsv
from train_match import true_counts

KEYS = ["s1_row", "src", "c_row"]


def aligned(frames):
    """Stack the prob columns of frames that list the same pairs (same order is checked;
    falls back to a key merge if the order differs)."""
    base = frames[0][KEYS].reset_index(drop=True)
    probs = [frames[0]["prob"].to_numpy()]
    for f in frames[1:]:
        same = len(f) == len(base) and all(np.array_equal(f[k].to_numpy(), base[k].to_numpy()) for k in KEYS)
        if same:
            probs.append(f["prob"].to_numpy())
        else:
            m = base.merge(f[KEYS + ["prob"]], on=KEYS, how="left")
            if m["prob"].isna().any():
                sys.exit("runs do not share the same candidate pairs; ensemble needs identical candidate sets")
            probs.append(m["prob"].to_numpy())
    return base, np.vstack(probs).T.astype(np.float32)


def val_probs(work_dir, tag, threads):
    with open(os.path.join(work_dir, "models", f"decision_{tag}.json")) as f:
        feats = json.load(f)["features"]
    model = lgb.Booster(model_file=os.path.join(work_dir, "models", f"lgb_{tag}.txt"))
    cols = list(dict.fromkeys(KEYS + ["is_val", "label"] + feats))
    va = pd.read_parquet(os.path.join(work_dir, "feat", tag, "train.parquet"), columns=cols,
                         filters=[("is_val", "==", 1)])          # fold-C rows only (saves memory)
    va = va[va["is_val"] == 1].reset_index(drop=True)
    va["prob"] = model.predict(va[feats].to_numpy(dtype=np.float32), num_threads=threads)
    return va[KEYS + ["label", "prob"]]


def best_threshold(df, s1_eval, tc, parent):
    """Same grid as train_match.py: threshold t and optional top-1 floor t1."""
    lab, s1r = df["label"].to_numpy(), df["s1_row"].to_numpy()
    best = None
    for t in np.round(np.arange(0.20, 0.96, 0.05), 2):
        for t1 in [None, 0.05, 0.10, 0.15, 0.20, 0.30]:
            if t1 is not None and t1 >= t:
                continue
            sel = decide(df, t, t1, parent)
            m = macro_f05(s1_eval, tc, s1r[sel], lab[sel])
            if best is None or m["f05"] > best[2]:
                best = (float(t), t1, m["f05"], m["precision_micro"], m["recall_micro"])
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.environ.get("DATA_DIR"))
    ap.add_argument("--work-dir", default=os.environ.get("WORK_DIR"))
    ap.add_argument("--out-dir", default=os.environ.get("OUT_DIR"))
    ap.add_argument("--tags", required=True, help="comma list of run tags")
    ap.add_argument("--out-tag", default="ens")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    args = ap.parse_args()
    tags = args.tags.split(",")
    t0 = time.time()

    # ------------------------------------------------------------ validation
    vals = [val_probs(args.work_dir, t, args.threads) for t in tags]
    base, P = aligned(vals)
    base["label"] = vals[0]["label"].to_numpy()
    tc, _ = true_counts(args.data_dir, args.work_dir)
    s1_eval = np.union1d(base["s1_row"].unique(),
                         np.load(os.path.join(args.work_dir, "feat", tags[0], "val_s1_rows.npy")))
    print(f"validation: {len(base):,} pairs, {len(s1_eval):,} S1 entities, {len(tags)} runs "
          f"({time.time() - t0:.0f}s)")

    combos = [tuple(float(i == j) for j in range(len(tags))) for i in range(len(tags))]
    combos.append(tuple([1.0 / len(tags)] * len(tags)))
    grid = [w for w in itertools.product(np.arange(0, 1.01, 0.25), repeat=len(tags)) if abs(sum(w) - 1) < 1e-6]
    combos += [tuple(float(x) for x in w) for w in grid if tuple(float(x) for x in w) not in combos]
    rows = []
    for w in combos:
        base["prob"] = P @ np.array(w, np.float32)
        parent = one_parent(base)
        t, t1, f, p, r = best_threshold(base, s1_eval, tc, parent)
        rows.append((w, t, t1, f, p, r))
    res = pd.DataFrame(rows, columns=["weights", "t", "t1", "f05", "precision", "recall"]).sort_values("f05", ascending=False)
    pd.set_option("display.width", 200)
    print("\nvalidation macro F0.5 (single runs first rows of the grid, best on top):")
    print(res.head(12).to_string(index=False))
    w, t = res.iloc[0]["weights"], float(res.iloc[0]["t"])
    t1 = None if pd.isna(res.iloc[0]["t1"]) else float(res.iloc[0]["t1"])
    print(f"\nBEST weights {dict(zip(tags, w))} t={t} t1={t1}: macro F0.5 {res.iloc[0]['f05']:.4f} | "
          f"precision {res.iloc[0]['precision']:.4f} | recall {res.iloc[0]['recall']:.4f}")

    # ------------------------------------------------------------ test
    tests = [pd.read_parquet(os.path.join(args.work_dir, "pred", f"{tg}_test.parquet"), columns=KEYS + ["prob"])
             for tg in tags]
    tb, TP = aligned(tests)
    del tests
    tb["prob"] = TP @ np.array(w, np.float32)
    sel = decide(tb, t, t1)
    print(f"test: {len(tb):,} pairs, {sel.sum():,} matches")

    s1 = load_ids(args.data_dir, "test", "source1")
    ids = {s: load_ids(args.data_dir, "test", f"source{s}")["entity_id"].to_numpy() for s in (2, 3)}
    src, crow, s1r = tb["src"].to_numpy(), tb["c_row"].to_numpy(), tb["s1_row"].to_numpy()
    cid = np.where(src == 2, ids[2][np.where(src == 2, crow, 0)], ids[3][np.where(src == 3, crow, 0)])
    n_s1 = len(s1)
    cands = id_lists(s1r, cid, n_s1)
    matches = id_lists(s1r[sel], cid[sel], n_s1, order_key=-tb["prob"].to_numpy()[sel])
    n_match = np.array([0 if m == "" else m.count(",") + 1 for m in matches])
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
    with open(os.path.join(hist, "decision.json"), "w") as f:
        json.dump({"tags": tags, "weights": list(w), "t": t, "t1": t1, "val_f05": float(res.iloc[0]["f05"])}, f, indent=2)
    tb.assign(selected=sel.astype(np.int8)).to_parquet(
        os.path.join(args.work_dir, "pred", f"{args.out_tag}_test.parquet"), index=False)
    print(f"wrote {mpath}\n      copy in {hist}")

    root = os.path.dirname(os.path.abspath(args.data_dir))
    r = subprocess.run([sys.executable, os.path.join(root, "utils", "validate_submission.py"), "--matching", mpath,
                        "--candidate", cpath, "--test-dir", os.path.join(args.data_dir, "test")],
                       capture_output=True, text=True)
    print(r.stdout[-300:] + r.stderr[-300:])
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
