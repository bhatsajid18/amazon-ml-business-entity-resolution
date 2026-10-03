"""Score the test candidates, apply the decision rule and write the submission files.

Input : WORK_DIR/feat/<tag>/test.parquet, WORK_DIR/models/lgb_<tag>.txt, decision_<tag>.json
Output: OUT_DIR/matching_results.tsv   final matches (one row per test S1 entity)
        OUT_DIR/candidate_pairs.tsv    every pair the model scored (the pruned blocking set)
        OUT_DIR/history/<tag>_<timestamp>/  a copy of both, for the version history
Then runs utils/validate_submission.py on the two files.

Usage:  python src/predict.py --tag labse0
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from decision import decide
from common import load_ids


def id_lists(s1_rows, ids, n_s1, order_key=None):
    """Comma-joined id list per S1 row (all n_s1 rows, empty string when none)."""
    if order_key is None:
        order = np.argsort(s1_rows, kind="stable")
    else:
        order = np.lexsort((order_key, s1_rows))
    s, v = s1_rows[order], ids[order]
    out = np.full(n_s1, "", dtype=object)
    if len(s):
        starts = np.r_[0, np.flatnonzero(np.diff(s)) + 1]
        ends = np.r_[starts[1:], len(s)]
        for a, b in zip(starts, ends):
            out[s[a]] = ",".join(v[a:b])
    return out


def write_tsv(path, header, first, second):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"source1_entity_id\t{header}\n")
        f.writelines(f"{a}\t{b}\n" for a, b in zip(first, second))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.environ.get("DATA_DIR"))
    ap.add_argument("--work-dir", default=os.environ.get("WORK_DIR"))
    ap.add_argument("--out-dir", default=os.environ.get("OUT_DIR"))
    ap.add_argument("--tag", default="labse0")
    ap.add_argument("--t", type=float, help="override the tuned threshold")
    ap.add_argument("--t1", type=float, help="override the tuned top-1 floor")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    args = ap.parse_args()
    t0 = time.time()

    with open(os.path.join(args.work_dir, "models", f"decision_{args.tag}.json")) as f:
        dec = json.load(f)
    feats = dec["features"]
    t = args.t if args.t is not None else dec["t"]
    t1 = args.t1 if args.t1 is not None else dec["t1"]
    model = lgb.Booster(model_file=os.path.join(args.work_dir, "models", f"lgb_{args.tag}.txt"))

    path = os.path.join(args.work_dir, "feat", args.tag, "test.parquet")
    pf = pq.ParquetFile(path)
    keys, probs = [], []
    for i in range(pf.num_row_groups):
        part = pf.read_row_group(i, columns=["s1_row", "src", "c_row"] + [c for c in feats if c not in ("src",)]).to_pandas()
        probs.append(model.predict(part[feats], num_threads=args.threads).astype(np.float32))
        keys.append(part[["s1_row", "src", "c_row"]])
    df = pd.concat(keys, ignore_index=True)
    df["prob"] = np.concatenate(probs)
    print(f"scored {len(df):,} candidate pairs in {time.time() - t0:.0f}s")

    sel = decide(df, t, t1)
    print(f"decision t={t} t1={t1}: {sel.sum():,} matches")
    pred_path = os.path.join(args.work_dir, "pred", f"{args.tag}_test.parquet")
    os.makedirs(os.path.dirname(pred_path), exist_ok=True)
    df.assign(selected=sel.astype(np.int8)).to_parquet(pred_path, index=False)   # for src/diagnose.py

    s1 = load_ids(args.data_dir, "test", "source1")
    ids = {s: load_ids(args.data_dir, "test", f"source{s}")["entity_id"].to_numpy()
           for s in (2, 3)}
    src, crow = df["src"].to_numpy(), df["c_row"].to_numpy()
    cid = np.where(src == 2, ids[2][np.where(src == 2, crow, 0)], ids[3][np.where(src == 3, crow, 0)])
    s1r = df["s1_row"].to_numpy()
    n_s1 = len(s1)

    cands = id_lists(s1r, cid, n_s1)
    matches = id_lists(s1r[sel], cid[sel], n_s1, order_key=-df["prob"].to_numpy()[sel])
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
    hist = os.path.join(args.out_dir, "history", f"{args.tag}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(hist, exist_ok=True)
    for p in (mpath, cpath):
        shutil.copy2(p, hist)
    with open(os.path.join(hist, "decision.json"), "w") as f:
        json.dump({**dec, "t_used": t, "t1_used": t1}, f, indent=2)
    print(f"wrote {mpath}\n      {cpath}\n      copy in {hist}")

    root = os.path.dirname(os.path.abspath(args.data_dir))
    validator = os.path.join(root, "utils", "validate_submission.py")
    if os.path.exists(validator):
        r = subprocess.run([sys.executable, validator, "--matching", mpath, "--candidate", cpath,
                            "--test-dir", os.path.join(args.data_dir, "test")], capture_output=True, text=True)
        print("\n--- validate_submission.py ---\n" + r.stdout + r.stderr)
        if r.returncode != 0:
            sys.exit("validator reported problems; do NOT upload until fixed")
    else:
        print(f"WARNING: validator not found at {validator}; run it by hand before uploading")
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
