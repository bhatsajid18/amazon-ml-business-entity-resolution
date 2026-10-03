"""Re-apply the decision rule to saved test probabilities with a different threshold, globally or
for chosen country labels (labels come from the command line; nothing is hard-coded). No
re-scoring: reads WORK_DIR/pred/<tag>_test.parquet; the candidate list is copied unchanged from
the run's history folder, so matches stay a subset of the candidates.

Usage:  python src/redecide.py --tag labse2k --run labse2k_20260927_191555 --t 0.75 \
            --t-country France=0.85 --out-tag labse2k_fr85
"""
import argparse
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from common import load_ids
from decision import one_parent
from predict import id_lists, write_tsv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.environ.get("DATA_DIR"))
    ap.add_argument("--work-dir", default=os.environ.get("WORK_DIR"))
    ap.add_argument("--out-dir", default=os.environ.get("OUT_DIR"))
    ap.add_argument("--tag", required=True, help="pred/<tag>_test.parquet")
    ap.add_argument("--run", required=True, help="history folder whose candidate_pairs.tsv is reused")
    ap.add_argument("--t", type=float, required=True, help="threshold for every other label")
    ap.add_argument("--t-country", action="append", default=[], help="LABEL=THRESHOLD, repeatable")
    ap.add_argument("--out-tag", required=True)
    args = ap.parse_args()
    t0 = time.time()

    df = pd.read_parquet(os.path.join(args.work_dir, "pred", f"{args.tag}_test.parquet"),
                         columns=["s1_row", "src", "c_row", "prob"])
    s1 = load_ids(args.data_dir, "test", "source1")
    country = s1["country"].to_numpy()
    s1r = df["s1_row"].to_numpy()
    thr = np.full(len(s1), args.t, np.float32)
    for item in args.t_country:
        label, value = item.rsplit("=", 1)
        m = country == label
        if not m.any():
            sys.exit(f"no test S1 record has country label {label!r}")
        thr[m] = float(value)
        print(f"{label}: threshold {float(value)} for {m.sum():,} S1 records")
    prob = df["prob"].to_numpy()
    sel = one_parent(df) & (prob >= thr[s1r])
    print(f"{len(df):,} pairs -> {sel.sum():,} matches ({time.time() - t0:.0f}s)")

    ids = {s: load_ids(args.data_dir, "test", f"source{s}")["entity_id"].to_numpy() for s in (2, 3)}
    src, crow = df["src"].to_numpy()[sel], df["c_row"].to_numpy()[sel]
    cid = np.where(src == 2, ids[2][np.where(src == 2, crow, 0)], ids[3][np.where(src == 3, crow, 0)])
    matches = id_lists(s1r[sel], cid, len(s1), order_key=-prob[sel])
    n_match = np.array([0 if m == "" else m.count(",") + 1 for m in matches])
    for c in sorted(pd.unique(country)):
        m = country == c
        print(f"  {c:>8s}: {m.sum():>9,} S1 | empty {np.mean(n_match[m] == 0):6.2%} | mean {n_match[m].mean():.2f}")

    root = os.path.dirname(os.path.abspath(args.data_dir))
    hist = os.path.join(args.out_dir, "history", f"{args.out_tag}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(hist, exist_ok=True)
    mpath, cpath = os.path.join(hist, "matching_results.tsv"), os.path.join(hist, "candidate_pairs.tsv")
    write_tsv(mpath, "matched_entity_ids", s1["entity_id"], matches)
    shutil.copy2(os.path.join(args.out_dir, "history", args.run, "candidate_pairs.tsv"), cpath)
    src_dec = os.path.join(args.out_dir, "history", args.run, "decision.json")
    if os.path.exists(src_dec):
        shutil.copy2(src_dec, os.path.join(hist, "decision_base.json"))
    with open(os.path.join(hist, "redecide.txt"), "w") as f:
        f.write(" ".join(sys.argv) + "\n")
    print(f"wrote {hist}")
    r = subprocess.run([sys.executable, os.path.join(root, "utils", "validate_submission.py"), "--matching", mpath,
                        "--candidate", cpath, "--test-dir", os.path.join(args.data_dir, "test")],
                       capture_output=True, text=True)
    print(r.stdout[-300:] + r.stderr[-300:])
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
