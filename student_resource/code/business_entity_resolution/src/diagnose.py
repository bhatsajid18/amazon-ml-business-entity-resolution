"""Test-set diagnostics per country label, and the "blank one country" probe file.

stats : per-country picture of the test predictions saved by predict.py
        (WORK_DIR/pred/<tag>_test.parquet):
          - S1 records, mean candidates and mean matches per S1 record, empty-list share
          - share of the country's S2 / S3 records assigned to some S1 record
          - share of candidate pairs with probability in [0.3, 0.7] (model unsure) and share of
            S1 records whose best candidate falls in that band
        --show <label> prints --n random S1 records of that country with their top candidates,
        probabilities and decisions, for eyeballing.

blank : copies a finished run's matching_results.tsv with every S1 record of one country
        label emptied (candidate_pairs.tsv unchanged, so the subset rule still holds).
        Uploading it next to the unmodified file measures that country's score on the
        public leaderboard:
            F_country ~= singleton_share + (LB_full - LB_blank) / country_share
        (an emptied record scores 1.0 only if it is truly a singleton). The shares are printed.

Country labels are taken from the command line / data, never hard-coded.

Usage:
  python src/diagnose.py stats --tag labse2 --show France --n 30
  python src/diagnose.py blank --run labse2_20260927_051906 --country France
"""
import argparse
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

from common import load_ids, read_tsv, source_path


def stats(args):
    pred = pd.read_parquet(os.path.join(args.work_dir, "pred", f"{args.tag}_test.parquet"))
    s1 = load_ids(args.data_dir, "test", "source1")
    src = {s: load_ids(args.data_dir, "test", f"source{s}") for s in (2, 3)}
    c1 = s1["country"].to_numpy()
    pred["country"] = c1[pred["s1_row"].to_numpy()]
    band = (pred["prob"] >= 0.3) & (pred["prob"] <= 0.7)
    best = pred.groupby("s1_row")["prob"].max()
    n_match = np.bincount(pred.loc[pred["selected"] == 1, "s1_row"], minlength=len(s1))
    n_cand = np.bincount(pred["s1_row"], minlength=len(s1))

    rows = []
    for c in sorted(pd.unique(c1)):
        m1 = c1 == c
        sub = pred[pred["country"] == c]
        sel = sub[sub["selected"] == 1]
        row = {"country": c, "S1": int(m1.sum()), "cand/S1": n_cand[m1].mean(), "match/S1": n_match[m1].mean(),
               "empty": (n_match[m1] == 0).mean(), "pairs in [.3,.7]": band[pred["country"] == c].mean()}
        b = best.reindex(np.flatnonzero(m1))
        row["S1 best in [.3,.7]"] = ((b >= 0.3) & (b <= 0.7)).mean()
        for s in (2, 3):
            n_rec = int((src[s]["country"].to_numpy() == c).sum())
            row[f"S{s} assigned"] = sel.loc[sel["src"] == s, "c_row"].nunique() / max(n_rec, 1)
        row["sel prob<0.85"] = (sel["prob"] < 0.85).mean()
        rows.append(row)
    table = pd.DataFrame(rows).set_index("country")
    pd.set_option("display.width", 200)
    print(table.round(4).to_string())

    if args.show:
        rec = {1: read_tsv(source_path(args.data_dir, "test", "source1"))}
        for s in (2, 3):
            rec[s] = read_tsv(source_path(args.data_dir, "test", f"source{s}"))
        fmt = lambda r: f"{r['entity_id']}  {r['business_name'][:50]} | {r['business_address'][:75]}"
        rows1 = np.flatnonzero(c1 == args.show)
        rng = np.random.default_rng(args.seed)
        print(f"\n--- {args.n} random {args.show} S1 records with their top candidates "
              f"(* = predicted match)")
        g = pred.set_index("s1_row").sort_index()
        for r in rng.choice(rows1, size=min(args.n, len(rows1)), replace=False):
            print("\nS1 " + fmt(rec[1].iloc[r]))
            if r not in g.index:
                print("   (no candidates)")
                continue
            cands = g.loc[[r]].sort_values("prob", ascending=False).head(args.top)
            for _, cr in cands.iterrows():
                mark = "*" if cr["selected"] == 1 else " "
                print(f" {mark} {cr['prob']:.3f}  " + fmt(rec[int(cr['src'])].iloc[int(cr['c_row'])]))


def blank(args):
    root = os.path.dirname(os.path.abspath(args.data_dir))
    run = os.path.join(root, "output", "history", args.run)
    s1 = load_ids(args.data_dir, "test", "source1")
    country = pd.Series(s1["country"].to_numpy(), index=s1["entity_id"].to_numpy())
    m = read_tsv(os.path.join(run, "matching_results.tsv"))
    is_c = m["source1_entity_id"].map(country).to_numpy() == args.country
    share = is_c.mean()
    empty_share = (m.loc[is_c, "matched_entity_ids"] == "").mean()
    m.loc[is_c, "matched_entity_ids"] = ""
    safe = "".join(ch if ch.isalnum() else "_" for ch in args.country)
    out_dir = os.path.join(root, "output", "diag", f"{args.run}_blank_{safe}")
    os.makedirs(out_dir, exist_ok=True)
    mpath, cpath = os.path.join(out_dir, "matching_results.tsv"), os.path.join(out_dir, "candidate_pairs.tsv")
    with open(mpath, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        f.writelines(f"{a}\t{b}\n" for a, b in zip(m["source1_entity_id"], m["matched_entity_ids"]))
    shutil.copy2(os.path.join(run, "candidate_pairs.tsv"), cpath)
    print(f"wrote {mpath}: {is_c.sum():,} {args.country} rows emptied")
    r = subprocess.run([sys.executable, os.path.join(root, "utils", "validate_submission.py"), "--matching", mpath,
                        "--candidate", cpath, "--test-dir", os.path.join(args.data_dir, "test")],
                       capture_output=True, text=True)
    print(r.stdout[-400:] + r.stderr[-400:])
    print(f"{args.country}: {share:.4f} of test S1 records; predicted-empty share in this run {empty_share:.4f} "
          f"(use as the singleton share)")
    print(f"after both scores are known:  F_{safe} ~= {empty_share:.4f} + (LB_full - LB_blank) / {share:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["stats", "blank"])
    ap.add_argument("--data-dir", default=os.environ.get("DATA_DIR"))
    ap.add_argument("--work-dir", default=os.environ.get("WORK_DIR"))
    ap.add_argument("--tag", default="labse2")
    ap.add_argument("--show", default="", help="country label whose records to print")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run", help="blank: folder name under output/history")
    ap.add_argument("--country", help="blank: country label to empty")
    args = ap.parse_args()
    stats(args) if args.cmd == "stats" else blank(args)


if __name__ == "__main__":
    main()
