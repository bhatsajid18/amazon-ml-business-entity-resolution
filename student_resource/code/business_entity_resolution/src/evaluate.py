"""Independent local check of the challenge metric on the held-out training fold.

1. Builds a matching_results-style TSV for the validation S1 entities (fold C sample)
   with exactly the same model + decision rule as predict.py uses for the test set:
       OUT_DIR/val_<tag>_matching_results.tsv
2. Scores that file against train_ground_truth.tsv with a stand-alone implementation of
   the formula in the problem statement (written separately from decision.py so the two
   cross-check each other):
     - per S1 entity: P = |pred & true| / |pred|,  R = |pred & true| / |true|
       F0.5 = 1.25 P R / (0.25 P + R)
     - no true matches: 1.0 if the predicted list is empty, else 0.0
     - true matches but empty prediction: 0.0
     - macro average over every S1 entity in the file.

Any matching_results-style file can also be scored directly:
    python src/evaluate.py --score-only path/to/file.tsv

Usage:  python src/evaluate.py --tag labse1
"""
import argparse
import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd

from common import read_tsv
from decision import decide
from common import load_ids


def show_errors(pred_path, gt_path, data_dir, n=12, seed=0, cand_path=None):
    """Print sample wrong merges and missed matches with their names and addresses.
    Missed matches are split into 'not in candidates' (blocking) and 'rejected' (matcher)
    when the validation candidate list (cand_path, same TSV layout) is given."""
    pred = read_tsv(pred_path).set_index("source1_entity_id")["matched_entity_ids"]
    gt = read_tsv(gt_path).set_index("source1_entity_id")["matched_entity_ids"].reindex(pred.index).fillna("")
    cand = read_tsv(cand_path).set_index("source1_entity_id")["candidate_entity_ids"] if cand_path else None
    fp, fn_block, fn_model = [], [], []
    for s1, m in pred.items():
        p = {x for x in m.split(",") if x}
        t = {x for x in gt[s1].split(",") if x}
        c = {x for x in cand.get(s1, "").split(",") if x} if cand is not None else None
        fp += [(s1, x) for x in p - t]
        for x in t - p:
            (fn_block if c is not None and x not in c else fn_model).append((s1, x))
    rng = np.random.default_rng(seed)
    ids = set()
    groups = [("WRONG MERGES (predicted, not true)", fp), ("MISSED, rejected by the matcher", fn_model),
              ("MISSED, not in the candidate set (blocking)", fn_block)]
    picks = []
    for title, lst in groups:
        sel = [lst[i] for i in rng.choice(len(lst), size=min(n, len(lst)), replace=False)] if lst else []
        picks.append((title, len(lst), sel))
        for a, b in sel:
            ids.update([a, b])
    rec = pd.concat([read_tsv(os.path.join(data_dir, "train", f"train_source{k}.tsv")) for k in (1, 2, 3)])
    rec = rec[rec["entity_id"].isin(ids)].set_index("entity_id")
    fmt = lambda i: (f"{i} [{rec.at[i, 'country']}] {rec.at[i, 'business_name'][:55]} | "
                     f"{rec.at[i, 'business_address'][:80]}") if i in rec.index else i
    for title, total, sel in picks:
        print(f"\n--- {title}: {total:,} pairs; {len(sel)} examples")
        for a, b in sel:
            print("  S1 " + fmt(a) + "\n   ~ " + fmt(b))


def score_file(pred_path, gt_path, s1_country=None):
    """Macro F0.5 of a matching TSV against the ground truth, over the S1 ids in the file."""
    pred = read_tsv(pred_path)
    gt = read_tsv(gt_path).set_index("source1_entity_id")["matched_entity_ids"]
    fs, rows = [], []
    for s1, m in zip(pred["source1_entity_id"], pred["matched_entity_ids"]):
        p = {x for x in m.split(",") if x}
        t = {x for x in gt.get(s1, "").split(",") if x}
        if not t:
            f = 1.0 if not p else 0.0
        elif not p:
            f = 0.0
        else:
            tp = len(p & t)
            prec, rec = tp / len(p), tp / len(t)
            f = 0.0 if tp == 0 else 1.25 * prec * rec / (0.25 * prec + rec)
        fs.append(f)
        rows.append((s1, len(p), len(t), len(p & t)))
    df = pd.DataFrame(rows, columns=["s1", "n_pred", "n_true", "tp"])
    df["f05"] = fs
    print(f"\nscored {pred_path}")
    print(f"  S1 entities       : {len(df):,}")
    print(f"  MACRO F0.5        : {df['f05'].mean():.4f}")
    print(f"  precision (micro) : {df['tp'].sum() / max(df['n_pred'].sum(), 1):.4f}")
    print(f"  recall (micro)    : {df['tp'].sum() / max(df['n_true'].sum(), 1):.4f}")
    sing = df["n_true"] == 0
    print(f"  singletons        : {sing.mean():.2%} of entities, F0.5 {df.loc[sing, 'f05'].mean():.4f}")
    print(f"  with matches      : F0.5 {df.loc[~sing, 'f05'].mean():.4f} | "
          f"empty prediction on {(df.loc[~sing, 'n_pred'] == 0).mean():.2%} of them")
    if s1_country is not None:
        df["country"] = df["s1"].map(s1_country)
        print("  by country:\n" + df.groupby("country")["f05"].agg(["size", "mean"]).round(4)
              .to_string().replace("\n", "\n    ").join(["    ", ""]))
    return df["f05"].mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.environ.get("DATA_DIR"))
    ap.add_argument("--work-dir", default=os.environ.get("WORK_DIR"))
    ap.add_argument("--out-dir", default=os.environ.get("OUT_DIR"))
    ap.add_argument("--tag", default="labse1")
    ap.add_argument("--score-only", help="score an existing matching TSV (train S1 ids) and exit")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    ap.add_argument("--show-errors", type=int, default=12, help="examples per error type (0 = none)")
    args = ap.parse_args()
    gt_path = os.path.join(args.data_dir, "train", "train_ground_truth.tsv")
    s1 = load_ids(args.data_dir, "train", "source1")
    country = pd.Series(s1["country"].to_numpy(), index=s1["entity_id"].to_numpy())

    if args.score_only:
        score_file(args.score_only, gt_path, country)
        return

    with open(os.path.join(args.work_dir, "models", f"decision_{args.tag}.json")) as f:
        dec = json.load(f)
    model = lgb.Booster(model_file=os.path.join(args.work_dir, "models", f"lgb_{args.tag}.txt"))
    feat_dir = os.path.join(args.work_dir, "feat", args.tag)
    va = pd.read_parquet(os.path.join(feat_dir, "train.parquet"),
                         columns=["s1_row", "src", "c_row", "is_val"] + [c for c in dec["features"] if c != "src"])
    va = va[va["is_val"] == 1].reset_index(drop=True)
    va["prob"] = model.predict(va[dec["features"]], num_threads=args.threads)
    sel = decide(va, dec["t"], dec["t1"])
    print(f"validation pairs {len(va):,} | predicted matches {sel.sum():,} (t={dec['t']}, t1={dec['t1']})")

    ids = {s: load_ids(args.data_dir, "train", f"source{s}")["entity_id"].to_numpy()
           for s in (2, 3)}
    p = va[sel]
    cid = np.where(p["src"] == 2, ids[2][np.where(p["src"] == 2, p["c_row"], 0)],
                   ids[3][np.where(p["src"] == 3, p["c_row"], 0)])
    lists = pd.Series(cid).groupby(p["s1_row"].to_numpy()).agg(",".join)
    val_rows = np.load(os.path.join(feat_dir, "val_s1_rows.npy"))
    out = os.path.join(args.out_dir, f"val_{args.tag}_matching_results.tsv")
    ent = s1["entity_id"].to_numpy()
    with open(out, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        f.writelines(f"{ent[r]}\t{lists.get(r, '')}\n" for r in val_rows)
    print(f"wrote {out} ({len(val_rows):,} validation S1 entities)")
    score_file(out, gt_path, country)
    print(f"\n(train_match.py reported {dec['val_f05']:.4f} for the same rows; the two should agree)")

    if args.show_errors:
        call = np.where(va["src"] == 2, ids[2][np.where(va["src"] == 2, va["c_row"], 0)],
                        ids[3][np.where(va["src"] == 3, va["c_row"], 0)])
        clists = pd.Series(call).groupby(va["s1_row"].to_numpy()).agg(",".join)
        cpath = os.path.join(args.out_dir, f"val_{args.tag}_candidate_pairs.tsv")
        with open(cpath, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tcandidate_entity_ids\n")
            f.writelines(f"{ent[r]}\t{clists.get(r, '')}\n" for r in val_rows)
        show_errors(out, gt_path, args.data_dir, n=args.show_errors, cand_path=cpath)


if __name__ == "__main__":
    main()
