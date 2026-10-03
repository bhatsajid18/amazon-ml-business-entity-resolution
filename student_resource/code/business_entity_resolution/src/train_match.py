"""Train the LightGBM matcher and tune the decision rule for macro F0.5.

Input : WORK_DIR/feat/<tag>/train.parquet  (from features.py; rows carry fold / is_val / label)
Output: WORK_DIR/models/lgb_<tag>.txt       LightGBM model
        WORK_DIR/models/decision_<tag>.json features, threshold t, top-1 floor t1, validation report

Training rows: is_val == 0. Validation rows (fold C) are used for early stopping and
for the grid search over (t, t1), scored with the exact challenge metric: macro F0.5
per S1 entity against the FULL ground truth (so matches that blocking missed count
as misses, and singletons count).

Usage:  python src/train_match.py --tag labse0
"""
import argparse
import json
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from common import read_tsv
from decision import decide, macro_f05, one_parent
from common import load_ids

NON_FEATURES = {"s1_row", "c_row", "fold", "is_val", "label"}


def true_counts(data_dir, work_dir):
    """Ground-truth match count per train S1 row (TSV order) and the S1 country labels."""
    s1 = load_ids(data_dir, "train", "source1")
    gt = read_tsv(os.path.join(data_dir, "train", "train_ground_truth.tsv"))
    n = gt["matched_entity_ids"].str.split(",").map(lambda l: sum(1 for x in l if x))
    cnt = pd.Series(n.to_numpy(), index=gt["source1_entity_id"].to_numpy())
    return cnt.reindex(s1["entity_id"]).fillna(0).to_numpy().astype(np.int64), s1["country"].to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.environ.get("DATA_DIR"))
    ap.add_argument("--work-dir", default=os.environ.get("WORK_DIR"))
    ap.add_argument("--tag", default="labse0")
    ap.add_argument("--rounds", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=0.08)
    ap.add_argument("--leaves", type=int, default=255)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    ap.add_argument("--train-countries", default="",
                    help="comma list of S1 country labels to train on (diagnostic, e.g. leave-one-country-out)")
    ap.add_argument("--val-countries", default="", help="comma list of S1 country labels to validate on")
    ap.add_argument("--out-suffix", default="", help="saved as lgb_<tag><suffix>.txt (keeps diagnostic runs apart)")
    args = ap.parse_args()
    t0 = time.time()

    df = pd.read_parquet(os.path.join(args.work_dir, "feat", args.tag, "train.parquet"))
    feats = [c for c in df.columns if c not in NON_FEATURES]
    tc, country = true_counts(args.data_dir, args.work_dir)
    val_s1_rows = np.load(os.path.join(args.work_dir, "feat", args.tag, "val_s1_rows.npy"))
    if args.train_countries or args.val_countries:
        row_country = country[df["s1_row"].to_numpy()]
        keep = np.ones(len(df), bool)
        if args.train_countries:
            keep &= (df["is_val"].to_numpy() == 1) | np.isin(row_country, args.train_countries.split(","))
        if args.val_countries:
            vc = args.val_countries.split(",")
            keep &= (df["is_val"].to_numpy() == 0) | np.isin(row_country, vc)
            val_s1_rows = val_s1_rows[np.isin(country[val_s1_rows], vc)]
        df = df[keep]
        print(f"country filter: train on {args.train_countries or 'all'}, validate on {args.val_countries or 'all'}")
    # memory-lean split: one float32 matrix for training, the frame itself is released
    is_val = df["is_val"].to_numpy() == 1
    va = df[is_val].reset_index(drop=True)
    y_tr = df.loc[~is_val, "label"].to_numpy()
    X_tr = df.loc[~is_val, feats].to_numpy(dtype=np.float32)
    del df
    print(f"{len(y_tr):,} train rows ({y_tr.mean():.2%} positive), {len(va):,} val rows, "
          f"{len(feats)} features, loaded in {time.time() - t0:.0f}s")

    params = dict(objective="binary", metric=["binary_logloss", "auc"], learning_rate=args.lr,
                  num_leaves=args.leaves, min_data_in_leaf=500, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, max_bin=255,
                  num_threads=args.threads, verbose=-1, seed=0)
    # the Dataset keeps the only reference to X_tr and frees it once LightGBM has binned it
    dtr = lgb.Dataset(X_tr, y_tr, feature_name=feats, categorical_feature=["c_script"], free_raw_data=True)
    del X_tr, y_tr
    dva = lgb.Dataset(va[feats].to_numpy(dtype=np.float32), va["label"].to_numpy(), feature_name=feats,
                      reference=dtr)   # bin mappers and categorical columns come from dtr
    model = lgb.train(params, dtr, args.rounds, valid_sets=[dva], valid_names=["val"],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)])
    print(f"trained {model.best_iteration} rounds in {(time.time() - t0) / 60:.1f} min")

    va["prob"] = model.predict(va[feats].to_numpy(dtype=np.float32), num_iteration=model.best_iteration)
    # every S1 record sampled into validation is scored, including ones with no candidates
    s1_eval = np.union1d(va["s1_row"].unique(), val_s1_rows)

    in_cands = np.bincount(va["s1_row"], weights=va["label"], minlength=len(tc))[s1_eval].sum()
    print(f"validation: {len(s1_eval):,} S1 entities | recall ceiling of candidates "
          f"{in_cands / max(tc[s1_eval].sum(), 1):.4f}")

    parent = one_parent(va)
    lab = va["label"].to_numpy()
    s1r = va["s1_row"].to_numpy()
    best = None
    rows = []
    for t in np.round(np.arange(0.20, 0.96, 0.05), 2):
        for t1 in [None, 0.05, 0.10, 0.15, 0.20, 0.30]:
            if t1 is not None and t1 >= t:
                continue
            sel = decide(va, t, t1, parent)
            m = macro_f05(s1_eval, tc, s1r[sel], lab[sel])
            rows.append((t, t1, m["f05"], m["precision_micro"], m["recall_micro"], m["empty_pred_frac"]))
            if best is None or m["f05"] > best[2]:
                best = rows[-1]
    grid = pd.DataFrame(rows, columns=["t", "t1", "f05", "precision", "recall", "empty_frac"])
    print("\ntop decision settings on validation:")
    print(grid.sort_values("f05", ascending=False).head(10).to_string(index=False))
    t, t1 = best[0], best[1]

    sel = decide(va, t, t1, parent)
    m = macro_f05(s1_eval, tc, s1r[sel], lab[sel])
    print(f"\nBEST t={t} t1={t1}: macro F0.5 {m['f05']:.4f} | precision {m['precision_micro']:.4f} | "
          f"recall {m['recall_micro']:.4f} | empty lists {m['empty_pred_frac']:.2%} "
          f"(singletons {m['singleton_frac']:.2%})")
    per = pd.DataFrame({"country": country[s1_eval], "f05": m["per_entity"], "true": tc[s1_eval]})
    per["bucket"] = np.where(per["true"] == 0, "singleton", "has matches")
    print(per.groupby(["country", "bucket"])["f05"].agg(["size", "mean"]).round(4).to_string())

    imp = pd.Series(model.feature_importance("gain"), index=feats).sort_values(ascending=False)
    print("\nfeature importance (gain, top 20):")
    print((imp / imp.sum()).head(20).round(4).to_string())

    out_dir = os.path.join(args.work_dir, "models")
    os.makedirs(out_dir, exist_ok=True)
    name = f"{args.tag}{args.out_suffix}"
    model.save_model(os.path.join(out_dir, f"lgb_{name}.txt"), num_iteration=model.best_iteration)
    with open(os.path.join(out_dir, f"decision_{name}.json"), "w") as f:
        json.dump({"features": feats, "t": float(t), "t1": None if t1 is None else float(t1),
                   "val_f05": m["f05"], "val_precision": m["precision_micro"], "val_recall": m["recall_micro"],
                   "best_iteration": model.best_iteration}, f, indent=2)
    print(f"saved model + decision to {out_dir} in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
