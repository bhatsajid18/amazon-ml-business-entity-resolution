"""Second candidate channel: exact-key blocking, merged with the embedding candidates.

The embedding search misses a small share of true matches, mostly records whose name is
badly garbled, whose address is empty, or whose text is in another script. This channel
pairs an S1 record with S2/S3 records of the same country label that share a *rare key*:
  n:<word>          a word of the normalised core name
  p:<postal>|<no>   postal code + first (house) number of the address
  h:<no>|<word>     first number + one of the first alphabetic address words
A key is used only if it is rare on both sides within the shard (<= --max-s1 S1 records and
<= --max-c S2/S3 records), so blocks stay small. For every pair the number of shared keys is
stored as key_hits; the embedding pairs keep their cos / ranks, key-only pairs get cos=NaN and
ranks -1. At most --max-new key-only pairs per S1 record are added (most shared keys first).

One task = one (split, country, source) shard, same order as block.py. Train tasks respect
--exclude-s1-buckets exactly like block.py and print the recall ceiling before/after.

Input : WORK_DIR/cand/<tag>/...   (embedding candidates)      WORK_DIR/norm/v2 (normalize.py)
Output: WORK_DIR/cand/<out-tag>/<split>_<country>_<source>.parquet
Usage:  python src/key_block.py --task 3 --tag labse2 --out-tag labse2k --exclude-s1-buckets 0-19
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from block import block_tasks
from common import cand_path, read_tsv, s1_fold
from normalize import load_normalized


def record_keys(core, addr, num1, postal, rows):
    """(row, key) pairs for the given records; keys deduplicated per record."""
    out_r, out_k = [], []
    for r, c, a, n1, pc in zip(rows, core, addr, num1, postal):
        keys = {"n:" + t for t in c.split() if len(t) >= 3 and not t.isdigit()}
        if pc and n1 and pc != n1:
            keys.add(f"p:{pc}|{n1}")
        elif pc:
            keys.add("p:" + pc)
        if n1:
            words = [t for t in a.split() if len(t) >= 4 and t.isalpha()][:6]
            keys.update(f"h:{n1}|{t}" for t in words)
        out_r.extend([r] * len(keys))
        out_k.extend(keys)
    return pd.DataFrame({"row": np.array(out_r, np.int64), "key": out_k})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.environ.get("DATA_DIR"))
    ap.add_argument("--work-dir", default=os.environ.get("WORK_DIR"))
    ap.add_argument("--tag", default="labse2", help="embedding candidates to extend")
    ap.add_argument("--out-tag", default="labse2k")
    ap.add_argument("--task", type=int, required=True)
    ap.add_argument("--exclude-s1-buckets", default="")
    ap.add_argument("--max-s1", type=int, default=30, help="key used only if <= this many S1 records carry it")
    ap.add_argument("--max-c", type=int, default=300, help="... and <= this many S2/S3 records")
    ap.add_argument("--max-new", type=int, default=10, help="key-only pairs added per S1 record")
    args = ap.parse_args()
    t0 = time.time()

    tasks = block_tasks(args.data_dir)
    if not 0 <= args.task < len(tasks):
        sys.exit(f"--task must be in [0, {len(tasks) - 1}]")
    split, country, source = tasks[args.task]
    out = cand_path(args.work_dir, args.out_tag, split, country, source)
    print(f"task {args.task}: {split} / {country} / {source} -> {out}")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    n1 = load_normalized(args.data_dir, args.work_dir, split, "source1")
    nc = load_normalized(args.data_dir, args.work_dir, split, source)
    q = np.flatnonzero(n1["country"].to_numpy() == country)
    if split == "train" and args.exclude_s1_buckets:
        lo, hi = map(int, args.exclude_s1_buckets.split("-"))
        b = s1_fold(n1["entity_id"].to_numpy()[q])
        q = q[(b < lo) | (b > hi)]
    d = np.flatnonzero(nc["country"].to_numpy() == country)
    print(f"{len(q):,} S1 x {len(d):,} {source} records")

    k1 = record_keys(n1["core"].to_numpy()[q], n1["addr"].to_numpy()[q], n1["num1"].to_numpy()[q],
                     n1["postal"].to_numpy()[q], q)
    k2 = record_keys(nc["core"].to_numpy()[d], nc["addr"].to_numpy()[d], nc["num1"].to_numpy()[d],
                     nc["postal"].to_numpy()[d], d)
    v1, v2 = k1["key"].value_counts(), k2["key"].value_counts()
    good = v1.index[v1 <= args.max_s1].intersection(v2.index[v2 <= args.max_c])
    k1, k2 = k1[k1["key"].isin(good)], k2[k2["key"].isin(good)]
    pairs = k1.merge(k2, on="key", suffixes=("_s1", "_c"))
    kp = pairs.groupby(["row_s1", "row_c"]).size().rename("key_hits").reset_index()
    kp.columns = ["s1_row", "c_row", "key_hits"]
    del pairs, k1, k2
    print(f"{len(good):,} usable keys -> {len(kp):,} key pairs in {time.time() - t0:.0f}s")

    emb = pd.read_parquet(cand_path(args.work_dir, args.tag, split, country, source))
    both = emb.merge(kp, on=["s1_row", "c_row"], how="left")
    both["key_hits"] = both["key_hits"].fillna(0)
    new = kp.merge(emb[["s1_row", "c_row"]], on=["s1_row", "c_row"], how="left", indicator=True)
    new = new[new["_merge"] == "left_only"].drop(columns="_merge")
    new = new.sort_values(["s1_row", "key_hits", "c_row"], ascending=[True, False, True])
    new = new.groupby("s1_row").head(args.max_new)
    new = new.assign(cos=np.float32(np.nan), fwd_rank=np.int8(-1), rev_rank=np.int8(-1))
    allp = pd.concat([both, new[both.columns]], ignore_index=True)
    allp["key_hits"] = allp["key_hits"].clip(upper=127).astype(np.int8)
    for col, dt in (("s1_row", np.int32), ("c_row", np.int32), ("fwd_rank", np.int8), ("rev_rank", np.int8),
                    ("cos", np.float32)):
        allp[col] = allp[col].astype(dt)
    print(f"embedding pairs {len(emb):,} | key-only pairs added {len(new):,} "
          f"({len(new) / max(len(q), 1):.2f} per S1)")
    pq.write_table(pa.Table.from_pandas(allp, preserve_index=False), out + ".tmp")
    os.replace(out + ".tmp", out)

    if split == "train":
        gt = read_tsv(os.path.join(args.data_dir, "train", "train_ground_truth.tsv"))
        gt = gt[gt["source1_entity_id"].isin(set(n1["entity_id"].to_numpy()[q]))]
        pre = "S2-" if source == "source2" else "S3-"
        ex = gt.assign(m=gt["matched_entity_ids"].str.split(",")).explode("m")
        ex = ex[ex["m"].str.startswith(pre, na=False)]
        r1 = pd.Series(np.arange(len(n1)), index=n1["entity_id"].to_numpy())
        rc = pd.Series(np.arange(len(nc)), index=nc["entity_id"].to_numpy())
        true = np.unique(r1[ex["source1_entity_id"]].to_numpy() * len(nc) + rc[ex["m"]].to_numpy())
        key = lambda f: f["s1_row"].to_numpy().astype(np.int64) * len(nc) + f["c_row"].to_numpy()
        print(f"recall ceiling ({len(true):,} true pairs): embedding {np.isin(true, key(emb)).mean():.4f} "
              f"-> with key channel {np.isin(true, key(allp)).mean():.4f} | "
              f"cands/S1 {len(emb) / len(q):.1f} -> {len(allp) / len(q):.1f}")
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
