"""Candidate generation (blocking): exact bidirectional top-k search on the GPU.

One task = one (split, country, candidate source). Inside a country shard:
  * forward : every S1 record keeps its top --k-fwd records from the source,
  * reverse : every source record keeps its top --k-rev S1 records,
both taken from the same fp16 score blocks in a single pass (no approximate index).
The union of the two is written to
    WORK_DIR/cand/<tag>/<split>_<country>_<source>.parquet
with columns  s1_row, c_row (row numbers in the TSV files), cos,
              fwd_rank, rev_rank (0-based, -1 = not in that list).
Large k values are stored once; the final cut-offs are tuned later without re-running.

For the training split the task also prints the recall ceiling for several
(k_fwd, k_rev) cut-offs, measured against the ground truth.

Usage:
    python src/block.py --count
    python src/block.py --task 3 --tag labse0
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from common import CAND_SOURCES, SPLITS, cand_path, load_embeddings, read_tsv, s1_fold, source_path


def block_tasks(data_dir):
    """Ordered (split, country, source) list; countries are whatever S1 labels exist."""
    tasks = []
    for split in SPLITS:
        countries = sorted(read_tsv(source_path(data_dir, split, "source1"), usecols=["country"])["country"].unique())
        for country in countries:
            for source in CAND_SOURCES:
                tasks.append((split, country, source))
    return tasks


@torch.inference_mode()
def bidirectional_topk(Q, D, k_fwd, k_rev, q_chunk, d_chunk):
    """Exact top-k in both directions between row-normalised Q (nq,d) and D (nd,d) on GPU.
    Returns forward (vals, idx) of shape (nq, k_fwd) and reverse (vals, idx) of (k_rev, nd)."""
    nq, nd = Q.shape[0], D.shape[0]
    k_fwd, k_rev = min(k_fwd, nd), min(k_rev, nq)
    dev = Q.device
    rev_v = torch.full((k_rev, nd), -float("inf"), dtype=Q.dtype, device=dev)
    rev_i = torch.zeros((k_rev, nd), dtype=torch.int64, device=dev)
    fwd_v = torch.empty((nq, k_fwd), dtype=Q.dtype, device=dev)
    fwd_i = torch.empty((nq, k_fwd), dtype=torch.int64, device=dev)
    t0 = time.time()
    for qs in range(0, nq, q_chunk):
        q = Q[qs:qs + q_chunk]
        bv = bi = None
        for ds in range(0, nd, d_chunk):
            s = q @ D[ds:ds + d_chunk].T                           # (qc, dc) cosine scores
            # forward: best candidates for each query, merged with the running best
            v, i = s.topk(min(k_fwd, s.shape[1]), dim=1)
            i += ds
            if bv is None:
                bv, bi = v, i
            else:
                v, pos = torch.cat([bv, v], 1).topk(k_fwd, dim=1)
                bi = torch.cat([bi, i], 1).gather(1, pos)
                bv = v
            # reverse: best queries for each candidate, merged with the running best
            kr = min(k_rev, s.shape[0])
            v, i = s.topk(kr, dim=0)
            i += qs
            v, pos = torch.cat([rev_v[:, ds:ds + d_chunk], v], 0).topk(k_rev, dim=0)
            rev_i[:, ds:ds + d_chunk] = torch.cat([rev_i[:, ds:ds + d_chunk], i], 0).gather(0, pos)
            rev_v[:, ds:ds + d_chunk] = v
        fwd_v[qs:qs + q_chunk], fwd_i[qs:qs + q_chunk] = bv, bi
        done = min(qs + q_chunk, nq)
        el = time.time() - t0
        if (qs // q_chunk) % 20 == 0 or done == nq:
            print(f"  queries {done:>9,}/{nq:,}  {el / 60:5.1f} min  eta {el / done * (nq - done) / 60:5.1f} min",
                  flush=True)
    return fwd_v, fwd_i, rev_v, rev_i


def union_pairs(fwd_v, fwd_i, rev_v, rev_i, q_glob, d_glob, n_src):
    """Union of forward and reverse pairs with their ranks, in global row numbers."""
    nq, kf = fwd_i.shape
    kr, nd = rev_i.shape
    f_q = np.repeat(q_glob, kf)
    f_d = d_glob[fwd_i.cpu().numpy().ravel()]
    f_v = fwd_v.float().cpu().numpy().ravel()
    f_r = np.tile(np.arange(kf, dtype=np.int8), nq)
    r_q = q_glob[rev_i.cpu().numpy().T.ravel()]
    r_d = np.repeat(d_glob, kr)
    r_v = rev_v.float().cpu().numpy().T.ravel()
    r_r = np.tile(np.arange(kr, dtype=np.int8), nd)
    ok = np.isfinite(r_v)                                          # guard for shards with < k_rev S1s
    r_q, r_d, r_v, r_r = r_q[ok], r_d[ok], r_v[ok], r_r[ok]

    fk = f_q.astype(np.int64) * n_src + f_d
    rk = r_q.astype(np.int64) * n_src + r_d
    keys = np.union1d(fk, rk)
    fwd_rank = np.full(len(keys), -1, np.int8)
    rev_rank = np.full(len(keys), -1, np.int8)
    cos = np.zeros(len(keys), np.float32)
    fpos, rpos = np.searchsorted(keys, fk), np.searchsorted(keys, rk)
    fwd_rank[fpos], cos[fpos] = f_r, f_v
    rev_rank[rpos], cos[rpos] = r_r, r_v
    return pd.DataFrame({"s1_row": (keys // n_src).astype(np.int32), "c_row": (keys % n_src).astype(np.int32),
                         "cos": cos, "fwd_rank": fwd_rank, "rev_rank": rev_rank})


def recall_report(pairs, data_dir, source, s1_ids, src_ids, country_s1_rows):
    """Recall ceiling of several (k_fwd, k_rev) cut-offs against the training ground truth."""
    gt = read_tsv(os.path.join(data_dir, "train", "train_ground_truth.tsv"))
    gt = gt[gt["source1_entity_id"].isin(set(s1_ids[country_s1_rows]))]
    prefix = "S2-" if source == "source2" else "S3-"
    ex = gt.assign(m=gt["matched_entity_ids"].str.split(",")).explode("m")
    ex = ex[ex["m"].str.startswith(prefix, na=False)]
    s1_row = pd.Series(np.arange(len(s1_ids)), index=s1_ids)
    c_row = pd.Series(np.arange(len(src_ids)), index=src_ids)
    n_src = len(src_ids)
    true = np.unique(s1_row[ex["source1_entity_id"]].to_numpy().astype(np.int64) * n_src
                     + c_row[ex["m"]].to_numpy())
    key = pairs["s1_row"].to_numpy().astype(np.int64) * n_src + pairs["c_row"].to_numpy()
    hit = np.isin(key, true)
    fr, rr = pairs["fwd_rank"].to_numpy(), pairs["rev_rank"].to_numpy()
    n_s1 = len(country_s1_rows)
    print(f"\nrecall ceiling vs ground truth ({len(true):,} true pairs, {n_s1:,} S1 records)")
    print("  k_fwd k_rev   recall   cands/S1")
    for kf in [3, 5, 8, 10, 15]:
        for kr in [0, 1, 3, 5]:
            sel = ((fr >= 0) & (fr < kf)) | ((rr >= 0) & (rr < kr))
            print(f"  {kf:>5} {kr:>5}  {hit[sel].sum() / max(len(true), 1):7.4f}  {sel.sum() / n_s1:9.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.environ.get("DATA_DIR"))
    ap.add_argument("--work-dir", default=os.environ.get("WORK_DIR"))
    ap.add_argument("--tag", default="labse0")
    ap.add_argument("--task", type=int)
    ap.add_argument("--count", action="store_true")
    ap.add_argument("--k-fwd", type=int, default=15)
    ap.add_argument("--k-rev", type=int, default=5)
    ap.add_argument("--q-chunk", type=int, default=8192)
    ap.add_argument("--d-chunk", type=int, default=131072)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-tag", default=None, help="write candidates under this tag (default: --tag)")
    ap.add_argument("--exclude-s1-buckets", default="",
                    help='train only, e.g. "0-19": S1 entities in these id-hash buckets are left out of the '
                         "index, so their S2/S3 records become unmatched distractors. Test has ~40%% unmatched "
                         "S2/S3 records vs 26%% in train; leaving out 20%% of train S1 matches that density.")
    args = ap.parse_args()

    tasks = block_tasks(args.data_dir)
    if args.count:
        print(len(tasks))
        for i, t in enumerate(tasks):
            print(i, *t)
        return
    if args.task is None or not 0 <= args.task < len(tasks):
        sys.exit(f"--task must be in [0, {len(tasks) - 1}] (got {args.task})")
    split, country, source = tasks[args.task]
    out = cand_path(args.work_dir, args.out_tag or args.tag, split, country, source)
    print(f"task {args.task}/{len(tasks) - 1}: {split} / {country} / {source} -> {out}")
    if os.path.exists(out):
        print("already done, skipping")
        return
    os.makedirs(os.path.dirname(out), exist_ok=True)

    t0 = time.time()
    s1 = read_tsv(source_path(args.data_dir, split, "source1"), usecols=["entity_id", "country"])
    src = read_tsv(source_path(args.data_dir, split, source), usecols=["entity_id", "country"])
    q_glob = np.flatnonzero(s1["country"].to_numpy() == country)
    if split == "train" and args.exclude_s1_buckets:
        lo, hi = map(int, args.exclude_s1_buckets.split("-"))
        bucket = s1_fold(s1["entity_id"].to_numpy())
        keep = (bucket < lo) | (bucket > hi)
        print(f"excluding S1 buckets {lo}-{hi}: {(~keep[q_glob]).sum():,} of {len(q_glob):,} S1 records left out")
        q_glob = q_glob[keep[q_glob]]
    d_glob = np.flatnonzero(src["country"].to_numpy() == country)
    print(f"{len(q_glob):,} S1 records x {len(d_glob):,} {source} records in {country}")

    Qn, n1 = load_embeddings(args.work_dir, args.tag, split, "source1", q_glob)
    Dn, n2 = load_embeddings(args.work_dir, args.tag, split, source, d_glob)
    assert n1 == len(s1) and n2 == len(src), "embedding row count does not match the TSV"
    print(f"loaded embeddings in {time.time() - t0:.0f}s")

    dev = torch.device(args.device)
    Q = torch.from_numpy(Qn).to(dev)
    D = torch.from_numpy(Dn).to(dev)
    if dev.type == "cpu":                                   # CPU fallback for testing only
        Q, D = Q.float(), D.float()
    del Qn, Dn
    fwd_v, fwd_i, rev_v, rev_i = bidirectional_topk(Q, D, args.k_fwd, args.k_rev, args.q_chunk, args.d_chunk)
    pairs = union_pairs(fwd_v, fwd_i, rev_v, rev_i, q_glob, d_glob, len(src))
    print(f"{len(pairs):,} candidate pairs ({len(pairs) / max(len(q_glob), 1):.1f} per S1) "
          f"in {(time.time() - t0) / 60:.1f} min")

    tmp = out + ".tmp"
    pq.write_table(pa.Table.from_pandas(pairs, preserve_index=False), tmp)
    os.replace(tmp, out)

    if split == "train":
        recall_report(pairs, args.data_dir, source, s1["entity_id"].to_numpy(),
                      src["entity_id"].to_numpy(), q_glob)
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
