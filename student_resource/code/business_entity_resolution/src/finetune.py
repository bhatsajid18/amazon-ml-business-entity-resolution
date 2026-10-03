"""Round 2: fine-tune LaBSE on training pairs of fold A, focused on what round 1 missed.

Examples are (S1 text, matched S2/S3 text, hard negative text):
  * one positive per fold-A S1 entity; when round-1 blocking (tag --from-tag, cut-off
    --k-fwd / --k-rev) missed some of its true matches, a missed one is chosen,
  * S1 entities with a missed match are taken first (up to half the budget), the rest
    are sampled at random, so the model sees both hard and typical cases,
  * the hard negative is the highest-ranked wrong round-1 candidate from the same source
    (a look-alike business), falling back to a random record of the same country.
Loss: MultipleNegativesRankingLoss (in-batch negatives + the hard negative), batches
without duplicate texts. Token embeddings stay frozen (fits P100 memory, keeps the
multilingual vocabulary intact for unseen countries such as France). fp16 AMP.

Folds B and C are never used here, so LightGBM (fold B) and validation (fold C) stay
free of pairs the encoder has memorized.

Output: WORK_DIR/models/labse_ft   (load with SentenceTransformer(path))
Usage:  python src/finetune.py --max-examples 350000
"""
import argparse
import glob
import math
import os
import re
import time

import numpy as np
import pandas as pd
import torch

from common import FOLD_A, embed_texts, read_tsv, s1_fold, source_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.environ.get("DATA_DIR"))
    ap.add_argument("--work-dir", default=os.environ.get("WORK_DIR"))
    ap.add_argument("--from-tag", default="labse0", help="round-1 candidates used to find misses / hard negatives")
    ap.add_argument("--base-model", default="sentence-transformers/LaBSE")
    ap.add_argument("--out", default=None)
    ap.add_argument("--k-fwd", type=int, default=10)
    ap.add_argument("--k-rev", type=int, default=3)
    ap.add_argument("--max-examples", type=int, default=350_000)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=48)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--reverse-frac", type=float, default=0.0,
                    help="share of examples trained in the candidate -> S1 direction")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = args.out or os.path.join(args.work_dir, "models", "labse_ft")
    t0 = time.time()
    rng = np.random.default_rng(args.seed)

    # ------------------------------------------------------------ records and texts
    s1 = read_tsv(source_path(args.data_dir, "train", "source1"))
    src = {s: read_tsv(source_path(args.data_dir, "train", f"source{s}")) for s in (2, 3)}
    n_s1 = len(s1)
    fold_a = np.isin(s1_fold(s1["entity_id"].to_numpy()), FOLD_A)
    print(f"fold A: {fold_a.sum():,} of {n_s1:,} S1 entities")

    # ------------------------------------------------------------ true pairs of fold A
    gt = read_tsv(os.path.join(args.data_dir, "train", "train_ground_truth.tsv"))
    ex = gt.assign(m=gt["matched_entity_ids"].str.split(",")).explode("m")
    ex = ex[ex["m"].str.len() > 0]
    s1_row = pd.Series(np.arange(n_s1), index=s1["entity_id"].to_numpy())
    ex["s1_row"] = s1_row[ex["source1_entity_id"]].to_numpy()
    ex = ex[fold_a[ex["s1_row"].to_numpy()]]
    ex["src"] = ex["m"].str[1].astype(int)
    rows = {s: pd.Series(np.arange(len(src[s])), index=src[s]["entity_id"].to_numpy()) for s in (2, 3)}
    ex["c_row"] = np.where(ex["src"] == 2, rows[2].reindex(ex["m"]).fillna(-1).to_numpy(),
                           rows[3].reindex(ex["m"]).fillna(-1).to_numpy()).astype(np.int64)
    ex = ex[ex["c_row"] >= 0].reset_index(drop=True)
    key = lambda s, a, c: (np.asarray(s, np.int64) * n_s1 + np.asarray(a, np.int64)) * 10**7 + np.asarray(c, np.int64)
    true_keys = key(ex["src"], ex["s1_row"], ex["c_row"])
    print(f"{len(ex):,} true pairs in fold A")

    # ------------------------------------------------------------ round-1 candidates
    parts = []
    for f in glob.glob(os.path.join(args.work_dir, "cand", args.from_tag, "train_*_source[23].parquet")):
        c = pd.read_parquet(f)
        c["src"] = np.int8(int(re.search(r"source([23])\.parquet$", f).group(1)))
        parts.append(c)
    cand_all = pd.concat(parts, ignore_index=True)
    rev_all = cand_all.loc[cand_all["rev_rank"] >= 0, ["src", "c_row", "s1_row", "rev_rank"]]
    cand = cand_all[fold_a[cand_all["s1_row"].to_numpy()]].reset_index(drop=True)
    del cand_all
    fr, rr = cand["fwd_rank"].to_numpy(), cand["rev_rank"].to_numpy()
    kept = ((fr >= 0) & (fr < args.k_fwd)) | ((rr >= 0) & (rr < args.k_rev))
    ck = key(cand["src"], cand["s1_row"], cand["c_row"])
    ex["missed"] = ~np.isin(true_keys, ck[kept])
    print(f"recall ceiling of '{args.from_tag}' candidates on fold A: {1 - ex['missed'].mean():.4f}")

    # hard negative: best-ranked wrong forward candidate per (S1, source)
    neg = cand[(fr >= 0) & ~np.isin(ck, true_keys)].sort_values(["s1_row", "src", "fwd_rank"])
    neg = neg.drop_duplicates(["s1_row", "src"]).set_index(["s1_row", "src"])["c_row"]

    # ------------------------------------------------------------ choose examples
    ex = ex.sample(frac=1.0, random_state=0)
    ex = ex.sort_values("missed", ascending=False, kind="stable")           # a missed positive first
    ex = ex.drop_duplicates("s1_row")                                         # one positive per S1
    hard, easy = ex[ex["missed"]], ex[~ex["missed"]]
    n_hard = min(len(hard), args.max_examples // 2)
    pick = pd.concat([hard.iloc[:n_hard], easy.sample(n=min(len(easy), args.max_examples - n_hard),
                                                       random_state=0)]).sample(frac=1.0, random_state=1)
    print(f"examples: {len(pick):,} ({n_hard:,} with a miss of '{args.from_tag}')")

    neg_row = neg.reindex(pd.MultiIndex.from_arrays([pick["s1_row"], pick["src"]])).to_numpy()
    country = {s: src[s]["country"].to_numpy() for s in (2, 3)}
    s1_country = s1["country"].to_numpy()
    for i in np.flatnonzero(pd.isna(neg_row)):                                # fallback: random same-country record
        s = pick["src"].iat[i]
        same = np.flatnonzero(country[s] == s1_country[pick["s1_row"].iat[i]])
        neg_row[i] = rng.choice(same)
    neg_row = neg_row.astype(np.int64)

    # reverse direction (candidate -> S1): the hard negative is the S1 record the previous
    # model ranked highest for this candidate that is not its true parent
    n_rev = int(round(args.reverse_frac * len(pick)))
    is_rev = np.zeros(len(pick), bool)
    is_rev[rng.choice(len(pick), size=n_rev, replace=False)] = True
    q = pd.DataFrame({"i": np.arange(len(pick)), "src": pick["src"].to_numpy(),
                      "c_row": pick["c_row"].to_numpy(), "parent": pick["s1_row"].to_numpy()})[is_rev]
    m = q.merge(rev_all, on=["src", "c_row"])
    m = m[m["s1_row"] != m["parent"]].sort_values(["i", "rev_rank"]).drop_duplicates("i")
    rneg = np.full(len(pick), -1, np.int64)
    rneg[m["i"].to_numpy()] = m["s1_row"].to_numpy()
    for i in np.flatnonzero(is_rev & (rneg < 0)):                            # fallback: random same-country S1
        same = np.flatnonzero(s1_country == s1_country[pick["s1_row"].iat[i]])
        rneg[i] = rng.choice(same)
    del rev_all
    print(f"reverse-direction examples: {n_rev:,} ({(m['i'].size / max(n_rev, 1)):.1%} with a model-ranked hard negative)")

    def texts(df, idx):
        sub = df.iloc[idx]
        return embed_texts(sub["business_name"], sub["business_address"])

    anchor = texts(s1, pick["s1_row"].to_numpy())
    pos, negt = [None] * len(pick), [None] * len(pick)
    for s in (2, 3):
        m = np.flatnonzero(pick["src"].to_numpy() == s)
        for arr, rws in ((pos, pick["c_row"].to_numpy()[m]), (negt, neg_row[m])):
            for j, t in zip(m, texts(src[s], rws)):
                arr[j] = t
    if n_rev:
        r = np.flatnonzero(is_rev)
        s1_neg = texts(s1, rneg[r])
        for j, k in enumerate(r):                  # anchor = candidate record, positive = its S1 record
            anchor[k], pos[k], negt[k] = pos[k], anchor[k], s1_neg[j]
    del s1, src, cand, ex
    print(f"texts ready in {time.time() - t0:.0f}s; e.g.\n  A: {anchor[0]}\n  P: {pos[0]}\n  N: {negt[0]}")

    # ------------------------------------------------------------ train
    from datasets import Dataset
    from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                       SentenceTransformerTrainingArguments, losses)
    from sentence_transformers.training_args import BatchSamplers
    from transformers import TrainerCallback

    class Progress(TrainerCallback):
        """One readable log line per logging step: loss, speed and ETA."""
        def on_log(self, targs, state, control, logs=None, **kw):
            if logs and "loss" in logs and state.global_step:
                el = time.time() - t_train
                eta = el / state.global_step * (state.max_steps - state.global_step)
                print(f"  step {state.global_step:>6,}/{state.max_steps:,}  loss {logs['loss']:.4f}  "
                      f"{el / 60:5.1f} min  eta {eta / 60:5.1f} min", flush=True)

    model = SentenceTransformer(args.base_model, device=args.device)
    model.max_seq_length = args.max_len
    model[0].auto_model.embeddings.requires_grad_(False)
    ds = Dataset.from_dict({"anchor": anchor, "positive": pos, "negative": negt})
    steps = math.ceil(len(ds) / args.batch)
    targs = SentenceTransformerTrainingArguments(
        output_dir=os.path.join(args.work_dir, "models", "labse_ft_tmp"),
        per_device_train_batch_size=args.batch, learning_rate=args.lr, warmup_ratio=0.05,
        num_train_epochs=1, max_steps=steps, fp16=args.device.startswith("cuda"),
        batch_sampler=BatchSamplers.NO_DUPLICATES, logging_steps=50, save_strategy="no",
        report_to="none", dataloader_num_workers=2, seed=0, disable_tqdm=True)
    trainer = SentenceTransformerTrainer(model=model, args=targs, train_dataset=ds,
                                         loss=losses.MultipleNegativesRankingLoss(model),
                                         callbacks=[Progress()])
    print(f"training {steps:,} steps of {args.batch} (lr {args.lr}, max_len {args.max_len})", flush=True)
    t_train = time.time()
    trainer.train()
    os.makedirs(out, exist_ok=True)
    model.save(out, create_model_card=False)
    print(f"saved {out} in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
