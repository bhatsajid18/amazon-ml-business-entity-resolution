"""Encode records with a sentence-embedding model (LaBSE by default), one chunk per task.

Each Slurm array task encodes one chunk of rows of one source file and saves
    WORK_DIR/emb/<tag>/<split>_<source>_<chunk>.npy      (float16, L2-normalised)
in the same row order as the TSV. Tasks are listed in common.encode_tasks(), so
array index i always means the same chunk. Finished chunks are skipped, so a
failed array can simply be resubmitted.

Usage:
    python src/encode.py --count                       # number of tasks
    python src/encode.py --task 7 --tag labse0          # encode chunk 7
    python src/encode.py --task 7 --tag labse1 --model $WORK_DIR/models/labse_ft
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

from common import emb_chunk_path, embed_texts, encode_tasks, read_tsv, source_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.environ.get("DATA_DIR"))
    ap.add_argument("--work-dir", default=os.environ.get("WORK_DIR"))
    ap.add_argument("--model", default="sentence-transformers/LaBSE")
    ap.add_argument("--tag", default="labse0", help="output sub-folder under WORK_DIR/emb")
    ap.add_argument("--task", type=int)
    ap.add_argument("--count", action="store_true", help="print the number of tasks and exit")
    ap.add_argument("--chunk-size", type=int, default=1_000_000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--device", default="cuda", help="cpu only for testing")
    args = ap.parse_args()

    tasks = encode_tasks(args.data_dir, args.chunk_size)
    if args.count:
        print(len(tasks))
        return
    if args.task is None or not 0 <= args.task < len(tasks):
        sys.exit(f"--task must be in [0, {len(tasks) - 1}] (got {args.task})")

    split, source, chunk, start, stop = tasks[args.task]
    out = emb_chunk_path(args.work_dir, args.tag, split, source, chunk)
    print(f"task {args.task}/{len(tasks) - 1}: {split} {source} chunk {chunk} rows [{start}, {stop}) -> {out}")
    if os.path.exists(out):
        print("already done, skipping")
        return
    os.makedirs(os.path.dirname(out), exist_ok=True)

    t0 = time.time()
    df = read_tsv(source_path(args.data_dir, split, source),
                  usecols=["business_name", "business_address"]).iloc[start:stop]
    texts = embed_texts(df["business_name"], df["business_address"])
    del df
    print(f"read + cleaned {len(texts):,} texts in {time.time() - t0:.0f}s; e.g. {texts[:2]}")

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model, device=args.device)
    if args.device.startswith("cuda"):
        model.half()                          # P100: fp16 (bf16 is emulated and slow)
    model.max_seq_length = args.max_len

    emb = np.empty((len(texts), model.get_sentence_embedding_dimension()), dtype=np.float16)
    step, t1 = 100_000, time.time()
    for i in range(0, len(texts), step):
        e = model.encode(texts[i:i + step], batch_size=args.batch, normalize_embeddings=True,
                         convert_to_tensor=True, show_progress_bar=False)
        emb[i:i + len(e)] = e.to(torch.float16).cpu().numpy()
        done = i + len(e)
        rate = done / (time.time() - t1)
        print(f"  {done:>9,}/{len(texts):,}  {rate:,.0f} rec/s  eta {(len(texts) - done) / rate / 60:.1f} min",
              flush=True)

    tmp = out + ".tmp.npy"
    np.save(tmp, emb)
    os.replace(tmp, out)                      # atomic: a chunk file is either complete or absent
    print(f"saved {emb.shape} in {time.time() - t0:.0f}s total")


if __name__ == "__main__":
    main()
