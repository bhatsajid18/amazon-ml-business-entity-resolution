"""GPU smoke test and throughput benchmark for the P100 nodes.

1. Confirms PyTorch sees the GPU and was built with kernels for its architecture
   (sm_60 for the P100), and that fp16 matmul works (the P100 has no bf16).
2. Times LaBSE encoding of real "name | address" strings (bi-encoder, blocking).
3. Times XLM-R base scoring of record pairs (cross-encoder, optional re-ranking).
4. Converts both into estimated hours for the full train/test passes, so the
   later jobs can be sized from measured numbers.

Usage:  python src/gpu_check.py --data-dir ../../dataset --n 20000
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

SOURCES = ["source1", "source2", "source3"]


def count_records(path):
    """Number of data rows in a TSV (newlines minus the header), read in 16 MB chunks."""
    n, last = 0, b"\n"
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            n += chunk.count(b"\n")
            last = chunk[-1:]
    if last != b"\n":
        n += 1
    return n - 1


def load_texts(path, n):
    """First n records of a source file as 'name | address' strings."""
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                     quoting=csv.QUOTE_NONE, nrows=n)
    return (df["business_name"] + " | " + df["business_address"]).tolist()


def timed(fn):
    """Run fn with the GPU synchronised before and after; return (result, seconds)."""
    torch.cuda.synchronize()
    t = time.time()
    out = fn()
    torch.cuda.synchronize()
    return out, time.time() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--n", type=int, default=20000, help="records to benchmark on")
    ap.add_argument("--batch", type=int, default=256, help="LaBSE encode batch size")
    ap.add_argument("--max-len", type=int, default=64, help="LaBSE max tokens")
    ap.add_argument("--pair-batch", type=int, default=128)
    ap.add_argument("--pair-max-len", type=int, default=128)
    ap.add_argument("--cands-per-s1", type=float, default=25.0,
                    help="assumed candidates per S1 record, for the cross-encoder estimate")
    args = ap.parse_args()

    # ---------------------------------------------------------------- 1. device
    print(f"torch {torch.__version__} | built for CUDA {torch.version.cuda}")
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Check that the job requested "
              "--gres=gpu:P100-PCIE-16GB:1 and look at the driver version in nvidia-smi above.")
        sys.exit(1)
    dev = torch.device("cuda")
    cap = torch.cuda.get_device_capability(0)
    arch = f"sm_{cap[0]}{cap[1]}"
    props = torch.cuda.get_device_properties(0)
    print(f"GPU {props.name} | {arch} | {props.total_memory / 2**30:.1f} GiB")
    print(f"torch arch list: {torch.cuda.get_arch_list()}")
    if arch not in torch.cuda.get_arch_list():
        print(f"ERROR: this torch build has no kernels for {arch}.")
        sys.exit(1)
    print(f"bf16 supported: {torch.cuda.is_bf16_supported()}  -> use fp16 autocast on P100")

    a = torch.randn(8192, 8192, device=dev, dtype=torch.float16)
    b = torch.randn(8192, 8192, device=dev, dtype=torch.float16)
    timed(lambda: a @ b)                                   # warm-up
    _, dt = timed(lambda: [a @ b for _ in range(10)])
    print(f"fp16 matmul: {10 * 2 * 8192 ** 3 / dt / 1e12:.1f} TFLOPS")
    del a, b
    torch.cuda.empty_cache()

    # ---------------------------------------------------------------- 2. data sizes
    counts = {}
    for split in ["train", "test"]:
        for s in SOURCES:
            p = os.path.join(args.data_dir, split, f"{split}_{s}.tsv")
            counts[(split, s)] = count_records(p)
            print(f"{split:5s} {s}: {counts[(split, s)]:>10,d} records")
    n_train = sum(v for (sp, _), v in counts.items() if sp == "train")
    n_test = sum(v for (sp, _), v in counts.items() if sp == "test")

    texts_s2 = load_texts(os.path.join(args.data_dir, "test", "test_source2.tsv"), args.n)
    texts_s1 = load_texts(os.path.join(args.data_dir, "test", "test_source1.tsv"), args.n)

    # ---------------------------------------------------------------- 3. LaBSE
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/LaBSE", device="cuda")
    model.half()
    model.max_seq_length = args.max_len
    for name, texts in [("S1", texts_s1), ("S2", texts_s2)]:
        lens = np.array([len(x) for x in model.tokenizer(texts[:5000])["input_ids"]])
        pct = np.percentile(lens, [50, 90, 95, 99]).round().astype(int).tolist()
        print(f"LaBSE tokens per {name} record: p50/p90/p95/p99 = {pct}, max = {lens.max()}, "
              f"share > {args.max_len}: {(lens > args.max_len).mean():.2%}")

    model.encode(texts_s2[:1024], batch_size=args.batch)   # warm-up
    torch.cuda.reset_peak_memory_stats()
    _, dt = timed(lambda: model.encode(texts_s2, batch_size=args.batch,
                                       normalize_embeddings=True, convert_to_numpy=True))
    rps = len(texts_s2) / dt
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"LaBSE fp16 encode: {rps:,.0f} records/s (batch {args.batch}, max_len {args.max_len}), "
          f"peak {peak:.1f} GiB")
    del model
    torch.cuda.empty_cache()

    # ---------------------------------------------------------------- 4. XLM-R pairs
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")
    xm = AutoModelForSequenceClassification.from_pretrained(
        "FacebookAI/xlm-roberta-base", num_labels=1).to(dev).half().eval()
    n_pairs = min(len(texts_s1), len(texts_s2), 10000)
    left, right = texts_s1[:n_pairs], texts_s2[:n_pairs]

    def score(lo, hi):
        with torch.inference_mode():
            for i in range(lo, hi, args.pair_batch):
                j = min(i + args.pair_batch, hi)
                enc = tok(left[i:j], right[i:j], truncation=True, max_length=args.pair_max_len,
                          padding=True, return_tensors="pt").to(dev)
                xm(**enc).logits

    score(0, min(512, n_pairs))                             # warm-up
    torch.cuda.reset_peak_memory_stats()
    _, dt = timed(lambda: score(0, n_pairs))
    pps = n_pairs / dt
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"XLM-R base fp16 pair scoring: {pps:,.0f} pairs/s (batch {args.pair_batch}, "
          f"max_len {args.pair_max_len}), peak {peak:.1f} GiB")

    # ---------------------------------------------------------------- 5. estimates
    n_s1_test = counts[("test", "source1")]
    n_cand = args.cands_per_s1 * n_s1_test
    print("\n================ SUMMARY (1 GPU) ================")
    print(f"records  train {n_train:,}  test {n_test:,}")
    print(f"LaBSE encode all test : {n_test / rps / 3600:5.2f} h")
    print(f"LaBSE encode all train: {n_train / rps / 3600:5.2f} h")
    print(f"XLM-R on ALL test candidates (~{n_cand / 1e7:.1f} crore pairs): "
          f"{n_cand / pps / 3600:5.1f} h")
    print(f"XLM-R on 10% uncertain band: {0.1 * n_cand / pps / 3600:5.1f} h")
    print("Divide by the number of GPUs you can run at once (jobs are sharded).")


if __name__ == "__main__":
    main()
