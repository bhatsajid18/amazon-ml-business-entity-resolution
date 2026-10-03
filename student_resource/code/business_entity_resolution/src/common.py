"""Shared helpers for the Business Entity Resolution pipeline.

* TSV reading: tab separator, no quote handling, everything as strings
  (a few hundred test fields start with a quote character; pandas' default quoting
  would merge rows).
* Text cleaning used for embeddings: NFKC, lowercase, native-script digits -> ASCII,
  brackets / quotes -> space, collapsed whitespace.
* Deterministic A/B/C folds of the training S1 entities.
* Paths of embedding / candidate files, shared by every stage.
"""
import csv
import os
import re
import zlib

import numpy as np
import pandas as pd

SPLITS = ("train", "test")
SOURCES = ("source1", "source2", "source3")
CAND_SOURCES = ("source2", "source3")

# Training S1 entities are split by a hash of their id into three disjoint folds:
#   A (0-39)  -> fine-tune LaBSE (round 2)
#   B (40-79) -> train LightGBM
#   C (80-99) -> tune the threshold and report local F0.5
FOLD_A, FOLD_B, FOLD_C = range(0, 40), range(40, 80), range(80, 100)

# ---------------------------------------------------------------- text cleaning
# Zero digit of every Unicode decimal-digit block likely to appear
# (Arabic-Indic, Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu,
#  Kannada, Malayalam, Sinhala, full-width).
_DIGIT_ZEROS = [0x0660, 0x06F0, 0x0966, 0x09E6, 0x0A66, 0x0AE6, 0x0B66, 0x0BE6,
                0x0C66, 0x0CE6, 0x0D66, 0x0DE6, 0xFF10]
CLEAN_TABLE = {z + d: ord("0") + d for z in _DIGIT_ZEROS for d in range(10)}
for _ch in "[](){}<>\"'`´‘’“”|":
    CLEAN_TABLE[ord(_ch)] = " "
_WS = re.compile(r"\s+")


def clean(s: pd.Series) -> pd.Series:
    """NFKC, lowercase, native digits -> ASCII, brackets/quotes -> space, squeeze spaces."""
    s = s.fillna("").astype(str).str.normalize("NFKC").str.lower().str.translate(CLEAN_TABLE)
    return s.str.replace(_WS, " ", regex=True).str.strip()


def embed_texts(names: pd.Series, addresses: pd.Series) -> list:
    """'name | address' (or just the name when the address is empty), cleaned."""
    n, a = clean(names), clean(addresses)
    return np.where(a.str.len() > 0, n + " | " + a, n).tolist()


# ---------------------------------------------------------------- files
def source_path(data_dir, split, source):
    return os.path.join(data_dir, split, f"{split}_{source}.tsv")


def read_tsv(path, usecols=None):
    """Read a challenge TSV literally (tab separated, no quoting, all strings)."""
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                       quoting=csv.QUOTE_NONE, usecols=usecols)


def load_ids(data_dir, split, source):
    """entity_id and country of one source file, in TSV row order (fast, no normalization)."""
    return read_tsv(source_path(data_dir, split, source), usecols=["entity_id", "country"])


def count_records(path):
    """Data rows in a TSV (newlines minus the header), read in 16 MB chunks."""
    n, last = 0, b"\n"
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            n += chunk.count(b"\n")
            last = chunk[-1:]
    if last != b"\n":
        n += 1
    return n - 1


def s1_fold(ids) -> np.ndarray:
    """Deterministic bucket 0..99 per S1 id (crc32), used for the A/B/C folds."""
    return np.fromiter((zlib.crc32(x.encode()) % 100 for x in ids), dtype=np.int16,
                       count=len(ids))


# ---------------------------------------------------------------- work-dir layout
def emb_dir(work_dir, tag):
    return os.path.join(work_dir, "emb", tag)


def emb_chunk_path(work_dir, tag, split, source, chunk):
    return os.path.join(emb_dir(work_dir, tag), f"{split}_{source}_{chunk:03d}.npy")


def cand_path(work_dir, tag, split, country, source):
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", country)
    return os.path.join(work_dir, "cand", tag, f"{split}_{safe}_{source}.parquet")


def encode_tasks(data_dir, chunk_size):
    """Ordered list of (split, source, chunk, start, stop) covering every record."""
    tasks = []
    for split in SPLITS:
        for source in SOURCES:
            n = count_records(source_path(data_dir, split, source))
            for c, start in enumerate(range(0, n, chunk_size)):
                tasks.append((split, source, c, start, min(start + chunk_size, n)))
    return tasks


def load_embeddings(work_dir, tag, split, source, rows=None):
    """Concatenate the fp16 embedding chunks of one source file, optionally only `rows`
    (sorted global row indices). Uses memory-mapped reads so only selected rows load."""
    d = emb_dir(work_dir, tag)
    files = sorted(f for f in os.listdir(d) if f.startswith(f"{split}_{source}_") and f.endswith(".npy"))
    if not files:
        raise FileNotFoundError(f"no embeddings for {split}/{source} in {d}")
    parts, offset = [], 0
    for f in files:
        m = np.load(os.path.join(d, f), mmap_mode="r")
        if rows is None:
            parts.append(np.asarray(m))
        else:
            lo, hi = np.searchsorted(rows, [offset, offset + len(m)])
            parts.append(np.asarray(m[rows[lo:hi] - offset]))
        offset += len(m)
    return np.concatenate(parts), offset
