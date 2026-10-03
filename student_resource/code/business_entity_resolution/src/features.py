"""Pair features (and labels for train) for every candidate pair the matcher scores.

Input : WORK_DIR/cand/<tag>/<split>_<country>_<source>.parquet   (from block.py)
Output: WORK_DIR/feat/<tag>/<split>.parquet

The candidate set is pruned to  fwd_rank < --k-fwd  OR  rev_rank < --k-rev ; exactly
these pairs are what the model scores, so predict.py writes them to candidate_pairs.tsv.

Feature groups
  embedding context : cos, forward / reverse ranks, mutual-best flag, margins to the S1
                      record's best candidate and to the candidate's best S1 record,
                      the candidate's own top-1 vs top-2 gap, candidate-list sizes
  name strings      : rapidfuzz ratio / token-set / token-sort / partial / Jaro-Winkler on the
                      romanized name, the legal-form-free core and the consonant skeleton
  address strings   : ratio / token-set / partial, digit-token overlap, first number and
                      postal-code equality
  meta              : source (2/3), candidate script, lengths, empty-address flag
  name rarity (--idf): overlap of the core names weighted by word rarity among S1 records of
                      the same country label; rarest missing / extra word (see idf_features)
  (--idf-addr)      : the same for address words (rare locality / street names weigh most)
  (--soft-tfidf)    : soft TF-IDF of core names (typo-tolerant word matching, rarity weighted)
  (--char-tfidf-addr): character 3-gram TF-IDF cosine of the addresses
  (--char-tfidf)    : cosine of character 3-gram TF-IDF name vectors, fitted on the split's own
                      names (typo-robust: "sdpreeme" still shares most rare 3-grams with "supreme")
  list context      : per S1 record, each candidate's gap to the best candidate and its rank
                      for cos / name / core / skeleton / address / number similarities, plus
                      counts of strong candidates (see list_context_features)
Country is never a feature (the test set has a country absent from training).

Training rows are sub-sampled by S1 entity: folds listed in --train-folds (default A,B)
at --train-frac, fold C at --val-frac (see common.FOLD_*).

Usage:  python src/features.py --split train --tag labse0
        python src/features.py --split test  --tag labse0
"""
import argparse
import glob
import os
import re
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

from common import FOLD_A, FOLD_B, FOLD_C, read_tsv, s1_fold
from normalize import load_normalized

SRC_NAMES = {2: "source2", 3: "source3"}


def pair_key(src, s1_row, c_row, n_s1):
    """Unique int64 per (source, S1 row, candidate row); candidate rows are < 10**7."""
    return (np.asarray(src, np.int64) * n_s1 + np.asarray(s1_row, np.int64)) * 10**7 + np.asarray(c_row, np.int64)


def load_candidates(work_dir, tag, split):
    parts = []
    for f in sorted(glob.glob(os.path.join(work_dir, "cand", tag, f"{split}_*_source[23].parquet"))):
        df = pd.read_parquet(f)
        df["src"] = np.int8(int(re.search(r"source([23])\.parquet$", f).group(1)))
        parts.append(df)
    if not parts:
        raise FileNotFoundError(f"no candidate files for {split} in {work_dir}/cand/{tag}")
    return pd.concat(parts, ignore_index=True)


def context_features(cand, n_s1, n_src, k_fwd, k_rev):
    """Embedding-context features from the full stored lists, then prune to the cut-offs."""
    s1 = cand["s1_row"].to_numpy()
    cr = cand["c_row"].to_numpy()
    src = cand["src"].to_numpy()
    cos = cand["cos"].to_numpy().astype(np.float32)
    fr = cand["fwd_rank"].to_numpy()
    rr = cand["rev_rank"].to_numpy()

    s1_best_src = np.full((4, n_s1), np.nan, np.float32)      # indexed [src, s1_row]
    s1_second_src = np.full((4, n_s1), np.nan, np.float32)
    c_best = {s: np.full(n_src[s], np.nan, np.float32) for s in (2, 3)}
    c_second = {s: np.full(n_src[s], np.nan, np.float32) for s in (2, 3)}
    for s in (2, 3):
        m = src == s
        m0, m1 = m & (fr == 0), m & (fr == 1)
        s1_best_src[s, s1[m0]] = cos[m0]
        s1_second_src[s, s1[m1]] = cos[m1]
        r0, r1 = m & (rr == 0), m & (rr == 1)
        c_best[s][cr[r0]] = cos[r0]
        c_second[s][cr[r1]] = cos[r1]
    s1_best = np.fmax(s1_best_src[2], s1_best_src[3])

    keep = ((fr >= 0) & (fr < k_fwd)) | ((rr >= 0) & (rr < k_rev))
    kh = cand["key_hits"].to_numpy() if "key_hits" in cand.columns else None
    if kh is not None:                       # second channel (key_block.py): pairs sharing rare keys
        keep |= kh > 0
    c = cand.loc[keep, ["s1_row", "c_row", "src"]].reset_index(drop=True)
    s1, cr, src, cos, fr, rr = s1[keep], cr[keep], src[keep], cos[keep], fr[keep], rr[keep]

    cb = np.where(src == 2, c_best[2][np.where(src == 2, cr, 0)], c_best[3][np.where(src == 3, cr, 0)])
    cs = np.where(src == 2, c_second[2][np.where(src == 2, cr, 0)], c_second[3][np.where(src == 3, cr, 0)])
    sb_src = s1_best_src[src, s1]
    ss_src = s1_second_src[src, s1]
    ckey = src.astype(np.int64) * (max(n_src.values()) + 1) + cr
    _, cinv, ccnt = np.unique(ckey, return_inverse=True, return_counts=True)

    c["cos"] = cos
    c["fwd_rank"] = np.where(fr >= 0, fr, 99).astype(np.int8)
    c["rev_rank"] = np.where(rr >= 0, rr, 99).astype(np.int8)
    c["mutual_best"] = ((fr == 0) & (rr == 0)).astype(np.int8)
    c["m_s1_best"] = s1_best[s1] - cos                 # 0 when this is the S1's best candidate overall
    c["m_s1_best_src"] = sb_src - cos
    c["s1_gap12_src"] = sb_src - ss_src                # how clear the S1's best in this source is
    c["m_c_best"] = cb - cos                           # 0 when this S1 is the candidate's best S1
    c["c_gap12"] = cb - cs                             # how clear the candidate's best S1 is
    c["n_cand_s1"] = np.bincount(s1, minlength=n_s1)[s1].astype(np.int16)
    c["n_cand_c"] = ccnt[cinv].astype(np.int16)
    order = np.lexsort((-cos, s1))
    rank = np.empty(len(s1), np.int16)
    grp_start = np.r_[0, np.flatnonzero(np.diff(s1[order])) + 1]
    pos = np.arange(len(s1)) - np.repeat(grp_start, np.diff(np.r_[grp_start, len(s1)]))
    rank[order] = pos
    c["s1_rank_all"] = rank
    if kh is not None:
        c["key_hits"] = kh[keep].astype(np.int8)
        c["key_only"] = ((fr < 0) & (rr < 0)).astype(np.int8)
    return c


def build_idf(core, country):
    """Document frequency of each core-name word among the S1 records of each country label
    (labels come from the data). Returns ({(label, word): df}, {label: n_records})."""
    df, n = {}, {}
    for c, text in zip(country, core):
        n[c] = n.get(c, 0) + 1
        for t in set(text.split()):
            df[(c, t)] = df.get((c, t), 0) + 1
    return df, n


def idf_features(core1, core2, country, df, n, df2, n2, prefix=""):
    """Name overlap weighted by word rarity within the S1 record's country label.

    S1 words are weighted by their rarity among S1 names, candidate words by their rarity among
    S2/S3 names of the same label (so noise words frequent in S2/S3, e.g. "mr", weigh little).
    idf_rec   : share of the S1 name's information (sum of idf) found in the candidate name
    idf_prec  : share of the candidate name's information found in the S1 name
    idf_miss  : idf of the rarest S1 word missing from the candidate  (a distinctive word lost)
    idf_extra : idf of the rarest candidate word absent from the S1 name ("Defi Replay *Union*")
    s1_idf    : total distinctiveness of the S1 name (rare names can match on name alone)
    Words never seen in the label get the maximum idf of that label.
    """
    out = np.full((len(core1), 5), np.nan, np.float32)
    log = np.log
    for i, (a, b, c) in enumerate(zip(core1, core2, country)):
        t1, t2 = set(a.split()), set(b.split())
        if not t1 or not t2:
            continue
        N, N2 = n.get(c, 1) + 1, n2.get(c, 1) + 1
        w1 = {t: log(N / (df.get((c, t), 0) + 1)) for t in t1}
        w2 = {t: log(N2 / (df2.get((c, t), 0) + 1)) for t in t2}
        both = t1 & t2
        s1, s2 = sum(w1.values()), sum(w2.values())
        out[i, 0] = sum(w1[t] for t in both) / s1 if s1 > 0 else np.nan
        out[i, 1] = sum(w2[t] for t in both) / s2 if s2 > 0 else np.nan
        out[i, 2] = max((w1[t] for t in t1 - t2), default=0.0)
        out[i, 3] = max((w2[t] for t in t2 - t1), default=0.0)
        out[i, 4] = s1
    names = ["idf_rec", "idf_prec", "idf_miss", "idf_extra", "s1_idf"]
    return pd.DataFrame(out, columns=[prefix + c for c in names])


def soft_tfidf(core1, core2, country, df, n, df2, n2, theta=0.9):
    """Soft TF-IDF similarity of the core names (Cohen, Ravikumar & Fienberg, 2003).

    Word weights are idf within the country label (S1 side from S1 names, candidate side from
    S2/S3 names), L2-normalised per name. Every S1 word is paired with its most similar
    candidate word by Jaro-Winkler; pairs with similarity >= theta contribute
    w1 * w2 * similarity. Unlike exact-word overlap it credits typo variants
    ("sdpreeme" ~ "supreme"), and unlike plain fuzzy ratios it weights them by rarity.
    """
    out = np.full(len(core1), np.nan, np.float32)
    log, sqrt, jw = np.log, np.sqrt, JaroWinkler.normalized_similarity
    for i, (a, b, c) in enumerate(zip(core1, core2, country)):
        t1, t2 = list(set(a.split())), list(set(b.split()))
        if not t1 or not t2:
            continue
        N, N2 = n.get(c, 1) + 1, n2.get(c, 1) + 1
        w1 = [log(N / (df.get((c, t), 0) + 1)) for t in t1]
        w2 = [log(N2 / (df2.get((c, t), 0) + 1)) for t in t2]
        z1, z2 = sqrt(sum(x * x for x in w1)) or 1.0, sqrt(sum(x * x for x in w2)) or 1.0
        tot = 0.0
        for u, wu in zip(t1, w1):
            best, bj = 0.0, -1
            for j, v in enumerate(t2):
                sim = 1.0 if u == v else jw(u, v)
                if sim > best:
                    best, bj = sim, j
            if best >= theta:
                tot += (wu / z1) * (w2[bj] / z2) * best
        out[i] = tot
    return out


LIST_CONTEXT_COLS = ["cos", "name_ratio", "name_tset", "core_jw", "skel_ratio", "addr_tset", "nums_tset"]
LIST_CONTEXT_IDF = ["idf_rec", "idf_prec", "addr_idf_rec", "name_tfidf", "soft_tfidf", "addr_tfidf"]


def list_context_features(f):
    """Where each candidate stands within its S1 record's full candidate list (both sources).

    A true match usually looks about as good as the best candidate of its S1 record, while a
    look-alike trails it; several near-equal strong candidates mean a large cluster. For each
    column in LIST_CONTEXT_COLS: <col>_dmax = list maximum minus this value, <col>_rk = rank in
    the list (1 = best). Plus counts of strong candidates in the list. Computed per S1 record
    only (its full list is always present, in train and test alike), so there is no leakage and
    no train/test mismatch. Adds columns to f in place.
    """
    g = f.groupby("s1_row", sort=False)
    for col in LIST_CONTEXT_COLS + [c for c in LIST_CONTEXT_IDF if c in f.columns]:
        f[f"{col}_dmax"] = (g[col].transform("max") - f[col]).astype(np.float32)
        f[f"{col}_rk"] = g[col].rank(method="min", ascending=False).astype(np.float32)
    strong_name = (f["name_tset"] >= 90).astype(np.int16)
    strong_both = ((f["name_tset"] >= 85) & (f["addr_tset"].fillna(0) >= 80)).astype(np.int16)
    f["n_strong_name"] = strong_name.groupby(f["s1_row"]).transform("sum").astype(np.int16)
    f["n_strong_both"] = strong_both.groupby(f["s1_row"]).transform("sum").astype(np.int16)
    f["n_strong_both_src"] = strong_both.groupby([f["s1_row"], f["src"]]).transform("sum").astype(np.int16)


def _cp(a, b, scorer, **kw):
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32, **kw)


def string_features(c, n1, n2):
    """String-similarity features for aligned object arrays of normalized fields."""
    f = {}
    for col, scorers in [("name", ["ratio", "tset", "tsort", "partial", "jw"]),
                         ("core", ["ratio", "tset", "jw"]),
                         ("skel", ["ratio", "tset"]),
                         ("addr", ["ratio", "tset", "partial"])]:
        a, b = n1[col], n2[col]
        for sc in scorers:
            if sc == "ratio":
                v = _cp(a, b, fuzz.ratio)
            elif sc == "tset":
                v = _cp(a, b, fuzz.token_set_ratio)
            elif sc == "tsort":
                v = _cp(a, b, fuzz.token_sort_ratio)
            elif sc == "partial":
                v = _cp(a, b, fuzz.partial_ratio)
            else:
                v = _cp(a, b, JaroWinkler.normalized_similarity) * 100
            f[f"{col}_{sc}"] = v
    empty_addr = (pd.Series(n1["addr"]).str.len().to_numpy() == 0) | (pd.Series(n2["addr"]).str.len().to_numpy() == 0)
    for k in ("addr_ratio", "addr_tset", "addr_partial"):
        f[k][empty_addr] = np.nan
    nums_a, nums_b = n1["nums"], n2["nums"]
    ne = (pd.Series(nums_a).str.len().to_numpy() == 0) | (pd.Series(nums_b).str.len().to_numpy() == 0)
    f["nums_tset"] = _cp(nums_a, nums_b, fuzz.token_set_ratio)
    f["nums_tset"][ne] = np.nan
    for col in ("num1", "postal"):
        a, b = n1[col], n2[col]
        miss = (pd.Series(a).str.len().to_numpy() == 0) | (pd.Series(b).str.len().to_numpy() == 0)
        f[f"{col}_eq"] = np.where(miss, np.nan, (a == b).astype(np.float32)).astype(np.float32)
    f["c_script"] = n2["script"].astype(np.int8)
    f["c_addr_empty"] = n2["addr_raw_empty"].astype(np.int8)
    len_a, len_b = pd.Series(n1["name"]).str.len().to_numpy(), pd.Series(n2["name"]).str.len().to_numpy()
    f["len_name_s1"], f["len_name_c"] = len_a.astype(np.int16), len_b.astype(np.int16)
    f["len_ratio_name"] = (np.minimum(len_a, len_b) / np.maximum(np.maximum(len_a, len_b), 1)).astype(np.float32)
    la, lb = pd.Series(n1["addr"]).str.len().to_numpy(), pd.Series(n2["addr"]).str.len().to_numpy()
    f["len_ratio_addr"] = np.where(empty_addr, np.nan, np.minimum(la, lb) / np.maximum(np.maximum(la, lb), 1)).astype(np.float32)
    return pd.DataFrame(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.environ.get("DATA_DIR"))
    ap.add_argument("--work-dir", default=os.environ.get("WORK_DIR"))
    ap.add_argument("--tag", default="labse0")
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--k-fwd", type=int, default=10)
    ap.add_argument("--k-rev", type=int, default=3)
    ap.add_argument("--train-folds", default="AB", help="folds used for LightGBM training rows")
    ap.add_argument("--train-frac", type=float, default=0.25)
    ap.add_argument("--val-frac", type=float, default=0.5)
    ap.add_argument("--chunk", type=int, default=2_000_000)
    ap.add_argument("--idf", action="store_true",
                    help="add rarity-weighted name-overlap features (idf per country label, from the data)")
    ap.add_argument("--idf-addr", action="store_true",
                    help="same rarity-weighted overlap for address words (addr_idf_*)")
    ap.add_argument("--char-tfidf", action="store_true",
                    help="cosine of character 3-gram TF-IDF vectors of the names (typo-robust rarity weighting)")
    ap.add_argument("--soft-tfidf", action="store_true",
                    help="soft TF-IDF of core names (needs --idf): typo-tolerant, rarity-weighted")
    ap.add_argument("--char-tfidf-addr", action="store_true",
                    help="cosine of character 3-gram TF-IDF vectors of the addresses")
    ap.add_argument("--no-list-context", action="store_true",
                    help="skip the per-S1 list-context features (rounds 1-3 were built without them)")
    args = ap.parse_args()
    if args.soft_tfidf and not args.idf:
        ap.error("--soft-tfidf needs --idf (it reuses the rarity tables)")
    t0 = time.time()

    norm = {s: load_normalized(args.data_dir, args.work_dir, args.split, f"source{s}") for s in (1, 2, 3)}
    n_s1 = len(norm[1])
    n_src = {2: len(norm[2]), 3: len(norm[3])}
    print(f"normalized tables ready in {time.time() - t0:.0f}s")

    cand = load_candidates(args.work_dir, args.tag, args.split)
    print(f"{len(cand):,} stored candidate pairs")
    c = context_features(cand, n_s1, n_src, args.k_fwd, args.k_rev)
    del cand
    print(f"{len(c):,} pairs after cut-off k_fwd={args.k_fwd}, k_rev={args.k_rev} "
          f"({len(c) / n_s1:.1f} per S1) in {time.time() - t0:.0f}s")

    if args.split == "train":
        fold = s1_fold(norm[1]["entity_id"].to_numpy())
        fold_letter = np.where(np.isin(fold, FOLD_A), "A", np.where(np.isin(fold, FOLD_B), "B", "C"))
        rng = np.random.default_rng(0)
        u = rng.random(n_s1)
        in_train = np.isin(fold_letter, list(args.train_folds)) & (u < args.train_frac)
        in_val = (fold_letter == "C") & (u < args.val_frac)
        sel = in_train | in_val
        os.makedirs(os.path.join(args.work_dir, "feat", args.tag), exist_ok=True)
        np.save(os.path.join(args.work_dir, "feat", args.tag, "val_s1_rows.npy"), np.flatnonzero(in_val))
        c = c[sel[c["s1_row"].to_numpy()]].reset_index(drop=True)
        c["fold"] = fold_letter[c["s1_row"].to_numpy()]
        c["is_val"] = in_val[c["s1_row"].to_numpy()].astype(np.int8)
        # labels from the ground truth
        gt = read_tsv(os.path.join(args.data_dir, "train", "train_ground_truth.tsv"))
        ex = gt.assign(m=gt["matched_entity_ids"].str.split(",")).explode("m")
        ex = ex[ex["m"].str.len() > 0]
        s1_row = pd.Series(np.arange(n_s1), index=norm[1]["entity_id"].to_numpy())
        keys = []
        for s in (2, 3):
            e = ex[ex["m"].str.startswith(f"S{s}-")]
            cr = pd.Series(np.arange(n_src[s]), index=norm[s]["entity_id"].to_numpy())
            keys.append(pair_key(np.full(len(e), s), s1_row[e["source1_entity_id"]].to_numpy(),
                                 cr[e["m"]].to_numpy(), n_s1))
        true = np.unique(np.concatenate(keys))
        ck = pair_key(c["src"].to_numpy(), c["s1_row"].to_numpy(), c["c_row"].to_numpy(), n_s1)
        c["label"] = np.isin(ck, true).astype(np.int8)
        print(f"train rows {int((1 - c['is_val']).sum()):,} | val rows {int(c['is_val'].sum()):,} | "
              f"positives {c['label'].mean():.3%}")

    out = os.path.join(args.work_dir, "feat", args.tag, f"{args.split}.parquet")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cols1 = {k: norm[1][k].to_numpy() for k in ("name", "core", "skel", "addr", "nums", "num1", "postal")}
    s1_country = norm[1]["country"].to_numpy()
    if args.idf:
        idf_df, idf_n = build_idf(cols1["core"], s1_country)
        idf_df2, idf_n2 = build_idf(np.concatenate([norm[2]["core"].to_numpy(), norm[3]["core"].to_numpy()]),
                                    np.concatenate([norm[2]["country"].to_numpy(), norm[3]["country"].to_numpy()]))
        # candidate-side country label = the S1 record's label (matches never cross labels)
    if args.idf_addr:
        a_df, a_n = build_idf(cols1["addr"], s1_country)
        a_df2, a_n2 = build_idf(np.concatenate([norm[2]["addr"].to_numpy(), norm[3]["addr"].to_numpy()]),
                                np.concatenate([norm[2]["country"].to_numpy(), norm[3]["country"].to_numpy()]))
    if args.char_tfidf:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=2, sublinear_tf=True,
                              dtype=np.float32)
        vec.fit(pd.unique(np.concatenate([norm[k]["name"].to_numpy() for k in (1, 2, 3)])))
        X1 = vec.transform(cols1["name"])
        X23 = vec.transform(np.concatenate([norm[2]["name"].to_numpy(), norm[3]["name"].to_numpy()]))
        off3 = len(norm[2])
        print(f"char 3-gram TF-IDF: {len(vec.vocabulary_):,} n-grams, fitted in {time.time() - t0:.0f}s", flush=True)
    if args.char_tfidf_addr:
        from sklearn.feature_extraction.text import TfidfVectorizer
        avec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=2, sublinear_tf=True,
                               dtype=np.float32)
        avec.fit(pd.unique(np.concatenate([norm[k]["addr"].to_numpy() for k in (1, 2, 3)])))
        A1 = avec.transform(cols1["addr"])
        A23 = avec.transform(np.concatenate([norm[2]["addr"].to_numpy(), norm[3]["addr"].to_numpy()]))
        aoff3 = len(norm[2])
        print(f"address char 3-gram TF-IDF: {len(avec.vocabulary_):,} n-grams, {time.time() - t0:.0f}s", flush=True)
    cols2 = {s: {k: norm[s][k].to_numpy() for k in ("name", "core", "skel", "addr", "nums", "num1", "postal",
                                                   "script", "addr_raw_empty")} for s in (2, 3)}
    parts = []
    for lo in range(0, len(c), args.chunk):
        part = c.iloc[lo:lo + args.chunk].reset_index(drop=True)
        s1r, crow, src = part["s1_row"].to_numpy(), part["c_row"].to_numpy(), part["src"].to_numpy()
        n1 = {k: v[s1r] for k, v in cols1.items()}
        n2 = {}
        for k in cols2[2]:
            a = cols2[2][k][np.where(src == 2, crow, 0)]
            b = cols2[3][k][np.where(src == 3, crow, 0)]
            n2[k] = np.where(src == 2, a, b)
        sf = string_features(part, n1, n2)
        if args.idf:
            sf = pd.concat([sf, idf_features(n1["core"], n2["core"], s1_country[s1r], idf_df, idf_n,
                                             idf_df2, idf_n2)], axis=1)
        if args.idf_addr:
            sf = pd.concat([sf, idf_features(n1["addr"], n2["addr"], s1_country[s1r], a_df, a_n, a_df2, a_n2,
                                             prefix="addr_")], axis=1)
        if args.char_tfidf:
            ci = np.where(src == 2, crow, crow + off3)
            sf["name_tfidf"] = np.asarray(X1[s1r].multiply(X23[ci]).sum(axis=1)).ravel().astype(np.float32)
        if args.soft_tfidf:
            sf["soft_tfidf"] = soft_tfidf(n1["core"], n2["core"], s1_country[s1r], idf_df, idf_n, idf_df2, idf_n2)
        if args.char_tfidf_addr:
            ci = np.where(src == 2, crow, crow + aoff3)
            v = np.asarray(A1[s1r].multiply(A23[ci]).sum(axis=1)).ravel().astype(np.float32)
            empty = (pd.Series(n1["addr"]).str.len().to_numpy() == 0) | (pd.Series(n2["addr"]).str.len().to_numpy() == 0)
            v[empty] = np.nan
            sf["addr_tfidf"] = v
        parts.append(sf)
        print(f"  string features {min(lo + args.chunk, len(c)):>11,}/{len(c):,}  {time.time() - t0:.0f}s", flush=True)
    del cols1, cols2, norm
    feats = pd.concat([c, pd.concat(parts, ignore_index=True)], axis=1)
    del parts, c
    if not args.no_list_context:
        list_context_features(feats)
        print(f"  list-context features added  {time.time() - t0:.0f}s", flush=True)

    writer = None
    for lo in range(0, len(feats), args.chunk):
        tbl = pa.Table.from_pandas(feats.iloc[lo:lo + args.chunk], preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out + ".tmp", tbl.schema)
        writer.write_table(tbl)
    writer.close()
    os.replace(out + ".tmp", out)
    print(f"wrote {out}: {len(feats):,} rows x {feats.shape[1]} columns in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
