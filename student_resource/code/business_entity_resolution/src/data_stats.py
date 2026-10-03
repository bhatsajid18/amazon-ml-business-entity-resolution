"""Dataset profiling for the Business Entity Resolution challenge (CPU only).

Answers the questions that fix the pipeline's design before any model is built:
  * how many matches an S1 entity has, per source (sets the forward top-k),
  * whether each S2/S3 record belongs to at most one S1 entity (one-parent rule),
  * how many S2/S3 records match nothing (orphans: the precision risk),
  * whether matches ever cross country labels (decides per-country index shards),
  * which scripts appear per source and country, and how long the texts are,
  * whether the files parse cleanly with tab separation and no quoting.
Prints readable tables and a few example clusters, and writes a JSON summary.

Usage:  python src/data_stats.py --data-dir ../../dataset --out ../../work/data_stats.json
"""
import argparse
import csv
import json
import os
import time

import numpy as np
import pandas as pd

# Unicode blocks used to label each string with its dominant script.
SCRIPTS = {
    "latin": "A-Za-zÀ-ɏ",
    "devanagari": "ऀ-ॿ",
    "bengali": "ঀ-৿",
    "gurmukhi": "਀-੿",
    "gujarati": "઀-૿",
    "oriya": "଀-୿",
    "tamil": "஀-௿",
    "telugu": "ఀ-౿",
    "kannada": "ಀ-೿",
    "malayalam": "ഀ-ൿ",
    "arabic": "؀-ۿ",
}
SAMPLE = 200_000
SUMMARY = {}


def section(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def count_records(path):
    """Data rows in a TSV counted from raw newlines (minus header)."""
    n, last = 0, b"\n"
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            n += chunk.count(b"\n")
            last = chunk[-1:]
    if last != b"\n":
        n += 1
    return n - 1


def read_tsv(path):
    """Read a challenge TSV literally (tab separator, no quote handling) and check it."""
    t = time.time()
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                     quoting=csv.QUOTE_NONE, on_bad_lines="skip")
    raw = count_records(path)
    quoted = int(sum(df[c].str.startswith('"').sum() for c in df.columns))
    print(f"  {os.path.basename(path):28s} rows {len(df):>10,d} | raw lines {raw:>10,d} | "
          f"skipped {raw - len(df):,d} | fields starting with a quote {quoted:,d} | "
          f"{time.time() - t:.0f}s")
    return df


def dominant_script(s):
    """Label each string with the script that has the most characters in it."""
    counts = pd.DataFrame({k: s.str.count(f"[{v}]") for k, v in SCRIPTS.items()})
    dom = counts.idxmax(axis=1)
    dom[counts.max(axis=1) == 0] = "none"
    return dom


def sample(df, n=SAMPLE):
    return df.sample(n=min(n, len(df)), random_state=0)


def pct(x):
    return f"{100 * x:.2f}%"


def jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, pd.Series):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, pd.DataFrame):
        return {str(k): jsonable(v) for k, v in o.to_dict().items()}
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    return o


def profile_sources(frames, split):
    """Counts, duplicate ids, country labels, scripts and lengths for each source file."""
    section(f"{split.upper()}: source files")
    out = {}
    for src, df in frames.items():
        info = {"rows": len(df), "duplicate_ids": int(df["entity_id"].duplicated().sum()),
                "bad_prefix": int((~df["entity_id"].str.startswith(f"S{src[-1]}-")).sum()),
                "empty_name": int((df["business_name"].str.strip() == "").sum()),
                "empty_address": int((df["business_address"].str.strip() == "").sum()),
                "country": df["country"].value_counts().to_dict()}
        s = sample(df)
        name_scr, addr_scr = dominant_script(s["business_name"]), dominant_script(s["business_address"])
        info["name_script_by_country"] = pd.crosstab(s["country"], name_scr, normalize="index", colnames=["name script"]).round(4)
        info["name_vs_address_script_differs"] = float((name_scr != addr_scr).mean())
        nl, al = s["business_name"].str.len(), s["business_address"].str.len()
        info["name_len_p50_p95_p99"] = np.percentile(nl, [50, 95, 99]).tolist()
        info["addr_len_p50_p95_p99"] = np.percentile(al, [50, 95, 99]).tolist()
        out[src] = info

        print(f"\n[{src}] rows {info['rows']:,} | duplicate ids {info['duplicate_ids']} | "
              f"wrong id prefix {info['bad_prefix']} | empty name {info['empty_name']} | "
              f"empty address {info['empty_address']}")
        print(f"  country labels: {info['country']}")
        print(f"  name script share by country (sample of {len(s):,}):")
        print("    " + info["name_script_by_country"].to_string().replace("\n", "\n    "))
        print(f"  name script != address script: {pct(info['name_vs_address_script_differs'])}")
        print(f"  name chars p50/p95/p99: {info['name_len_p50_p95_p99']} | "
              f"address chars p50/p95/p99: {info['addr_len_p50_p95_p99']}")
    return out


def profile_ground_truth(gt, s1, s2, s3):
    """Match-count distribution, one-parent check, orphans, cross-country matches."""
    section("TRAIN: ground truth")
    res = {}
    ex = gt[["source1_entity_id"]].assign(mid=gt["matched_entity_ids"].str.split(",")).explode("mid")
    ex["mid"] = ex["mid"].astype(str).str.strip()
    ex = ex[ex["mid"].str.len() > 0].reset_index(drop=True)
    ex["src"] = ex["mid"].str[:2]

    res["gt_rows"] = len(gt)
    res["gt_duplicate_s1_rows"] = int(gt["source1_entity_id"].duplicated().sum())
    res["s1_ids_missing_from_gt"] = int((~s1["entity_id"].isin(gt["source1_entity_id"])).sum())
    res["gt_s1_ids_not_in_source1"] = int((~gt["source1_entity_id"].isin(s1["entity_id"])).sum())
    res["duplicate_ids_within_a_list"] = int(ex.duplicated(["source1_entity_id", "mid"]).sum())
    known = set(s2["entity_id"]) | set(s3["entity_id"])
    res["matched_ids_not_in_s2_s3"] = int((~ex["mid"].isin(known)).sum())
    res["matched_prefix_counts"] = ex["src"].value_counts().to_dict()

    per = ex.groupby(["source1_entity_id", "src"]).size().unstack(fill_value=0)
    per = per.reindex(gt["source1_entity_id"].unique(), fill_value=0)
    for c in ["S2", "S3"]:
        if c not in per:
            per[c] = 0
    total = per["S2"] + per["S3"]
    res["matches_per_s1_hist"] = total.value_counts().sort_index().to_dict()
    res["singleton_fraction"] = float((total == 0).mean())
    res["mean_matches_all"] = float(total.mean())
    res["mean_matches_non_singleton"] = float(total[total > 0].mean())
    res["max_matches"] = int(total.max())
    res["max_from_s2"], res["max_from_s3"] = int(per["S2"].max()), int(per["S3"].max())
    res["p99_from_s2"] = float(np.percentile(per["S2"], 99))
    res["p99_from_s3"] = float(np.percentile(per["S3"], 99))

    print(f"GT rows {res['gt_rows']:,} | duplicate S1 rows {res['gt_duplicate_s1_rows']} | "
          f"S1 ids missing from GT {res['s1_ids_missing_from_gt']} | "
          f"GT ids not in source1 {res['gt_s1_ids_not_in_source1']}")
    print(f"dup ids inside a list {res['duplicate_ids_within_a_list']} | "
          f"matched ids not found in S2/S3 {res['matched_ids_not_in_s2_s3']} | "
          f"prefixes {res['matched_prefix_counts']}")
    print(f"\nmatches per S1 (count: entities): {res['matches_per_s1_hist']}")
    print(f"singletons {pct(res['singleton_fraction'])} | mean {res['mean_matches_all']:.2f} "
          f"(non-singletons {res['mean_matches_non_singleton']:.2f}) | max {res['max_matches']}")
    print(f"per source: max from S2 {res['max_from_s2']} (p99 {res['p99_from_s2']:.0f}) | "
          f"max from S3 {res['max_from_s3']} (p99 {res['p99_from_s3']:.0f})")

    # one-parent rule: can an S2/S3 record belong to more than one S1 entity?
    parents = ex["mid"].value_counts()
    res["records_with_multiple_parents"] = int((parents > 1).sum())
    res["max_parents"] = int(parents.max()) if len(parents) else 0
    print(f"\nS2/S3 records linked to >1 S1: {res['records_with_multiple_parents']:,} "
          f"(max parents {res['max_parents']})  -> one-parent rule is "
          f"{'SAFE' if res['records_with_multiple_parents'] == 0 else 'NOT exact'}")

    # orphans: S2/S3 records that match nothing
    matched = set(ex["mid"])
    for name, df in [("S2", s2), ("S3", s3)]:
        orphan = float((~df["entity_id"].isin(matched)).mean())
        res[f"orphan_fraction_{name}"] = orphan
        print(f"{name} records with no S1 match: {pct(orphan)}")

    # country consistency: do matches cross country labels?
    c1 = s1.drop_duplicates("entity_id").set_index("entity_id")["country"]
    c23 = pd.concat([s2, s3]).drop_duplicates("entity_id").set_index("entity_id")["country"]
    ex["c_s1"], ex["c_m"] = ex["source1_entity_id"].map(c1), ex["mid"].map(c23)
    res["cross_country_fraction"] = float((ex["c_s1"] != ex["c_m"]).mean())
    res["country_pairs"] = pd.crosstab(ex["c_s1"], ex["c_m"], rownames=["S1 country"], colnames=["match country"])
    print(f"\nmatched pairs whose country labels differ: {pct(res['cross_country_fraction'])} "
          f"-> per-country shards are {'SAFE' if res['cross_country_fraction'] == 0 else 'RISKY'}")
    print(res["country_pairs"].to_string())

    g = pd.DataFrame({"country": gt["source1_entity_id"].map(c1).values,
                      "n": total.reindex(gt["source1_entity_id"]).values})
    res["by_country"] = g.groupby("country")["n"].agg(
        entities="size", singleton_frac=lambda x: (x == 0).mean(), mean="mean", max="max").round(4)
    print("\nper S1 country:\n" + res["by_country"].to_string())

    # script of S1 name vs script of the matched record's name
    se = sample(ex)
    n1 = s1.drop_duplicates("entity_id").set_index("entity_id")["business_name"]
    n23 = pd.concat([s2, s3]).drop_duplicates("entity_id").set_index("entity_id")["business_name"]
    sc1 = dominant_script(se["source1_entity_id"].map(n1).fillna(""))
    sc2 = dominant_script(se["mid"].map(n23).fillna(""))
    res["pair_script_s1_vs_match"] = pd.crosstab(sc1.values, sc2.values, rownames=["S1 name"], colnames=["match name"])
    res["pair_script_by_source"] = pd.crosstab(se["src"].values, sc2.values, normalize="index",
                                                 rownames=["source"], colnames=["match name"]).round(4)
    print(f"\nname script, S1 (rows) vs matched record (cols), sample {len(se):,}:")
    print(res["pair_script_s1_vs_match"].to_string())
    print("\nmatched-record name script share by source:")
    print(res["pair_script_by_source"].to_string())
    return res, ex


def show_examples(gt, ex, s1, s2, s3, n_per_country=4, seed=0):
    """Print a few clusters per country so the noise patterns can be eyeballed."""
    section("EXAMPLE CLUSTERS")
    rec = pd.concat([s1, s2, s3]).drop_duplicates("entity_id").set_index("entity_id")
    size = ex.groupby("source1_entity_id").size()
    rng = np.random.default_rng(seed)

    def line(i):
        if i not in rec.index:
            return f"    {i}  <missing>"
        r = rec.loc[i]
        return f"    {i}  [{r['country']}]  {r['business_name'][:60]}  |  {r['business_address'][:90]}"

    s1c = s1.drop_duplicates("entity_id").set_index("entity_id")["country"]
    for country in s1["country"].value_counts().index:
        mask = (size >= 2).values & np.asarray(size.index.map(s1c) == country)
        ids = size.index[mask]
        if len(ids) == 0:
            continue
        print(f"\n--- {country}: clusters with >= 2 matches")
        for s in rng.choice(ids, size=min(n_per_country, len(ids)), replace=False):
            print(line(s))
            for m in ex.loc[ex["source1_entity_id"] == s, "mid"]:
                print("  ->" + line(m)[4:])
    singles = gt.loc[gt["matched_entity_ids"].str.strip() == "", "source1_entity_id"]
    if len(singles):
        print("\n--- singletons (no matches)")
        for s in rng.choice(singles.values, size=min(4, len(singles)), replace=False):
            print(line(s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()

    section("READING FILES")
    tr = {s: read_tsv(os.path.join(args.data_dir, "train", f"train_{s}.tsv"))
          for s in ["source1", "source2", "source3"]}
    gt = read_tsv(os.path.join(args.data_dir, "train", "train_ground_truth.tsv"))
    te = {s: read_tsv(os.path.join(args.data_dir, "test", f"test_{s}.tsv"))
          for s in ["source1", "source2", "source3"]}

    SUMMARY["train_sources"] = profile_sources(tr, "train")
    SUMMARY["test_sources"] = profile_sources(te, "test")
    SUMMARY["ground_truth"], ex = profile_ground_truth(gt, tr["source1"], tr["source2"], tr["source3"])
    show_examples(gt, ex, tr["source1"], tr["source2"], tr["source3"])

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(jsonable(SUMMARY), f, indent=2, ensure_ascii=False)
    section(f"DONE in {time.time() - t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
