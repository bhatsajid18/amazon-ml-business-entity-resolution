"""Text normalization for string-similarity features.

Per record it produces
  name   : romanized, accent-free, lowercase, abbreviations expanded
  core   : name without legal-form words (private, limited, llc, inc, sarl, ...)
  skel   : consonant skeleton of the name ("gujarat food" and the transliterated
           "gujarat phud" both become "gjrt fd"), robust to transliteration spelling
  addr   : romanized, abbreviations canonicalised, region names -> short codes, stopwords out
  nums   : digit tokens of the address, space separated (house numbers, PIN / ZIP)
  num1   : first digit token of the address ("" if none)
  postal : longest digit token with >= 5 digits (PIN, ZIP, French postcode)
  script : dominant script of the name (0 = Latin/none, 1.. = Indic scripts)

Indic scripts are romanized with indic-transliteration (MIT, rule-based, offline)
to IAST, then accents are stripped; for the north-Indian scripts the inherent word-
final "a" is dropped ("limiteda" -> "limited"). Results are cached per source file in
WORK_DIR/norm/<NORM_VERSION>/<split>_<source>.parquet.
The same tables apply to every record regardless of its country label (English, French
and Indian conventions side by side), so an unseen country needs no special code.
"""
import os
import re
import time
import unicodedata

import numpy as np
import pandas as pd

from common import CLEAN_TABLE, read_tsv, source_path

try:
    from indic_transliteration import sanscript
    SCRIPTS = [  # (id, name, first, last, sanscript scheme, drop final inherent 'a')
        (1, "devanagari", 0x0900, 0x097F, sanscript.DEVANAGARI, True),
        (2, "bengali", 0x0980, 0x09FF, sanscript.BENGALI, True),
        (3, "gurmukhi", 0x0A00, 0x0A7F, sanscript.GURMUKHI, True),
        (4, "gujarati", 0x0A80, 0x0AFF, sanscript.GUJARATI, True),
        (5, "oriya", 0x0B00, 0x0B7F, sanscript.ORIYA, True),
        (6, "tamil", 0x0B80, 0x0BFF, sanscript.TAMIL, False),
        (7, "telugu", 0x0C00, 0x0C7F, sanscript.TELUGU, False),
        (8, "kannada", 0x0C80, 0x0CFF, sanscript.KANNADA, False),
        (9, "malayalam", 0x0D00, 0x0D7F, sanscript.MALAYALAM, False),
    ]
except ImportError:  # keeps the module importable for tools that do not need it
    sanscript, SCRIPTS = None, []
INDIC_RE = re.compile("[ऀ-ൿ]")
NORM_VERSION = "v2"   # cache folder; bump whenever the tables or rules below change

# ---------------------------------------------------------------- shared tables
# One set of tables for every record, whatever its country label: no branch per country.
# Where a short form is ambiguous across languages, both readings are collapsed to the same
# short canonical token instead of being expanded, so every reading still matches itself:
# Street / Saint -> "st", Suite / Sainte -> "ste".
STOPWORDS = {"of", "the", "de", "du", "des", "la", "le", "les", "l", "d"}
LEGAL = {"private", "limited", "pvt", "ltd", "llc", "inc", "incorporated", "corporation", "corp",
         "company", "co", "llp", "lp", "plc", "partners", "partnership", "sarl", "sas", "sa", "eurl",
         "sasu", "sci", "snc", "gmbh", "pllc", "pc", "enterprises", "and", "cie", "compagnie",
         "etablissements", "societe", "ste", "group", "holdings", "firm"} | STOPWORDS
NAME_ABBR = {"pvt": "private", "ltd": "limited", "corp": "corporation",
             "inc": "incorporated", "co": "company", "intl": "international", "mfg": "manufacturing",
             "svcs": "services", "svc": "services", "mgmt": "management", "assoc": "associates",
             "bros": "brothers", "tech": "technologies", "n": "and", "et": "and", "sri": "shri",
             "shree": "shri", "lnc": "incorporated", "1nc": "incorporated",
             "ets": "etablissements", "etab": "etablissements", "cie": "compagnie",
             "saint": "st", "sainte": "ste",
             # common transliterations of English legal / business words
             "praivet": "private", "pryvet": "private", "prayivet": "private", "privet": "private",
             "limitad": "limited", "limitid": "limited", "limted": "limited", "kampani": "company",
             "kampni": "company", "kampany": "company", "kansaltansi": "consultancy",
             "tredars": "traders", "tredarsa": "traders", "treidars": "traders", "enterprajes": "enterprises",
             "saravisej": "services", "sarvisej": "services", "sarvises": "services"}
# OCR-style digit-for-letter swaps inside alphabetic name tokens ("coasta1" -> "coastal")
_OCR = str.maketrans({"0": "o", "1": "l", "3": "e", "5": "s"})
_MIXED = re.compile(r"\b(?=\w*[a-z])(?=\w*\d)\w+\b")
ADDR_ABBR = {"street": "st", "str": "st", "saint": "st", "suite": "ste", "sainte": "ste",
             "rd": "road", "ave": "avenue", "av": "avenue", "dr": "drive", "ln": "lane",
             "blvd": "boulevard", "bd": "boulevard", "bld": "boulevard", "boul": "boulevard",
             "ct": "court", "pl": "place", "r": "rue", "rte": "route", "chem": "chemin", "imp": "impasse",
             "fg": "faubourg", "fbg": "faubourg", "hwy": "highway", "pkwy": "parkway", "cir": "circle",
             "ter": "terrace", "trl": "trail", "sq": "square", "apt": "apartment", "bldg": "building",
             "fl": "floor", "flr": "floor", "nr": "near", "opp": "opposite", "bh": "behind",
             "mkt": "market", "marg": "road", "no": "number", "h": "house", "hno": "house number",
             "n": "north", "s": "south", "e": "east", "w": "west", "ne": "northeast", "nw": "northwest",
             "se": "southeast", "sw": "southwest", "mt": "mount", "ft": "fort", "et": "and"}
ADDR_DROP = STOPWORDS | {"cedex"}          # CEDEX is a delivery marker, present on one side only
# Region names -> one short code (names are collapsed, codes are left alone), for every record.
# Matched records never cross country labels, so a code shared by two regions is harmless.
REGION_PHRASES = {
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny",
    "north carolina": "nc", "north dakota": "nd", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "west virginia": "wv", "district of columbia": "dc",
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "himachal pradesh": "hp",
    "jammu and kashmir": "jk", "madhya pradesh": "mp", "tamil nadu": "tn", "uttar pradesh": "up",
    "west bengal": "wb"}
REGION_TOKENS = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
    "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md", "massachusetts": "ma",
    "michigan": "mi", "minnesota": "mn", "mississippi": "ms", "missouri": "mo", "montana": "mt",
    "nebraska": "ne", "nevada": "nv", "ohio": "oh", "oklahoma": "ok", "oregon": "or",
    "pennsylvania": "pa", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "wisconsin": "wi", "wyoming": "wy",
    "assam": "as", "bihar": "br", "chhattisgarh": "cg", "goa": "ga", "gujarat": "gj", "guj": "gj",
    "haryana": "hr", "jharkhand": "jh", "karnataka": "ka", "kar": "ka", "kerala": "kl", "ker": "kl",
    "maharashtra": "mh", "mah": "mh", "manipur": "mn", "meghalaya": "ml", "mizoram": "mz",
    "nagaland": "nl", "odisha": "od", "orissa": "od", "punjab": "pb", "rajasthan": "rj", "raj": "rj",
    "sikkim": "sk", "telangana": "tg", "ts": "tg", "tripura": "tr", "uttarakhand": "uk",
    "delhi": "dl", "del": "dl", "chandigarh": "ch", "puducherry": "py", "pondicherry": "py",
    "bombay": "mumbai", "calcutta": "kolkata", "madras": "chennai", "bangalore": "bengaluru"}
_REGION_RE = re.compile(r"\b(" + "|".join(sorted(REGION_PHRASES, key=len, reverse=True)) + r")\b")
# Letters NFKD cannot split into base letter + accent (would otherwise be dropped).
_FOLD = str.maketrans({"œ": "oe", "æ": "ae", "ß": "ss", "ø": "o", "ł": "l", "đ": "d", "ð": "d",
                       "þ": "th", "ı": "i"})
_DIGIT_ALPHA = re.compile(r"(?<=\d)(?=[a-z])|(?<=[a-z])(?=\d)")   # "12bis" -> "12 bis", "no852" -> "no 852"

_NON_ALNUM = re.compile(r"[^0-9a-z ]+")
_DIGITS = re.compile(r"\d+")
_WS = re.compile(r"\s+")
_SKEL_SUBS = [("ph", "f"), ("bh", "b"), ("dh", "d"), ("th", "t"), ("gh", "g"), ("kh", "k"),
              ("sh", "s"), ("ch", "c"), ("ck", "k"), ("q", "k"), ("c", "k"), ("z", "s"),
              ("x", "ks"), ("w", "v")]
_VOWELS = re.compile(r"[aeiouy]")
_REPEAT = re.compile(r"(.)\1+")


def _romanize(text, scheme, drop_final_a):
    r = sanscript.transliterate(text, scheme, sanscript.IAST)
    r = "".join(c for c in unicodedata.normalize("NFKD", r) if not unicodedata.combining(c)).lower()
    if drop_final_a:
        r = re.sub(r"(?<=[^aeiou\W\d])a\b", "", r)
    return r


def _base(s: pd.Series) -> pd.Series:
    """Lowercase, native digits -> ASCII, brackets out, Indic romanized, accents folded (é->e,
    œ->oe) for every record; hyphens, apostrophes and other punctuation become spaces, so
    "Saint-Denis" -> "saint denis" and "l'Étoile" -> "l etoile"."""
    s = s.fillna("").astype(str).str.normalize("NFKC").str.lower().str.translate(CLEAN_TABLE)
    has_indic = s.str.contains(INDIC_RE)
    if has_indic.any() and SCRIPTS:
        sub = s[has_indic]
        counts = np.stack([sub.str.count(f"[{chr(a)}-{chr(b)}]").to_numpy() for _, _, a, b, _, _ in SCRIPTS], 1)
        dom = counts.argmax(1)
        out = sub.copy()
        for j, (_, _, _, _, scheme, drop) in enumerate(SCRIPTS):
            m = dom == j
            if m.any():
                uniq = pd.unique(sub[m])
                table = {u: _romanize(u, scheme, drop) for u in uniq}
                out[m] = sub[m].map(table)
        s = s.copy()
        s[has_indic] = out
    s = s.str.translate(_FOLD)
    s = s.str.normalize("NFKD").str.encode("ascii", "ignore").str.decode("ascii")   # é->e, ç->c, every record
    s = s.str.replace(".", "", regex=False)             # l.l.c. -> llc, c.g.h.s -> cghs
    s = s.str.replace("&", " and ", regex=False)
    s = s.str.replace(_NON_ALNUM, " ", regex=True)
    return s.str.replace(_WS, " ", regex=True).str.strip()


def _map_tokens(s: pd.Series, table: dict, drop: set = None) -> pd.Series:
    def f(x):
        toks = [table.get(t, t) for t in x.split()]
        if drop:
            toks = [t for t in toks if t not in drop]
        return " ".join(toks)
    uniq = pd.unique(s)
    return s.map(dict(zip(uniq, map(f, uniq))))


def _skeleton(s: pd.Series) -> pd.Series:
    def f(x):
        out = []
        for t in x.split():
            if t.isdigit():
                out.append(t)
                continue
            for a, b in _SKEL_SUBS:
                t = t.replace(a, b)
            k = _REPEAT.sub(r"\1", t[0] + _VOWELS.sub("", t[1:])) if t else t
            out.append(k)
        return " ".join(out)
    uniq = pd.unique(s)
    return s.map(dict(zip(uniq, map(f, uniq))))


def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalized columns for a frame with business_name / business_address / country."""
    name = _base(df["business_name"])
    name = name.str.replace(_MIXED, lambda m: m.group(0).translate(_OCR), regex=True)
    name = _map_tokens(name, NAME_ABBR)
    core = _map_tokens(name, {}, LEGAL)          # legal forms, connectors and stopwords removed
    core = core.where(core.str.len() > 0, name)
    addr = _base(df["business_address"])
    addr = addr.str.replace(_DIGIT_ALPHA, " ", regex=True)
    uniq = pd.unique(addr)
    addr = addr.map(dict(zip(uniq, (_REGION_RE.sub(lambda m: REGION_PHRASES[m.group(1)], u) for u in uniq))))
    addr = _map_tokens(addr, REGION_TOKENS)
    addr = _map_tokens(addr, ADDR_ABBR, ADDR_DROP)
    nums = addr.str.findall(_DIGITS).str.join(" ")
    num1 = nums.str.split(" ").str[0].fillna("")
    postal = nums.map(lambda x: max((t for t in x.split() if len(t) >= 5), key=len, default=""))
    raw = df["business_name"].fillna("").astype(str)
    script = np.zeros(len(df), np.int8)
    ind = raw.str.contains(INDIC_RE).to_numpy()
    if ind.any():
        counts = np.stack([raw[ind].str.count(f"[{chr(a)}-{chr(b)}]").to_numpy() for _, _, a, b, _, _ in SCRIPTS], 1)
        script[ind] = np.array([s[0] for s in SCRIPTS], np.int8)[counts.argmax(1)]
    return pd.DataFrame({"name": name.to_numpy(), "core": core.to_numpy(), "skel": _skeleton(name).to_numpy(),
                         "addr": addr.to_numpy(), "nums": nums.to_numpy(), "num1": num1.to_numpy(),
                         "postal": postal.to_numpy(), "script": script,
                         "addr_raw_empty": (df["business_address"].fillna("").str.strip() == "").to_numpy()})


def load_normalized(data_dir, work_dir, split, source):
    """Normalized table of one source file (row order = TSV), cached as parquet."""
    path = os.path.join(work_dir, "norm", NORM_VERSION, f"{split}_{source}.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    t = time.time()
    df = read_tsv(source_path(data_dir, split, source))
    out = normalize_frame(df)
    out.insert(0, "entity_id", df["entity_id"].to_numpy())
    out.insert(1, "country", df["country"].to_numpy())
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out.to_parquet(path + ".tmp", index=False)
    os.replace(path + ".tmp", path)
    print(f"normalized {split} {source}: {len(out):,} rows in {time.time() - t:.0f}s")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Build the normalized-text caches of one split.")
    ap.add_argument("--data-dir", default=os.environ.get("DATA_DIR"))
    ap.add_argument("--work-dir", default=os.environ.get("WORK_DIR"))
    ap.add_argument("--split", required=True, choices=["train", "test"])
    a = ap.parse_args()
    for src in ("source1", "source2", "source3"):
        t = load_normalized(a.data_dir, a.work_dir, a.split, src)
        print(t.head(3).to_string(), flush=True)
