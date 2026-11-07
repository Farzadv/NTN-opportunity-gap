#!/usr/bin/env python3
"""
rail_multi.py — generation-aware loading/cleaning for the GB rail campaign.

One campaign, three technologies (2G/4G/5G). Each has the SAME Global_View
skeleton but a different quality metric and a few different columns, so we keep a
small spec per generation and ONE clean loader that reads them all.

Per-generation quality model
----------------------------
  2G : coverage = psch (RxLev, dBm)   quality = c_i  (C/I, dB)  [c_i ~43% present]
  4G : coverage = rsrp (dBm)          quality = sinr (dB)
  5G : coverage = rsrp (dBm)          quality = sinr (dB)

Shared quirks handled in every loader (verified, not assumed):
  * a PHANTOM empty column sits immediately before reading_count in all three
    files (header has N names, data rows have N+1 fields) -> we declare an explicit
    'phantom' name so parsing stays by-name, never position-from-end;
  * operator strings appear as UTF-8 and mojibake -> canonicalised; 'Three UK' and
    'Three' folded; 'Network Rail' (GSM-R) kept as its own label;
  * the `speed` column is empty -> use `gps_speed`;
  * implausible sentinel values (e.g. RSRP 0 or -205) are bounded out;
  * `seq_id` is NOT a unique key -> never used to dedup.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

ROOT = "data"   # place the Global_View_*.csv files here (see README)

# big public operators we report as first-class (others -> "Other")
MAIN_OPERATORS = ["EE", "O2 (Telefónica UK)", "Three", "Vodafone UK"]

# ---- per-generation specification ------------------------------------------
# header = the real header names IN ORDER (reading_count last). We insert a
# 'phantom' just before reading_count to absorb the extra empty field.
GEN = {
    "2g": {
        "path": f"{ROOT}/Global_View_2G.csv",
        "header": ["seq_id","Latitude","Longitude","speed","gps_speed","train","date",
                   "time","mnc","mcc","operator","bcch","dlfreq","cellid","psch","c_i",
                   "reading_count"],
        "cov":  ("psch", -120.0, -10.0),   # RxLev
        "qual": ("c_i",   -10.0, 50.0),    # C/I  (sparsely present)
        "freq": "dlfreq", "cell": "cellid",
        "qual_present": "partial",         # c_i only ~43% populated
    },
    "4g": {
        "path": f"{ROOT}/Global_View_4G.csv",
        "header": ["seq_id","Latitude","Longitude","speed","gps_speed","train","date",
                   "time","mnc","mcc","operator","earfcn","dlfreq","cellid","pci","rsrp",
                   "WB_Rsrp","ptotal","rsrq","WB_Rsrq","WB_Rssi","sinr","nr","reading_count"],
        "cov":  ("rsrp", -141.0, -30.0),
        "qual": ("sinr",  -40.0, 60.0),
        "freq": "dlfreq", "cell": "pci",
        "qual_present": "full",
    },
    "5g": {
        "path": f"{ROOT}/Global_View_5G_1.csv",
        "header": ["seq_id","Latitude","Longitude","speed","gps_speed","train","date",
                   "time","mnc","mcc","operator","nrarfcn","dlfreq","rssi","pci","rsrp",
                   "ssb_idx","rsrq","sinr","reading_count"],
        "cov":  ("rsrp", -160.0, -20.0),
        "qual": ("sinr",  -50.0, 50.0),
        "freq": "dlfreq", "cell": "pci",
        "qual_present": "full",
    },
}


def names_with_phantom(header: list[str]) -> list[str]:
    out = header[:-1] + ["phantom", header[-1]]   # phantom just before reading_count
    return out


def canon_operator(s: pd.Series) -> pd.Series:
    s = s.astype("string").fillna("")
    s = (s.str.replace("Ã³","ó",regex=False).str.replace("Ã©","é",regex=False)
          .str.replace("Ã¡","á",regex=False).str.replace("Ã­","í",regex=False)
          .str.replace("Ã±","ñ",regex=False).str.strip())
    out = s.copy()
    up = s.str.upper()
    out[up.str.startswith("O2")]    = "O2 (Telefónica UK)"
    out[up.str.startswith("THREE")] = "Three"
    out[up.str.startswith("VODAFONE")] = "Vodafone UK"
    out[up.str.startswith("NETWORK RAIL")] = "Network Rail (GSM-R)"
    return out


def band_for(gen: str, freq: pd.Series) -> pd.Series:
    f = freq.astype("float64")
    b = pd.Series("other", index=f.index, dtype="object")
    if gen == "2g":
        b[(f >= 870) & (f <= 960)]   = "GSM900"
        b[(f >= 1710) & (f <= 1880)] = "GSM1800"
    elif gen == "4g":
        b[(f >= 758) & (f <= 803)]   = "B28 (700)"
        b[(f >= 791) & (f <= 821)]   = "B20 (800)"
        b[(f >= 925) & (f <= 960)]   = "B8 (900)"
        b[(f >= 1452) & (f <= 1496)] = "B32 (1400)"
        b[(f >= 1805) & (f <= 1880)] = "B3 (1800)"
        b[(f >= 2110) & (f <= 2170)] = "B1 (2100)"
        b[(f >= 2620) & (f <= 2690)] = "B7 (2600)"
    else:  # 5g
        b[(f >= 758) & (f <= 803)]   = "n28 (700)"
        b[(f >= 2110) & (f <= 2170)] = "n1 (2100)"
        b[(f >= 2620) & (f <= 2690)] = "n7 (2600)"
    return b


def serving_samples(df: pd.DataFrame, keys=("train","date","time","op"),
                    verbose: bool = True) -> pd.DataFrame:
    """Best-cell reduction: per group `keys`, keep the row with the strongest
    COVERAGE metric (`cov`), reading quality (`qual`) off that same row.
    Generation-agnostic because cov/qual were normalised in load_clean().
      keys=(train,date,time,op)        -> operator's strongest cell at an instant
      keys=(train,date,time,op,band)   -> strongest cell per band at an instant
    """
    keys = list(keys)
    df = df.sort_values("cov", kind="stable")          # ascending; keep last = max
    serv = df.drop_duplicates(subset=keys, keep="last").copy()
    if verbose:
        print(f"[serving] {len(df):,} readings -> {len(serv):,} samples (keys={keys})")
    return serv


def load_clean(gen: str, sample: int | None = None, verbose: bool = True):
    """Read + clean one generation's CSV. Returns (df, stats).
    df columns are normalised to: Latitude, Longitude, gps_speed, train, date,
    time, operator, op, dlfreq, band, cell, cov, qual, tsec, reading_count.
    `cov` and `qual` are the generation's coverage/quality metrics (renamed),
    so all downstream code is generation-agnostic."""
    spec = GEN[gen]
    names = names_with_phantom(spec["header"])
    cov_c, cov_lo, cov_hi = spec["cov"]
    qual_c, qual_lo, qual_hi = spec["qual"]
    use = ["Latitude","Longitude","gps_speed","train","date","time","operator",
           spec["freq"], spec["cell"], cov_c, qual_c, "reading_count"]
    use = list(dict.fromkeys(use))  # de-dup if cell==freq etc.

    kw = dict(header=0, names=names, usecols=use, engine="c",
              dtype={"train":"string","date":"string","time":"string",
                     "operator":"string", spec["cell"]:"string"},
              na_values=[""], keep_default_na=True)
    if sample: kw["nrows"] = sample
    if verbose: print(f"[{gen}] reading {spec['path'].split('/')[-1]}"
                      + (f" (first {sample:,})" if sample else ""))
    df = pd.read_csv(spec["path"], **kw)
    n_raw = len(df)

    for c in ["Latitude","Longitude","gps_speed",spec["freq"],cov_c,qual_c]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # core = geo + coverage metric (always present). Quality may be missing.
    before = len(df)
    df = df.dropna(subset=["Latitude","Longitude",cov_c])
    df = df[df.Latitude.between(49,61) & df.Longitude.between(-9,2)
            & df[cov_c].between(cov_lo,cov_hi)]
    n_drop = before - len(df)

    # bound quality to plausible range; out-of-range -> NaN (treated as "unknown")
    df[qual_c] = df[qual_c].where(df[qual_c].between(qual_lo,qual_hi))

    # normalise column names so downstream is generation-agnostic
    df = df.rename(columns={spec["freq"]:"dlfreq", spec["cell"]:"cell",
                            cov_c:"cov", qual_c:"qual"})
    df["operator"] = canon_operator(df["operator"])
    df["op"] = df["operator"].where(df["operator"].isin(MAIN_OPERATORS), "Other")
    df["band"] = band_for(gen, df["dlfreq"])
    t = df["time"].str.split(":", expand=True)
    df["tsec"] = (pd.to_numeric(t[0],errors="coerce")*3600
                  + pd.to_numeric(t[1],errors="coerce")*60
                  + pd.to_numeric(t[2],errors="coerce"))

    stats = {
        "gen": gen, "rows_raw": int(n_raw), "rows_dropped": int(n_drop),
        "rows_clean": int(len(df)),
        "qual_present_pct": round(100*df["qual"].notna().mean(), 1),
        "cov_metric": cov_c, "qual_metric": qual_c,
    }
    if verbose:
        print(f"[{gen}] raw={n_raw:,} dropped={n_drop:,} clean={len(df):,} "
              f"| {qual_c} present {stats['qual_present_pct']}%")
    return df, stats
