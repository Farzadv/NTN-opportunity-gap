#!/usr/bin/env python3
"""
make_deficiency_maps.py — derived railway deficiency maps (the released product).

For every ~2 km cell of surveyed track and every generation, summarise how the
terrestrial network behaves there, using exactly the definitions of the paper:

  regime per serving sample, given coverage threshold tau and quality sigma
    coverage-limited     : cov <  tau
    interference-limited : cov >= tau and qual present and qual <  sigma
    acceptable           : cov >= tau and qual present and qual >= sigma
    unknown              : cov >= tau and qual missing           (mainly 2G)
  thresholds: 2G (-95 dBm RxLev, 9 dB C/I), 4G and 5G (-110 dBm RSRP, 0 dB SINR)

Cells are 0.02 deg (about 2.2 km in latitude, 1.3 km in longitude at 54 N),
the same grid used for the map figure. Cells with fewer than 5 samples are
dropped, as in the paper.

Two views per cell:
  per-operator pooled : regime fractions over all main operators
  best-of-N           : the strongest operator present at each instant, which
                        is what an ideal multi-operator gateway would see
                        (median RSRP, and the share of passes in outage)

Outputs, written to maps/:
  deficiency_map.csv          one row per (generation, cell)
  deficiency_map_<gen>.geojson   same content as point features, for GIS tools

Requires cache/serving_<gen>.parquet, produced by build_serving.py.
"""
import os, json
import numpy as np, pandas as pd
import rail_multi as rm

OUT = "maps"; os.makedirs(OUT, exist_ok=True)
DEG = 0.02
MIN_SAMPLES = 5
THR = {"2g": (-95.0, 9.0), "4g": (-110.0, 0.0), "5g": (-110.0, 0.0)}
COVM = {"2g": "RxLev", "4g": "RSRP", "5g": "RSRP"}
QUALM = {"2g": "C/I", "4g": "SINR", "5g": "SINR"}


def serving(gen):
    """One row per (train, date, time, operator): that operator's best cell."""
    s = pd.read_parquet(f"cache/serving_{gen}.parquet")
    s = (s.sort_values("cov", kind="stable")
           .drop_duplicates(subset=["train", "date", "time", "op"], keep="last"))
    return s[s.op.isin(rm.MAIN_OPERATORS)].rename(columns={"cov": "rxp"})


def cell_table(gen):
    tau, sigma = THR[gen]
    s = serving(gen).copy()
    s["bx"] = np.floor(s.Latitude / DEG) * DEG + DEG / 2      # cell centre lat
    s["by"] = np.floor(s.Longitude / DEG) * DEG + DEG / 2     # cell centre lon

    cov_lim = s.rxp < tau
    known = s["qual"].notna()
    s["cov_lim"] = cov_lim.astype(float)
    s["int_lim"] = ((~cov_lim) & known & (s["qual"] < sigma)).astype(float)
    s["accept"] = ((~cov_lim) & known & (s["qual"] >= sigma)).astype(float)
    s["unknown"] = ((~cov_lim) & (~known)).astype(float)

    g = (s.groupby(["bx", "by"], observed=True)
           .agg(n_samples=("rxp", "size"),
                n_operators=("op", "nunique"),
                coverage_limited_frac=("cov_lim", "mean"),
                interference_limited_frac=("int_lim", "mean"),
                acceptable_frac=("accept", "mean"),
                unknown_frac=("unknown", "mean"),
                median_rsrp_dbm=("rxp", "median"))
           .reset_index())

    # best-of-N: strongest operator at each measurement instant
    inst = (s.groupby(["train", "date", "time"], observed=True)
              .agg(bx=("bx", "first"), by=("by", "first"), best=("rxp", "max"))
              .reset_index())
    inst["bad"] = (inst.best < tau).astype(int)
    # per (cell, pass): is the cell in outage on that journey?
    bp = (inst.groupby(["bx", "by", "train", "date"], observed=True)
              .agg(best=("best", "median"), bad=("bad", "mean")).reset_index())
    bp["pass_bad"] = (bp.bad > 0.5).astype(int)
    b = (bp.groupby(["bx", "by"], observed=True)
           .agg(n_passes=("pass_bad", "size"),
                best_of_n_median_rsrp_dbm=("best", "median"),
                passes_in_outage_frac=("pass_bad", "mean")).reset_index())

    g = g.merge(b, on=["bx", "by"], how="left")
    g = g[g.n_samples >= MIN_SAMPLES].copy()

    # persistence class, only where the cell was visited on >= 2 journeys
    def pclass(row):
        if not np.isfinite(row.n_passes) or row.n_passes < 2:
            return "unclassified"
        f = row.passes_in_outage_frac
        if f == 0: return "always_good"
        if f == 1: return "always_bad"
        return "sometimes"
    g["bestofn_persistence_class"] = g.apply(pclass, axis=1)


    # per-operator persistence, the definition used for Fig. 7 of the paper:
    # for each (operator, cell) visited on at least two journeys, is the cell in
    # outage on every pass, some passes, or none?
    pp = (s.groupby(["op", "bx", "by", "train", "date"], observed=True)
            .rxp.median().reset_index())
    pp["out"] = (pp.rxp < tau).astype(int)
    ob = (pp.groupby(["op", "bx", "by"], observed=True)
            .agg(n=("out", "size"), of=("out", "mean")).reset_index())
    ob = ob[ob.n >= 2]
    ob["always_good"] = (ob.of == 0).astype(float)
    ob["always_bad"] = (ob.of == 1).astype(float)
    ob["sometimes"] = ((ob.of > 0) & (ob.of < 1)).astype(float)
    pc = (ob.groupby(["bx", "by"], observed=True)
            .agg(per_operator_bins=("n", "size"),
                 per_operator_always_good_frac=("always_good", "mean"),
                 per_operator_sometimes_frac=("sometimes", "mean"),
                 per_operator_always_bad_frac=("always_bad", "mean")).reset_index())
    g = g.merge(pc, on=["bx", "by"], how="left")

    g.insert(0, "generation", gen.upper())
    g = g.rename(columns={"bx": "lat", "by": "lon"})
    g["coverage_metric"] = COVM[gen]
    g["quality_metric"] = QUALM[gen]
    g["tau_dbm"] = tau
    g["sigma_db"] = sigma
    num = ["coverage_limited_frac", "interference_limited_frac", "acceptable_frac",
           "unknown_frac", "passes_in_outage_frac",
           "per_operator_always_good_frac", "per_operator_sometimes_frac",
           "per_operator_always_bad_frac"]
    g[num] = g[num].round(4)
    for c in ("median_rsrp_dbm", "best_of_n_median_rsrp_dbm"):
        g[c] = g[c].round(1)
    g[["lat", "lon"]] = g[["lat", "lon"]].round(4)
    return g


def to_geojson(df, path):
    feats = []
    for r in df.itertuples(index=False):
        d = {k: (None if (isinstance(v, float) and not np.isfinite(v)) else v)
             for k, v in r._asdict().items() if k not in ("lat", "lon")}
        feats.append({"type": "Feature",
                      "geometry": {"type": "Point", "coordinates": [r.lon, r.lat]},
                      "properties": d})
    with open(path, "w") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f)


if __name__ == "__main__":
    parts = []
    for gen in ("2g", "4g", "5g"):
        g = cell_table(gen)
        parts.append(g)
        to_geojson(g, f"{OUT}/deficiency_map_{gen}.geojson")
        print(f"[{gen}] {len(g):,} cells | "
              f"cov-lim {g.coverage_limited_frac.mean():.3f} "
              f"int-lim {g.interference_limited_frac.mean():.3f} "
              f"accept {g.acceptable_frac.mean():.3f} | "
              f"always_bad: best-of-N cells {100*(g.bestofn_persistence_class=='always_bad').mean():.1f}%, "
              f"per-operator bins {100*np.average(g.per_operator_always_bad_frac.fillna(0), weights=g.per_operator_bins.fillna(0)+1e-9):.1f}%")
    allg = pd.concat(parts, ignore_index=True)
    allg.to_csv(f"{OUT}/deficiency_map.csv", index=False)
    print(f"\nwrote {OUT}/deficiency_map.csv  ({len(allg):,} rows)")
    print("columns:", ", ".join(allg.columns))
