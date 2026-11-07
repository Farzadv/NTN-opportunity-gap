#!/usr/bin/env python3
"""
build_serving.py — Step 3. Full clean + best-cell reduction for every generation,
cached to cache/serving_<gen>.parquet (op,band granularity) + a stats JSON.
Later steps read the cache, so the 36M-row scan happens once.
"""
import os, json, sys
import pandas as pd
import rail_multi as rm

OUT = "cache"; os.makedirs(OUT, exist_ok=True)

def build(gen):
    df, st = rm.load_clean(gen)                       # full file
    st["readings_by_op"]   = {k:int(v) for k,v in df.op.value_counts().items()}
    st["readings_by_band"] = {k:int(v) for k,v in df.band.value_counts().items()}
    st["trains"] = sorted(df.train.dropna().unique().tolist())
    keep = ["Latitude","Longitude","gps_speed","train","date","time","tsec",
            "operator","op","band","dlfreq","cell","cov","qual","reading_count"]
    serv = rm.serving_samples(df, keys=("train","date","time","op","band"))[keep]
    serv_op = (serv.sort_values("cov", kind="stable")
                   .drop_duplicates(subset=["train","date","time","op"], keep="last"))
    st["serving_opband"] = int(len(serv))
    st["serving_op"]     = int(len(serv_op))
    serv.to_parquet(f"{OUT}/serving_{gen}.parquet", index=False)
    json.dump(st, open(f"{OUT}/serving_{gen}_meta.json","w"), ensure_ascii=False, indent=1)
    print(f"[{gen}] cached {len(serv):,} (op,band) serving samples "
          f"-> {len(serv_op):,} per-operator | qual present {st['qual_present_pct']}%\n")

if __name__ == "__main__":
    gens = sys.argv[1:] or ["2g","4g","5g"]
    for g in gens:
        build(g)
