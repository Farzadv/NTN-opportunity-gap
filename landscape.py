#!/usr/bin/env python3
"""
landscape.py — Step 4. Per-generation coverage & quality landscape from the
serving caches: CDFs of the coverage metric (RxLev/RSRP) and quality metric
(C-I/SINR) per operator and per band, plus a coverage-probability table.

Outputs -> results/figures/ , results/tables/
"""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); matplotlib.rcParams["pdf.fonttype"]=42; matplotlib.rcParams["ps.fonttype"]=42
import matplotlib.pyplot as plt
import rail_multi as rm

FIG, TBL = "results/figures", "results/tables"
os.makedirs(FIG, exist_ok=True); os.makedirs(TBL, exist_ok=True)

GENS = {
    "2g": {"name":"2G (GSM)",  "cov":"RxLev (dBm)", "qual":"C/I (dB)",  "cov_tau":-100},
    "4g": {"name":"4G (LTE)",  "cov":"RSRP (dBm)",  "qual":"SINR (dB)", "cov_tau":-110},
    "5g": {"name":"5G (NR)",   "cov":"RSRP (dBm)",  "qual":"SINR (dB)", "cov_tau":-110},
}
# Okabe-Ito palette: high-contrast, well-separated hues, colourblind-safe
OP_COL = {"O2 (Telefónica UK)":"#0072B2", "EE":"#E69F00", "Three":"#CC79A7",
          "Vodafone UK":"#D55E00", "Network Rail (GSM-R)":"#56B4E9", "Other":"#999999"}
# line styles so operator curves stay distinguishable in a black-and-white print
OP_LS = {"O2 (Telefónica UK)":"-", "EE":"--", "Three":"-.", "Vodafone UK":(0,(1,1)),
         "Network Rail (GSM-R)":(0,(3,1,1,1)), "Other":":"}
plt.rcParams.update({"figure.dpi":130, "font.size":15, "axes.grid":True,
                     "grid.alpha":.25, "axes.axisbelow":True})

def serv_op(gen):
    s = pd.read_parquet(f"cache/serving_{gen}.parquet")
    s = (s.sort_values("cov", kind="stable")
          .drop_duplicates(subset=["train","date","time","op"], keep="last"))
    return s.rename(columns={"cov":"rxp"})   # 'cov' clashes with DataFrame.cov()

def cdf(ax, data, label, color, ls="-"):
    d = np.sort(np.asarray(data, float)); d = d[~np.isnan(d)]
    if not len(d): return
    ax.plot(d, np.arange(1,len(d)+1)/len(d), label=label, color=color, ls=ls, lw=1.7)

def ops_present(s, nmin=2000):
    return [o for o in rm.MAIN_OPERATORS + ["Network Rail (GSM-R)"]
            if (s.op==o).sum() >= nmin]

def pct(x,q): x=np.asarray(x,float); x=x[~np.isnan(x)]; return float(np.percentile(x,q)) if len(x) else np.nan

def run_gen(gen):
    cfg = GENS[gen]; s = serv_op(gen)
    ops = ops_present(s)
    # ---- figure: coverage + quality CDFs by operator ----
    fig,(a1,a2)=plt.subplots(1,2,figsize=(9,3.05))
    for o in ops:
        g=s[s.op==o]
        cdf(a1, g.rxp, o.split(" ")[0], OP_COL.get(o), OP_LS.get(o,"-"))
        cdf(a2, g["qual"], o.split(" ")[0], OP_COL.get(o), OP_LS.get(o,"-"))
    a1.axvline(cfg["cov_tau"], color="k", ls="--", lw=1, alpha=.6)
    a1.set_xlabel(cfg["cov"]); a1.set_ylabel("CDF")
    a2.set_xlabel(cfg["qual"]); a2.legend(fontsize=12)
    fig.tight_layout(); save(fig,f"fig_landscape_{gen}")
    # ---- figure: coverage CDF by band ----
    fig,ax=plt.subplots(figsize=(5.6,3.6))
    bands=[b for b in s.band.value_counts().index if b!="other" and (s.band==b).sum()>=2000]
    for b in bands: cdf(ax, s[s.band==b].rxp, b, None)
    ax.axvline(cfg["cov_tau"], color="k", ls="--", lw=1, alpha=.6)
    ax.set_xlabel(cfg["cov"]); ax.set_ylabel("CDF"); ax.set_title(f"Coverage by band — {cfg['name']}")
    ax.legend(fontsize=12); fig.tight_layout(); save(fig,f"fig_bands_{gen}")
    # ---- table ----
    rows=[]
    def blk(g,o,b):
        if not len(g): return
        rows.append({"gen":gen,"operator":o,"band":b,"n":len(g),
            "cov_p10":pct(g.rxp,10),"cov_p50":pct(g.rxp,50),"cov_p90":pct(g.rxp,90),
            "qual_present_pct":round(100*g["qual"].notna().mean(),1),
            "qual_p10":pct(g["qual"],10),"qual_p50":pct(g["qual"],50),"qual_p90":pct(g["qual"],90),
            "cov_prob": float((g.rxp>cfg["cov_tau"]).mean())})
    for o in ops: blk(s[s.op==o],o,"ALL")
    for (o,b),g in s.groupby(["op","band"],observed=True):
        if len(g)>=2000: blk(g,o,b)
    pd.DataFrame(rows).round(2).to_csv(f"{TBL}/landscape_{gen}.csv",index=False)
    print(f"[{gen}] {len(s):,} serving | ops={[o.split(' ')[0] for o in ops]} | "
          f"cov_p50={pct(s.rxp,50):.1f} qual_p50={pct(s.qual,50):.1f}")
    return s

def save(fig,name):
    for e in ("pdf","png"): fig.savefig(f"{FIG}/{name}.{e}",bbox_inches="tight")
    plt.close(fig); print(f"[fig] {FIG}/{name}.png")

if __name__=="__main__":
    for g in ["2g","4g","5g"]: run_gen(g)
