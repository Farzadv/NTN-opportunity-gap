#!/usr/bin/env python3
"""3-panel joint coverage-quality density (2G/4G/5G) for the paper."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); matplotlib.rcParams["pdf.fonttype"]=42; matplotlib.rcParams["ps.fonttype"]=42; import matplotlib.pyplot as plt
import rail_multi as rm
THR={"2g":(-95,9,"RxLev (dBm)","C/I (dB)","2G (GSM)"),
     "4g":(-110,0,"RSRP (dBm)","SINR (dB)","4G (LTE)"),
     "5g":(-110,0,"RSRP (dBm)","SINR (dB)","5G (NR)")}
def so(g):
    s=pd.read_parquet(f"cache/serving_{g}.parquet")
    s=s.sort_values("cov",kind="stable").drop_duplicates(["train","date","time","op"],keep="last")
    return s.rename(columns={"cov":"rxp"})
fig,axs=plt.subplots(1,3,figsize=(12,3.3),gridspec_kw={"wspace":0.32})
for ax,g in zip(axs,["2g","4g","5g"]):
    tau,sig,cx,qy,nm=THR[g]; s=so(g); d=s[s["qual"].notna()]
    hb=ax.hexbin(d.rxp,d["qual"],gridsize=60,bins="log",cmap="viridis",mincnt=1)
    hb.set_rasterized(True)   # vector-PDF backend drops hexbin; rasterize this layer
    ax.axvline(tau,color="w",ls="--",lw=1.1); ax.axhline(sig,color="w",ls="--",lw=1.1)
    ax.set_xlabel(cx,fontsize=13); ax.set_ylabel(qy,fontsize=13,labelpad=1); ax.set_title(nm,fontsize=14)
    ax.tick_params(labelsize=11)
cb=fig.colorbar(hb,ax=axs,shrink=.85)
cb.set_label("log$_{10}$ count",fontsize=12); cb.ax.tick_params(labelsize=11)
for e in ("pdf","png"): fig.savefig(f"results/figures/fig_density_all.{e}",bbox_inches="tight",dpi=200)
print("wrote results/figures/fig_density_all.png")
