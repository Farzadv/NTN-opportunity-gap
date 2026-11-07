#!/usr/bin/env python3
"""
atlas.py — Step 8 maps. Geographic interference-limited fraction per ~2km cell,
one panel per generation, on a shared scale. Shows interference deficiency is
widespread (not just urban) and worsens 2G -> 5G.  Outputs -> results/figures.
"""
import os, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); matplotlib.rcParams["pdf.fonttype"]=42; matplotlib.rcParams["ps.fonttype"]=42
import matplotlib.pyplot as plt
import rail_multi as rm

FIG="results/figures"; os.makedirs(FIG,exist_ok=True)
DEG=0.02
THR={"2g":(-95,9),"4g":(-110,0),"5g":(-110,0)}
NAME={"2g":"2G (GSM)","4g":"4G (LTE)","5g":"5G (NR)"}

def uk_polys(path="data/uk_outline.geojson"):
    """Exterior rings of the UK outline, for a faint map backdrop."""
    if not os.path.exists(path): return []
    d=json.load(open(path)); rings=[]
    for f in d.get("features",[d]):
        g=f["geometry"]; polys=g["coordinates"] if g["type"]=="MultiPolygon" else [g["coordinates"]]
        for poly in polys: rings.append(np.asarray(poly[0],float))   # exterior ring
    return rings

def serv_op(gen):
    s=pd.read_parquet(f"cache/serving_{gen}.parquet")
    s=(s.sort_values("cov",kind="stable")
        .drop_duplicates(subset=["train","date","time","op"],keep="last"))
    return s.rename(columns={"cov":"rxp"})

def cells(gen):
    tau,sig=THR[gen]; s=serv_op(gen)
    s=s[s.op.isin(rm.MAIN_OPERATORS)].copy()
    cov=s.rxp<tau; known=s["qual"].notna()
    s["intf"]=((~cov)&known&(s["qual"]<sig)).astype(float)
    s["bx"]=np.floor(s.Latitude/DEG)*DEG+DEG/2
    s["by"]=np.floor(s.Longitude/DEG)*DEG+DEG/2
    g=s.groupby(["bx","by"],observed=True).agg(intf=("intf","mean"),n=("rxp","size")).reset_index()
    return g[g.n>=5]

# Single-column layout: three mini-maps + shared colorbar, fonts sized for
# printing at \linewidth in one IEEE column.
RINGS=uk_polys()
fig,axs=plt.subplots(1,3,figsize=(7.2,3.4),sharex=True,sharey=True,
                     gridspec_kw={"wspace":0.06})
aspect=1/np.cos(np.radians(54))
xmn=ymn=1e9; xmx=ymx=-1e9
for ax,gen in zip(axs,["2g","4g","5g"]):
    g=cells(gen)
    for r in RINGS:                       # faint UK landmass behind the points
        ax.fill(r[:,0],r[:,1],facecolor="#eef2f6",edgecolor="#c3ccd6",linewidth=0.3,zorder=0)
    sc=ax.scatter(g.by,g.bx,c=g.intf,s=2.4,cmap="magma",vmin=0,vmax=1,linewidths=0,zorder=2)
    ax.set_title(f"{NAME[gen].split()[0]} ({len(g):,})",fontsize=15); ax.set_aspect(aspect)
    ax.tick_params(labelsize=12); ax.grid(True,alpha=.2)
    ax.set_xticks([-5,-2.5,0]); ax.set_yticks([51,53,55,57])
    xmn=min(xmn,g.by.min()); xmx=max(xmx,g.by.max()); ymn=min(ymn,g.bx.min()); ymx=max(ymx,g.bx.max())
m=0.35; axs[0].set_xlim(xmn-m,xmx+m); axs[0].set_ylim(ymn-m,ymx+m)   # shared axes -> clips to GB rail extent
cb=fig.colorbar(sc,ax=axs,shrink=.85,pad=0.02)
cb.set_label("interference-limited fraction",fontsize=13); cb.ax.tick_params(labelsize=12)
for e in ("pdf","png"): fig.savefig(f"{FIG}/fig_atlas_interference.{e}",bbox_inches="tight")
print(f"[fig] {FIG}/fig_atlas_interference.png")
