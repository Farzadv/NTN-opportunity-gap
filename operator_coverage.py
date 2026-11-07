#!/usr/bin/env python3
"""
operator_coverage.py — per-operator vs multi-operator (best-of-N) covered share of
the route, by generation. Covered = that operator's best-of-N signal >= tau at a
measured instant; an instant where the operator is not detected counts as uncovered
for it (a single-operator subscriber has no service there). Best-of-N = any operator
>= tau. Denominator = total within-run track distance (same as the coverage budget).
Outputs -> results/figures/fig_operator_coverage.* , results/tables/operator_coverage.csv
"""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); matplotlib.rcParams["pdf.fonttype"]=42; matplotlib.rcParams["ps.fonttype"]=42
import matplotlib.pyplot as plt
import rail_multi as rm

FIG, TBL = "results/figures", "results/tables"
os.makedirs(FIG, exist_ok=True); os.makedirs(TBL, exist_ok=True)
TAU={"2g":-95,"4g":-110,"5g":-110}; NAME={"2g":"2G (GSM)","4g":"4G (LTE)","5g":"5G (NR)"}
OPS=["EE","O2 (Telefónica UK)","Three","Vodafone UK"]
# Okabe-Ito palette (high-contrast, colourblind-safe)
OP_COL={"EE":"#E69F00","O2":"#0072B2","Three":"#CC79A7","Vodafone":"#D55E00","Best-of-N":"#009E73"}
# hatch textures so bars stay distinguishable in a black-and-white print
OP_HATCH={"EE":"///","O2":"\\\\","Three":"xxx","Vodafone":"...","Best-of-N":""}
plt.rcParams.update({"figure.dpi":130,"font.size":14,"axes.grid":True,"grid.alpha":.25,
                     "axes.axisbelow":True,"hatch.linewidth":0.6})

def hav(a1,o1,a2,o2):
    R=6371.0;p1,p2=np.radians(a1),np.radians(a2);d1=np.radians(a2-a1);d2=np.radians(o2-o1)
    h=np.sin(d1/2)**2+np.cos(p1)*np.cos(p2)*np.sin(d2/2)**2;return 2*R*np.arcsin(np.sqrt(h))

def coverage(gen):
    tau=TAU[gen]
    s=pd.read_parquet(f"cache/serving_{gen}.parquet").rename(columns={"cov":"rxp"})
    s=s[s.op.isin(OPS)]
    piv=s.groupby(["train","date","time","op"]).rxp.max().unstack("op")
    meta=s.groupby(["train","date","time"]).agg(lat=("Latitude","first"),lon=("Longitude","first"),tsec=("tsec","first"))
    df=meta.join(piv).reset_index().sort_values(["train","date","tsec"]).reset_index(drop=True)
    tr=df.train.to_numpy();da=df.date.to_numpy();ts=df.tsec.to_numpy(float)
    la=df.lat.to_numpy(float);lo=df.lon.to_numpy(float)
    step=np.zeros(len(df));dt=np.zeros(len(df))
    step[1:]=hav(la[:-1],lo[:-1],la[1:],lo[1:]); dt[1:]=ts[1:]-ts[:-1]
    within=np.zeros(len(df),bool); within[1:]=(tr[1:]==tr[:-1])&(da[1:]==da[:-1])&(dt[1:]>=0)&(dt[1:]<=600)&(step[1:]<=3)
    black=within&(dt>12)&(step>0.15); meas=within&~black
    tot=float(step[within].sum())
    present=[o for o in OPS if o in df.columns and int(s.op.eq(o).sum())>5000]
    out={}; anycov=np.zeros(len(df),bool)
    for o in present:
        cov=np.nan_to_num(df[o].to_numpy(float),nan=-999)>=tau
        anycov|=cov; out[o.split()[0]]=100*float(step[meas&cov].sum())/tot
    out["Best-of-N"]=100*float(step[meas&anycov].sum())/tot
    return out

rows=[]
for gen in ["2g","4g","5g"]:
    c=coverage(gen); c["gen"]=NAME[gen]; rows.append(c)
df=pd.DataFrame(rows).set_index("gen")
order=[c for c in ["EE","O2","Three","Vodafone","Best-of-N"] if c in df.columns]
df=df[order]; df.round(1).to_csv(f"{TBL}/operator_coverage.csv")
print(df.round(1).to_string())

# figure: one group per generation, bars = operators + Best-of-N
fig,ax=plt.subplots(figsize=(8,3.4)); gens=df.index.tolist(); x=np.arange(len(gens))
n=len(order); w=0.8/n
for j,col in enumerate(order):
    vals=df[col].values
    ax.bar(x+(j-(n-1)/2)*w, vals, w, label=col, color=OP_COL.get(col,"#888"),
           hatch=OP_HATCH.get(col,""), edgecolor="black",
           linewidth=1.3 if col=="Best-of-N" else 0.5)
ax.set_xticks(x); ax.set_xticklabels(gens); ax.set_ylim(0,100)
ax.set_ylabel("route covered (%)",fontsize=16)
ax.legend(ncol=n,fontsize=12,loc="lower left",bbox_to_anchor=(0,1.02,1,0.12),
          mode="expand",borderaxespad=0,frameon=False,columnspacing=1.0,handletextpad=0.5)
fig.tight_layout()
for e in ("pdf","png"): fig.savefig(f"{FIG}/fig_operator_coverage.{e}",bbox_inches="tight")
print(f"[fig] {FIG}/fig_operator_coverage.png")
