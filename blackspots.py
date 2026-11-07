#!/usr/bin/env python3
"""
blackspots.py — km-based coverage budget per generation (replaces the bin-based UOF).

One consistent ruler: distance along the train's path. Every metre of track is in
exactly ONE of three buckets, so they sum to 100% with no overlap:
  covered    : best operator (best-of-N) >= tau
  weak       : best operator measured but < tau          (the coverage hole)
  blackspot  : train moving but nothing logged (gap)     (no data; includes tunnels)

"weak" is reported two ways:
  weak_instant   : any segment where best-of-N < tau (counts brief fades too)
  weak_sustained : only contiguous below-tau holes >= MIN_HOLE_KM (NTN-relevant)

Outputs -> results/tables/coverage_budget.csv , results/figures/fig_coverage_budget.*
NOTE: blackspot includes tunnels and any logging gaps; it is an UPPER bound on the
NTN-addressable unmeasured hole (a satellite cannot serve a train in a tunnel).
"""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); matplotlib.rcParams["pdf.fonttype"]=42; matplotlib.rcParams["ps.fonttype"]=42
import matplotlib.pyplot as plt
import rail_multi as rm

FIG, TBL = "results/figures", "results/tables"
os.makedirs(FIG, exist_ok=True); os.makedirs(TBL, exist_ok=True)
GAP_S, RUN_S, JUMP_KM, MIN_HOLE_KM, MIN_GAP_KM = 12.0, 600.0, 3.0, 0.20, 0.15
GENS = {"2g":{"name":"2G (GSM)","tau":-95}, "4g":{"name":"4G (LTE)","tau":-110},
        "5g":{"name":"5G (NR)","tau":-110}}
# Okabe-Ito palette (high-contrast, colourblind-safe): green=good, blue=weak, red=blackspot
COL = {"covered":"#009E73","weak":"#0072B2","blackspot":"#D55E00"}
# hatch textures so bars stay distinguishable in a black-and-white print
HATCH = {"covered":"","weak":"xxx","blackspot":"///"}
plt.rcParams.update({"figure.dpi":130,"font.size":15,"axes.grid":True,"grid.alpha":.25,
                     "axes.axisbelow":True,"hatch.linewidth":0.6})

def hav(a1,o1,a2,o2):
    R=6371.0;p1,p2=np.radians(a1),np.radians(a2);d1=np.radians(a2-a1);d2=np.radians(o2-o1)
    h=np.sin(d1/2)**2+np.cos(p1)*np.cos(p2)*np.sin(d2/2)**2;return 2*R*np.arcsin(np.sqrt(h))

def budget(gen, tau):
    s=pd.read_parquet(f"cache/serving_{gen}.parquet").rename(columns={"cov":"rxp"})
    s=s[s.op.isin(rm.MAIN_OPERATORS)]
    # best-of-N (any operator) per measurement instant
    inst=(s.groupby(["train","date","time"])
            .agg(lat=("Latitude","first"),lon=("Longitude","first"),
                 tsec=("tsec","first"),best=("rxp","max"))
            .reset_index().sort_values(["train","date","tsec"]).reset_index(drop=True))
    lat=inst.lat.to_numpy(float); lon=inst.lon.to_numpy(float); tsec=inst.tsec.to_numpy(float)
    best=inst.best.to_numpy(float); tr=inst.train.to_numpy(); da=inst.date.to_numpy()
    same=np.empty(len(inst),bool); same[0]=False
    same[1:]=(tr[1:]==tr[:-1])&(da[1:]==da[:-1])
    dt=np.empty(len(inst)); dt[0]=np.nan; dt[1:]=tsec[1:]-tsec[:-1]
    dd=np.empty(len(inst)); dd[0]=np.nan; dd[1:]=hav(lat[:-1],lon[:-1],lat[1:],lon[1:])
    within=same&(dt>=0)&(dt<=RUN_S)&(dd<=JUMP_KM)              # part of a continuous run
    black=within&(dt>GAP_S)&(dd>MIN_GAP_KM)                    # gap: >12s AND moved >150m
    measured=within&~black                                     # everything else is measured
    bad=best<tau                                               # endpoint best-of-N below tau

    cov_km   = float(np.nansum(dd[measured & ~bad]))
    weak_i   = float(np.nansum(dd[measured &  bad]))
    black_km = float(np.nansum(dd[black]))
    # sustained holes: contiguous measured below-tau runs >= MIN_HOLE_KM
    weak_s=0.0; cur=0.0
    for i in range(len(inst)):
        if measured[i] and bad[i]:
            cur+=dd[i]
        else:
            if cur>=MIN_HOLE_KM: weak_s+=cur
            cur=0.0
    if cur>=MIN_HOLE_KM: weak_s+=cur
    tot=cov_km+weak_i+black_km
    return {"gen":gen,"name":GENS[gen]["name"],"tau":tau,"total_km":round(tot,0),
            "covered_pct":round(100*cov_km/tot,1),
            "weak_instant_pct":round(100*weak_i/tot,1),
            "weak_sustained_pct":round(100*weak_s/tot,1),
            "blackspot_pct":round(100*black_km/tot,1),
            "uncovered_instant_pct":round(100*(weak_i+black_km)/tot,1),
            "uncovered_sustained_pct":round(100*(weak_s+black_km)/tot,1)}

rows=[budget(g,GENS[g]["tau"]) for g in GENS]
df=pd.DataFrame(rows); df.to_csv(f"{TBL}/coverage_budget.csv",index=False)
print(df.to_string(index=False))

# stacked bar: covered / weak (instant) / blackspot per generation
fig,ax=plt.subplots(figsize=(7.6,3.7)); x=np.arange(len(df)); bottom=np.zeros(len(df))
cen={}
for k,lab in [("covered_pct","covered"),("weak_instant_pct","weak (below threshold)"),
              ("blackspot_pct","blackspot (no data)")]:
    key=lab.split()[0]; col=COL[key]
    ax.bar(x,df[k],bottom=bottom,label=lab,color=col,width=.55,
           hatch=HATCH[key],edgecolor="black",linewidth=0.5)
    for i,v in enumerate(df[k]): cen[(i,k)]=(bottom[i]+v/2,v)
    bottom+=df[k].values
# covered: label centred inside the bar
for i in range(len(df)):
    cy,v=cen[(i,"covered_pct")]; ax.text(x[i],cy,f"{v:.0f}%",ha="center",va="center",fontsize=13)
# weak + blackspot: thin top slices -> label to the RIGHT of the bar, vertically spread
MINGAP=9.0
for i in range(len(df)):
    keys=["weak_instant_pct","blackspot_pct"]
    ys=[cen[(i,k)][0] for k in keys]
    if ys[1]-ys[0]<MINGAP:                       # spread apart if too close
        mid=(ys[0]+ys[1])/2; ys=[mid-MINGAP/2,mid+MINGAP/2]
    for k,ly in zip(keys,ys):
        cy,v=cen[(i,k)]
        ax.annotate(f"{v:.0f}%",xy=(x[i]+0.265,cy),xytext=(x[i]+0.33,ly),
                    ha="left",va="center",fontsize=12,
                    arrowprops=dict(arrowstyle="-",lw=0.7,color="0.45"))
ax.set_xticks(x); ax.set_xticklabels(df.name); ax.set_ylim(0,106)
ax.set_xlim(-0.6,len(df)-0.2)
ax.set_ylabel("share of route length (%)")
ax.legend(ncol=3,fontsize=11,loc="lower left",bbox_to_anchor=(0,1.02,1,0.12),
          mode="expand",borderaxespad=0,frameon=False,columnspacing=1.0,handletextpad=0.5)
fig.tight_layout()
for e in ("pdf","png"): fig.savefig(f"{FIG}/fig_coverage_budget.{e}",bbox_inches="tight")
print(f"\n[fig] {FIG}/fig_coverage_budget.png")
