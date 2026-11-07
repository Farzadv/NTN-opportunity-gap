#!/usr/bin/env python3
"""
mobility.py — Step 7. Ordered-run features per generation, from time-sorted
per-(train,date,operator) tracks (serving caches).

  * handover proxy : raw best-server cell switches/km, confirmed (debounced,
    >=3-sample dwell) handovers/km, distinct cells/km
  * coverage holes : contiguous below-tau stretches -> length CDF, holes/100km,
    fraction of route in outage
  * persistence    : across repeated passes of the same 2km bin, is a bad spot
    bad every pass? (bimodal => deficiency is position-deterministic)

Segments break on time gap >600s OR spatial jump >3km. Outputs -> results/.
NOTE: 2G cellid is frequently blank, so 2G handover numbers are a weak proxy.
"""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); matplotlib.rcParams["pdf.fonttype"]=42; matplotlib.rcParams["ps.fonttype"]=42
import matplotlib.pyplot as plt
import rail_multi as rm

FIG, TBL = "results/figures", "results/tables"
os.makedirs(FIG, exist_ok=True); os.makedirs(TBL, exist_ok=True)
GAP_S, JUMP_KM, MIN_DWELL = 600.0, 3.0, 3
GENS = {"2g":{"name":"2G","tau":-95}, "4g":{"name":"4G","tau":-110}, "5g":{"name":"5G","tau":-110}}
# Okabe-Ito palette (high-contrast, colourblind-safe); gens run cool->warm as they worsen
OP_COL = {"O2 (Telefónica UK)":"#0072B2","EE":"#E69F00","Three":"#CC79A7","Vodafone UK":"#D55E00"}
GEN_COL = {"2g":"#0072B2","4g":"#E69F00","5g":"#D55E00"}
GEN_LS  = {"2g":"-","4g":"--","5g":"-."}   # line styles: distinguishable in black-and-white
plt.rcParams.update({"figure.dpi":130,"font.size":15,"axes.grid":True,"grid.alpha":.25,
                     "axes.axisbelow":True,"hatch.linewidth":0.6})

def serv_op(gen):
    s=pd.read_parquet(f"cache/serving_{gen}.parquet")
    s=(s.sort_values("cov",kind="stable")
        .drop_duplicates(subset=["train","date","time","op"],keep="last"))
    return s.rename(columns={"cov":"rxp"})

def hav(a1,o1,a2,o2):
    R=6371.0; p1,p2=np.radians(a1),np.radians(a2)
    d1=np.radians(a2-a1); d2=np.radians(o2-o1)
    h=np.sin(d1/2)**2+np.cos(p1)*np.cos(p2)*np.sin(d2/2)**2
    return 2*R*np.arcsin(np.sqrt(h))

def tracks(s):
    s=s[s.op.isin(rm.MAIN_OPERATORS)].sort_values(["op","train","date","tsec"]).reset_index(drop=True)
    same=((s.op.values[1:]==s.op.values[:-1])&(s.train.values[1:]==s.train.values[:-1])
          &(s.date.values[1:]==s.date.values[:-1]))
    dt=np.empty(len(s)); dt[0]=np.nan; dt[1:]=s.tsec.values[1:]-s.tsec.values[:-1]
    dx=np.empty(len(s)); dx[0]=np.nan
    dx[1:]=hav(s.Latitude.values[:-1],s.Longitude.values[:-1],s.Latitude.values[1:],s.Longitude.values[1:])
    cont=np.empty(len(s),bool); cont[0]=False
    cont[1:]=same&(dt[1:]>=0)&(dt[1:]<=GAP_S)&(dx[1:]<=JUMP_KM)
    s["step_km"]=np.where(cont,dx,np.nan); s["seg"]=(~cont).cumsum()
    return s

def confirmed(pci):
    if len(pci)==0: return 0,0
    vals,lens=[],[]; cur,c=pci[0],1
    for p in pci[1:]:
        if p==cur: c+=1
        else: vals.append(cur); lens.append(c); cur,c=p,1
    vals.append(cur); lens.append(c)
    stable=[v for v,l in zip(vals,lens) if l>=MIN_DWELL]
    conf=sum(1 for a,b in zip(stable,stable[1:]) if a!=b)
    return conf,len(set(v for v in vals if v!="?"))

def hole_lengths(g,tau):
    below=(g.rxp.values<tau); cont=g.step_km.notna().to_numpy(); step=np.nan_to_num(g.step_km.values)
    holes=[]; cur=0.0; inh=False
    for i in range(len(g)):
        if not cont[i]:
            if inh and cur>0: holes.append(cur)
            inh=below[i]; cur=0.0; continue
        if below[i] and below[i-1]: cur+=step[i]; inh=True
        else:
            if inh and cur>0: holes.append(cur)
            inh=below[i]; cur=0.0
    if inh and cur>0: holes.append(cur)
    return holes

def run():
    ho_rows=[]; hole_rows=[]; pers_rows=[]; hole_cdf={}
    for gen,cfg in GENS.items():
        tau=cfg["tau"]; s=tracks(serv_op(gen))
        # ---- handover ----
        for o in [x for x in rm.MAIN_OPERATORS if (s.op==x).sum()>1000]:
            g=s[s.op==o]; cont=g.step_km.notna().to_numpy()
            pci=g.cell.fillna("?").astype(str).to_numpy()
            raw=np.zeros(len(g),bool); raw[1:]=cont[1:]&(pci[1:]!=pci[:-1])
            km=float(np.nansum(g.step_km.to_numpy())); conf=dist=0
            for _,seg in g.groupby("seg",observed=True):
                c,d=confirmed(seg.cell.fillna("?").astype(str).to_numpy()); conf+=c; dist+=d
            ho_rows.append({"gen":gen,"operator":o,"track_km":round(km,1),
                "raw_switch_per_km":round(int(raw.sum())/km,2) if km else np.nan,
                "confirmed_ho_per_km":round(conf/km,3) if km else np.nan,
                "distinct_cells_per_km":round(dist/km,3) if km else np.nan})
        # ---- holes (pooled over operators) ----
        allhl=[]
        for o in [x for x in rm.MAIN_OPERATORS if (s.op==x).sum()>1000]:
            g=s[s.op==o]; km=float(np.nansum(g.step_km.to_numpy()))
            hl=np.array(hole_lengths(g,tau)); allhl+= list(hl)
            hole_rows.append({"gen":gen,"operator":o,"tau":tau,"track_km":round(km,1),
                "n_holes":len(hl),"holes_per_100km":round(len(hl)/km*100,2) if km else np.nan,
                "median_km":round(float(np.median(hl)),3) if len(hl) else 0,
                "p90_km":round(float(np.percentile(hl,90)),3) if len(hl) else 0,
                "max_km":round(float(hl.max()),2) if len(hl) else 0,
                "frac_route_outage":round(float(hl.sum())/km,4) if km else np.nan})
        hole_cdf[gen]=np.array(allhl)
        # ---- persistence ----
        g=s.copy(); g["bx"]=np.floor(g.Latitude/0.02).astype("int32"); g["by"]=np.floor(g.Longitude/0.02).astype("int32")
        pp=g.groupby(["op","bx","by","train","date"],observed=True).rxp.median().reset_index()
        pp["out"]=(pp.rxp<tau).astype(int)
        b=pp.groupby(["op","bx","by"],observed=True).agg(n=("out","size"),of=("out","mean")).reset_index()
        m=b[b.n>=2]
        pers_rows.append({"gen":gen,"n_multipass_bins":len(m),
            "always_good":round(float((m.of==0).mean()),3),
            "always_bad":round(float((m.of==1).mean()),3),
            "sometimes":round(float(((m.of>0)&(m.of<1)).mean()),3)})
        print(f"[{gen}] track_km={np.nansum(s.step_km):,.0f} | "
              f"holes/100km(pooled)={len(hole_cdf[gen])/max(1,np.nansum(s.step_km))*100:.1f}")
    pd.DataFrame(ho_rows).to_csv(f"{TBL}/handover_legacy_rawswitch.csv",index=False)  # superseded by handover.py (A3)
    pd.DataFrame(hole_rows).to_csv(f"{TBL}/holes_all.csv",index=False)
    pd.DataFrame(pers_rows).to_csv(f"{TBL}/persistence_all.csv",index=False)
    # ---- figure: hole-length CDF by generation ----
    fig,ax=plt.subplots(figsize=(6,3.8))
    for gen in GENS:
        hl=hole_cdf[gen];
        if len(hl)==0: continue
        d=np.sort(hl); ax.plot(d,np.arange(1,len(d)+1)/len(d),label=f"{GENS[gen]['name']} (n={len(hl):,})",color=GEN_COL[gen],ls=GEN_LS[gen],lw=1.8)
    ax.set_xscale("log"); ax.set_xlabel("coverage-hole length (km)"); ax.set_ylabel("CDF of holes")
    ax.legend(fontsize=12); ax.grid(True,which="both",alpha=.25)
    save(fig,"fig_holes_by_generation")
    # ---- figure: persistence (always-good / always-bad / sometimes) ----
    pdf=pd.DataFrame(pers_rows); fig,ax=plt.subplots(figsize=(7.2,3.7)); x=np.arange(len(pdf)); bottom=np.zeros(len(pdf))
    cen={}
    for k,c,h in [("always_good","#009E73",""),("sometimes","#0072B2","xxx"),("always_bad","#D55E00","///")]:
        ax.bar(x,pdf[k],bottom=bottom,label=k,color=c,width=.55,hatch=h,edgecolor="black",linewidth=0.5)
        for i,v in enumerate(pdf[k]): cen[(i,k)]=(bottom[i]+v/2,v)
        bottom+=pdf[k].values
    BIG=0.10   # >=10% labelled inside the bar; thinner slices -> callout to the right
    for i in range(len(pdf)):
        thin=[]
        for k in ("always_good","sometimes","always_bad"):
            cy,v=cen[(i,k)]
            if v>=BIG: ax.text(x[i],cy,f"{v*100:.0f}%",ha="center",va="center",fontsize=13)
            elif v>0.005: thin.append((cy,v))
        if thin:
            thin.sort(); ys=[t[0] for t in thin]; MG=0.075
            for j in range(1,len(ys)):
                if ys[j]-ys[j-1]<MG: ys[j]=ys[j-1]+MG
            for (cy,v),ly in zip(thin,ys):
                ax.annotate(f"{v*100:.0f}%",xy=(x[i]+0.265,cy),xytext=(x[i]+0.33,ly),
                            ha="left",va="center",fontsize=12,
                            arrowprops=dict(arrowstyle="-",lw=0.7,color="0.45"))
    ax.set_xticks(x); ax.set_xticklabels([GENS[g]['name'] for g in pdf.gen])
    ax.set_ylim(0,1.06); ax.set_xlim(-0.6,len(pdf)-0.2)
    ax.set_ylabel("share of multi-pass 2km bins")
    ax.legend(ncol=3,fontsize=12,loc="lower left",bbox_to_anchor=(0,1.02,1,0.12),
              mode="expand",borderaxespad=0,frameon=False,columnspacing=1.0,handletextpad=0.5)
    save(fig,"fig_persistence_by_generation")
    print("\n=== handover ==="); print(pd.DataFrame(ho_rows).to_string(index=False))
    print("\n=== persistence ==="); print(pdf.to_string(index=False))

def save(fig,name):
    for e in ("pdf","png"): fig.savefig(f"{FIG}/{name}.{e}",bbox_inches="tight")
    plt.close(fig); print(f"[fig] {FIG}/{name}.png")

if __name__=="__main__": run()
