#!/usr/bin/env python3
"""
failures.py — separate TRANSIENT FAILURE-LIKE outages from STRUCTURAL coverage
holes, per generation, using the repeated passes of the same route.

Baseline (per operator, ~2km bin, same binning as the persistence analysis):
a bin observed on >=3 passes is
  normally-good : in outage (pass-median cov < tau) on <=40% of passes
  mostly-bad    : in outage on >=60% of passes
Each coverage hole (contiguous below-tau stretch on ONE pass, as in Fig 7) is
then classified by where its length lies:
  failure-like : >50% of its length crosses normally-good bins
                 -> the signal usually present there was missing/unusable
                    on that particular journey (transient anomaly)
  structural   : >50% of its length crosses mostly-bad bins
  unclassified : sparse (<3 passes) or mixed bins
Recovery (duration in units of passes): for each bad pass of a normally-good
bin, is the SAME bin good again on the next observed pass?

Caveats (stated in the paper): the scanner observes radio presence, not user
plane service, so failure counts are a LOWER bound; temporal resolution is one
pass, so duration is "recovered by next pass", not minutes.

Outputs -> results/tables/failures_all.csv, results/tables/failure_recovery.csv,
           results/figures/fig_failure_span.*
"""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); matplotlib.rcParams["pdf.fonttype"]=42; matplotlib.rcParams["ps.fonttype"]=42
import matplotlib.pyplot as plt
import rail_multi as rm

FIG, TBL = "results/figures", "results/tables"
os.makedirs(FIG, exist_ok=True); os.makedirs(TBL, exist_ok=True)
GAP_S, JUMP_KM, DEG = 600.0, 3.0, 0.02          # same run/bin rules as mobility.py
MIN_PASSES, GOOD_MAX, BAD_MIN = 3, 0.4, 0.6     # baseline rules
GENS = {"2g":{"name":"2G","tau":-95}, "4g":{"name":"4G","tau":-110}, "5g":{"name":"5G","tau":-110}}
GEN_COL = {"2g":"#0072B2","4g":"#E69F00","5g":"#D55E00"}
GEN_LS  = {"2g":"-","4g":"--","5g":"-."}
plt.rcParams.update({"figure.dpi":130,"font.size":15,"axes.grid":True,"grid.alpha":.25,"axes.axisbelow":True})

def serv_op(gen):
    s=pd.read_parquet(f"cache/serving_{gen}.parquet")
    s=(s.sort_values("cov",kind="stable")
        .drop_duplicates(subset=["train","date","time","op"],keep="last"))
    return s.rename(columns={"cov":"rxp"})

def hav(a1,o1,a2,o2):
    R=6371.0;p1,p2=np.radians(a1),np.radians(a2);d1=np.radians(a2-a1);d2=np.radians(o2-o1)
    h=np.sin(d1/2)**2+np.cos(p1)*np.cos(p2)*np.sin(d2/2)**2;return 2*R*np.arcsin(np.sqrt(h))

def tracks(s):
    s=s[s.op.isin(rm.MAIN_OPERATORS)].sort_values(["op","train","date","tsec"]).reset_index(drop=True)
    same=((s.op.values[1:]==s.op.values[:-1])&(s.train.values[1:]==s.train.values[:-1])
          &(s.date.values[1:]==s.date.values[:-1]))
    dt=np.empty(len(s)); dt[0]=np.nan; dt[1:]=s.tsec.values[1:]-s.tsec.values[:-1]
    dx=np.empty(len(s)); dx[0]=np.nan
    dx[1:]=hav(s.Latitude.values[:-1],s.Longitude.values[:-1],s.Latitude.values[1:],s.Longitude.values[1:])
    cont=np.empty(len(s),bool); cont[0]=False
    cont[1:]=same&(dt[1:]>=0)&(dt[1:]<=GAP_S)&(dx[1:]<=JUMP_KM)
    s["step_km"]=np.where(cont,dx,np.nan)
    s["bx"]=np.floor(s.Latitude/DEG).astype("int32"); s["by"]=np.floor(s.Longitude/DEG).astype("int32")
    return s

def run():
    rows=[]; rec_rows=[]; span_cdf={}
    for gen,cfg in GENS.items():
        tau=cfg["tau"]; s=tracks(serv_op(gen))
        s["dt_real"]=pd.to_datetime(s.date,format="%d/%m/%Y")
        # ---- per (op,bin) across-pass baseline (pass = train,date; pass-median rxp) ----
        pp=(s.groupby(["op","bx","by","train","date"],observed=True)
              .agg(rxp=("rxp","median"),d=("dt_real","first")).reset_index())
        pp["bad"]=(pp.rxp<tau).astype(int)
        b=(pp.groupby(["op","bx","by"],observed=True)
             .agg(n=("bad","size"),fb=("bad","mean")).reset_index())
        b["status"]=np.where((b.n>=MIN_PASSES)&(b.fb<=GOOD_MAX),"good",
                    np.where((b.n>=MIN_PASSES)&(b.fb>=BAD_MIN),"bad","other"))
        stat={(o,x,y):st for o,x,y,st in zip(b.op,b.bx,b.by,b.status)}
        # ---- walk holes per pass and split their length by bin status ----
        spans_fail=[]; spans_struct=[]; n_fail=n_struct=n_unc=0; km_fail=km_struct=km_unc=0.0
        for o in [x for x in rm.MAIN_OPERATORS if (s.op==x).sum()>1000]:
            g=s[s.op==o]
            below=(g.rxp.values<tau); cont=g.step_km.notna().to_numpy()
            step=np.nan_to_num(g.step_km.values)
            bxv=g.bx.to_numpy(); byv=g.by.to_numpy()
            def close(kg,kb,ko,tot):
                nonlocal n_fail,n_struct,n_unc,km_fail,km_struct,km_unc
                if tot<=0: return
                if kg/tot>0.5: n_fail+=1; km_fail+=tot; spans_fail.append(tot)
                elif kb/tot>0.5: n_struct+=1; km_struct+=tot; spans_struct.append(tot)
                else: n_unc+=1; km_unc+=tot
            kg=kb=ko=tot=0.0; inh=False
            for i in range(len(g)):
                if cont[i] and below[i] and below[i-1]:
                    st=stat.get((o,bxv[i],byv[i]),"other")
                    tot+=step[i]
                    if st=="good": kg+=step[i]
                    elif st=="bad": kb+=step[i]
                    else: ko+=step[i]
                    inh=True; continue
                if inh: close(kg,kb,ko,tot)
                kg=kb=ko=tot=0.0; inh=below[i] if cont[i] else False
            if inh: close(kg,kb,ko,tot)
        span_cdf[gen]={"fail":np.array(spans_fail),"struct":np.array(spans_struct)}
        # ---- recovery: bad pass of a normally-good bin -> good on next pass? ----
        good=b[b.status=="good"][["op","bx","by"]]
        gp=pp.merge(good,on=["op","bx","by"])
        gp=gp.sort_values(["op","bx","by","d"])
        nb=(gp.bad.values==1); nxt=np.empty(len(gp)); nxt[:]=np.nan
        samebin=((gp.op.values[1:]==gp.op.values[:-1])&(gp.bx.values[1:]==gp.bx.values[:-1])
                 &(gp.by.values[1:]==gp.by.values[:-1]))
        rec=n_obs=0
        for i in range(len(gp)-1):
            if nb[i] and samebin[i]:
                n_obs+=1; rec+= (not nb[i+1])
        tot_holes=n_fail+n_struct+n_unc; tot_km=km_fail+km_struct+km_unc
        rows.append({"gen":gen,"tau":tau,
            "holes_total":tot_holes,"holes_failure_like":n_fail,"holes_structural":n_struct,
            "holes_unclassified":n_unc,
            "share_count_failure":round(n_fail/tot_holes,3) if tot_holes else np.nan,
            "share_km_failure":round(km_fail/tot_km,3) if tot_km else np.nan,
            "share_km_structural":round(km_struct/tot_km,3) if tot_km else np.nan,
            "span_median_km":round(float(np.median(spans_fail)),3) if spans_fail else np.nan,
            "span_p90_km":round(float(np.percentile(spans_fail,90)),3) if spans_fail else np.nan,
            "span_max_km":round(float(np.max(spans_fail)),2) if spans_fail else np.nan,
            "struct_span_median_km":round(float(np.median(spans_struct)),3) if spans_struct else np.nan,
            "struct_span_p90_km":round(float(np.percentile(spans_struct,90)),3) if spans_struct else np.nan,
            "struct_span_max_km":round(float(np.max(spans_struct)),2) if spans_struct else np.nan})
        rec_rows.append({"gen":gen,"n_bad_passes_in_good_bins":n_obs,
            "recovered_by_next_pass":round(rec/n_obs,3) if n_obs else np.nan})
        print(f"[{gen}] holes={tot_holes:,} failure-like={n_fail:,} ({100*n_fail/max(1,tot_holes):.0f}%) "
              f"structural={n_struct:,} | span med={np.median(spans_fail):.3f}km p90={np.percentile(spans_fail,90):.2f}km "
              f"| recovery next-pass={100*rec/max(1,n_obs):.0f}% (n={n_obs:,})")
    pd.DataFrame(rows).to_csv(f"{TBL}/failures_all.csv",index=False)
    pd.DataFrame(rec_rows).to_csv(f"{TBL}/failure_recovery.csv",index=False)
    np.savez(f"{TBL}/failure_spans.npz",
             **{f"{g}_{k}":span_cdf[g][k] for g in GENS for k in ("fail","struct")})
    plot_fig(span_cdf)

def plot_fig(spans):
    """Per-generation violins of outage-event span, failure-like vs structural."""
    from matplotlib.patches import Patch
    CF,CS="#0072B2","#D55E00"                       # Okabe-Ito blue / orange
    fig,ax=plt.subplots(figsize=(7,3.5))
    gmax=-1e9
    for gi,gen in enumerate(GENS):
        for k,off,col,hat in [("fail",-0.2,CF,""),("struct",+0.2,CS,"///")]:
            d=spans[gen][k]; d=np.log10(d[d>0])
            if len(d)<10: continue
            vp=ax.violinplot([d],positions=[gi+1+off],widths=0.36,showmedians=True)
            for b in vp["bodies"]:
                b.set_facecolor(col); b.set_alpha(.75); b.set_edgecolor("black")
                b.set_linewidth(0.6); b.set_hatch(hat)
            for key in ("cmedians","cbars","cmins","cmaxes"):
                vp[key].set_color("black"); vp[key].set_linewidth(0.9)
            n=len(spans[gen][k]); gmax=max(gmax,float(d.max()))
            ax.annotate(f"{n:,}",xy=(gi+1+off,d.max()),xytext=(0,4),
                        textcoords="offset points",ha="center",fontsize=10,color="0.25")
    ax.set_xticks([1,2,3]); ax.set_xticklabels([GENS[g]["name"] for g in GENS])
    ax.set_yticks([-3,-2,-1,0,1])
    ax.set_yticklabels(["1 m","10 m","100 m","1 km","10 km"])
    ax.set_ylim(top=gmax+0.55)                       # headroom above count labels
    ax.set_ylabel("outage event span")
    handles=[Patch(facecolor=CF,alpha=.75,edgecolor="black",label="transient anomaly"),
             Patch(facecolor=CS,alpha=.75,edgecolor="black",hatch="///",label="structural")]
    ax.legend(handles=handles,ncol=2,fontsize=12,loc="lower center",
              bbox_to_anchor=(0.5,1.02),frameon=False,columnspacing=1.6,handletextpad=0.6)
    fig.tight_layout()
    for e in ("pdf","png"): fig.savefig(f"{FIG}/fig_failure_span.{e}",bbox_inches="tight")
    plt.close(fig); print(f"[fig] {FIG}/fig_failure_span.png")

def load_spans():
    z=np.load(f"{TBL}/failure_spans.npz")
    return {g:{k:z[f"{g}_{k}"] for k in ("fail","struct")} for g in GENS}

if __name__=="__main__":
    import sys
    if "--plot-only" in sys.argv: plot_fig(load_spans())
    else: run()
