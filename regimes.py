#!/usr/bin/env python3
"""
regimes.py — Step 5 (headline). Coverage- vs interference-limited classification
per generation, with a threshold sweep and a quality-unknown bucket.

Per sample, given coverage threshold tau and quality threshold sigma:
  coverage-limited     : cov < tau                       (cov metric always present)
  interference-limited : cov >= tau AND qual present AND qual < sigma
  acceptable           : cov >= tau AND qual present AND qual >= sigma
  quality-unknown      : cov >= tau AND qual missing      (mainly 2G)

Headline metric (robust to unknowns): interference share OF DEFICIENCY =
  interference / (interference + coverage).

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
 "2g": {"name":"2G (GSM)", "cov":"RxLev (dBm)", "qual":"C/I (dB)",
        "tau":-95, "sigma":9,  "tau_grid":[-90,-95,-100,-105,-110], "sigma_grid":[6,9,12,18]},
 "4g": {"name":"4G (LTE)", "cov":"RSRP (dBm)", "qual":"SINR (dB)",
        "tau":-110,"sigma":0,  "tau_grid":[-100,-105,-110,-115,-120],"sigma_grid":[0,3,6,10]},
 "5g": {"name":"5G (NR)",  "cov":"RSRP (dBm)", "qual":"SINR (dB)",
        "tau":-110,"sigma":0,  "tau_grid":[-100,-105,-110,-115,-120],"sigma_grid":[0,3,6,10]},
}
# Okabe-Ito palette (high-contrast, colourblind-safe): green=good, blue=highlighted, red=bad
REG_COL = {"coverage":"#D55E00","interference":"#0072B2","acceptable":"#009E73","unknown":"#999999"}
# hatch textures so bars stay distinguishable in a black-and-white print
REG_HATCH = {"coverage":"///","interference":"xxx","acceptable":"...","unknown":""}
# display labels for the legend (keys stay short for the dicts/CSV)
REG_LABEL = {"coverage":"coverage-limited","interference":"interference-limited",
             "acceptable":"acceptable","unknown":"unknown"}
plt.rcParams.update({"figure.dpi":130,"font.size":15,"axes.grid":True,"grid.alpha":.25,
                     "axes.axisbelow":True,"hatch.linewidth":0.6})

def serv_op(gen):
    s = pd.read_parquet(f"cache/serving_{gen}.parquet")
    s = (s.sort_values("cov",kind="stable")
          .drop_duplicates(subset=["train","date","time","op"],keep="last"))
    return s.rename(columns={"cov":"rxp"})

def fractions(s, tau, sigma):
    cov  = (s.rxp < tau)
    known= s["qual"].notna()
    intf = (~cov) & known & (s["qual"] < sigma)
    acc  = (~cov) & known & (s["qual"] >= sigma)
    unk  = (~cov) & (~known)
    n=len(s)
    return {"coverage":cov.mean(),"interference":intf.mean(),
            "acceptable":acc.mean(),"unknown":unk.mean(),
            "intf_of_deficiency": float(intf.sum())/max(1,(intf.sum()+cov.sum())), "n":n}

def run():
    summary=[]
    for gen,cfg in GENS.items():
        s=serv_op(gen)
        # sweep table
        rows=[]
        for tau in cfg["tau_grid"]:
            for sig in cfg["sigma_grid"]:
                f=fractions(s,tau,sig); rows.append({"gen":gen,"tau":tau,"sigma":sig,**f})
        pd.DataFrame(rows).round(4).to_csv(f"{TBL}/regime_sweep_{gen}.csv",index=False)
        # anchor split
        f=fractions(s,cfg["tau"],cfg["sigma"]); f.update({"gen":gen,"name":cfg["name"],
             "tau":cfg["tau"],"sigma":cfg["sigma"]}); summary.append(f)
        # per-gen density (cov vs qual, where qual present)
        d=s[s["qual"].notna()]
        fig,ax=plt.subplots(figsize=(5.2,4.2))
        hb=ax.hexbin(d.rxp,d["qual"],gridsize=70,bins="log",cmap="viridis",mincnt=1)
        hb.set_rasterized(True)   # vector-PDF backend drops hexbin; rasterize this layer
        fig.colorbar(hb,ax=ax,label="log$_{10}$ count")
        ax.axvline(cfg["tau"],color="w",ls="--",lw=1.2); ax.axhline(cfg["sigma"],color="w",ls="--",lw=1.2)
        ax.set_xlabel(cfg["cov"]); ax.set_ylabel(cfg["qual"])
        ax.set_title(f"Regime density — {cfg['name']} (τ={cfg['tau']}, σ={cfg['sigma']})")
        fig.tight_layout(); save(fig,f"fig_regime_density_{gen}")
        print(f"[{gen}] cov={f['coverage']*100:.1f}% intf={f['interference']*100:.1f}% "
              f"acc={f['acceptable']*100:.1f}% unk={f['unknown']*100:.1f}% | "
              f"intf-of-deficiency={f['intf_of_deficiency']*100:.0f}%")
    sdf=pd.DataFrame(summary)[["gen","name","tau","sigma","coverage","interference",
                               "acceptable","unknown","intf_of_deficiency","n"]]
    sdf.round(4).to_csv(f"{TBL}/regime_summary.csv",index=False)
    # ---- headline cross-generation stacked bar ----
    fig,ax=plt.subplots(figsize=(7,3.4))
    order=["coverage","interference","acceptable","unknown"]
    labels=[GENS[g]["name"] for g in GENS]; x=np.arange(len(GENS)); bottom=np.zeros(len(GENS))
    for r in order:
        vals=[summary[i][r] for i in range(len(GENS))]
        ax.bar(x,vals,bottom=bottom,color=REG_COL[r],label=REG_LABEL[r],width=.45,
               hatch=REG_HATCH[r],edgecolor="black",linewidth=0.5)
        for i,v in enumerate(vals):
            if v>.03: ax.text(x[i],bottom[i]+v/2,f"{v*100:.0f}%",ha="center",va="center",fontsize=13)
        bottom+=vals
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0,1.06)
    ax.set_ylabel("share of serving samples")
    ax.legend(ncol=2,fontsize=12,loc="lower left",bbox_to_anchor=(0,1.02,1,0.16),
              mode="expand",borderaxespad=0,frameon=False,columnspacing=1.2,handletextpad=0.5)
    fig.tight_layout(); save(fig,"fig_regime_by_generation")
    print("\n=== regime_summary ==="); print(sdf.to_string(index=False))

def save(fig,name):
    for e in ("pdf","png"): fig.savefig(f"{FIG}/{name}.{e}",bbox_inches="tight",dpi=200)
    plt.close(fig); print(f"[fig] {FIG}/{name}.png")

if __name__=="__main__": run()
