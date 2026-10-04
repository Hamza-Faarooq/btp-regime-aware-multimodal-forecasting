# Step 5 — Evaluation Protocol Audit
# Generated from the attached BTP.ipynb; execute sequentially in one Colab runtime.

import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
ROOT=Path("/content/drive/MyDrive/btp"); RES=ROOT/"results"; FIG=ROOT/"figures"; FIG.mkdir(exist_ok=True)
STRESS=("2020-02-01","2020-04-30"); NBOOT=1000
REG_NAMES=["Low-vol","Mid-vol","High-vol"]
NAME={"M0":"Majority","M0r":"Random guess","M1":"Price","M2":"Price+News(emb)","M3":"Price+News+Regime","M4":"News only","M5":"Regime only","M6":"Price+Sentiment"}
def f1m(y,p):
    cm=np.bincount(y*3+p,minlength=9).reshape(3,3);tp=np.diag(cm).astype(float);den=cm.sum(0)+cm.sum(1)
    return float(np.mean(np.where(den>0,2*tp/np.maximum(den,1),0.0)))
P=pd.read_parquet(RES/"predictions_P1_P2_P3.parquet"); M=pd.read_csv(RES/"metrics_P1_P2_P3.csv"); T=M[M.eval_split=="test"]
print("seeds per cell:",T.groupby(["protocol","model"]).seed.nunique().unique().tolist())
tab=T.pivot_table(index="model",columns="protocol",values="macro_f1",aggfunc=["mean","std"]).round(3);print(tab)
rng=np.random.default_rng(0)
def seed_gap(model,pa,pb,n=2000):
    a=T[(T.model==model)&(T.protocol==pa)].macro_f1.to_numpy();b=T[(T.model==model)&(T.protocol==pb)].macro_f1.to_numpy()
    d=[rng.choice(a,len(a)).mean()-rng.choice(b,len(b)).mean() for _ in range(n)]
    return a.mean()-b.mean(),*np.percentile(d,[2.5,97.5])
rows=[]
for m in ["M1","M2","M3","M4","M5","M6"]:
    for pa in ["P1","P2"]:
        e,lo,hi=seed_gap(m,pa,"P3");rows.append(dict(model=m,gap=f"{pa} - P3",est=e,lo=lo,hi=hi))
gaps=pd.DataFrame(rows).round(3);print(gaps);tab.to_csv(RES/"audit_table.csv");gaps.to_csv(RES/"audit_gaps.csv",index=False)
fig,ax=plt.subplots(figsize=(10,4.2));ms=["M1","M2","M3","M6","M4","M5"];w=.26
for k,p in enumerate(["P1","P2","P3"]):
    mu=[T[(T.model==m)&(T.protocol==p)].macro_f1.mean() for m in ms];sd=[T[(T.model==m)&(T.protocol==p)].macro_f1.std() for m in ms]
    ax.bar(np.arange(len(ms))+(k-1)*w,mu,w,yerr=sd,capsize=2,label={"P1":"P1 random split, fit on all","P2":"P2 chronological, fit on all","P3":"P3 strict (train-only fit)"}[p])
ax.axhline(T[T.model=="M0r"].macro_f1.mean(),color="k",ls=":",lw=1,label="random-guess floor")
ax.set_xticks(range(len(ms)));ax.set_xticklabels([NAME[m] for m in ms],rotation=15);ax.set_ylabel("Test Macro-F1 (mean +/- sd over seeds)");ax.set_ylim(.2,None);ax.legend(fontsize=8);ax.set_title("Evaluation protocol vs apparent performance");plt.tight_layout();plt.savefig(FIG/"fig_protocol_audit.png",dpi=200);plt.show()
