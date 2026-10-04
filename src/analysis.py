# Step 6 — News Value by Regime & Stress
# Generated from the attached BTP.ipynb; execute sequentially in one Colab runtime.

p3=P[(P.protocol=="P3")&(P.part=="test")]
ids=np.sort(p3.id.unique());seeds=sorted(p3.seed.unique());base=p3.drop_duplicates("id").set_index("id").loc[ids]
yt=base.y_true.to_numpy().astype(int);rg=base.regime_p3.to_numpy().astype(int);dt=pd.to_datetime(base.date.to_numpy())
PR={m:np.stack([p3[(p3.model==m)&(p3.seed==s)].set_index("id").loc[ids,"y_pred"].to_numpy().astype(int) for s in seeds]) for m in NAME if m!="M0"}
print("P3 test rows:",len(ids),"| seeds:",len(seeds),"| rows per regime:",{REG_NAMES[r]:int((rg==r).sum()) for r in range(3)})
def level(m,idx):return float(np.mean([f1m(yt[idx],PR[m][s][idx]) for s in range(len(seeds))]))
def boot_group(sub,n=NBOOT,seed=0):
    r=np.random.default_rng(seed);sub=np.asarray(sub);out={m:np.empty(n) for m in PR}
    for b in range(n):
        i=sub[r.integers(0,len(sub),len(sub))]
        for m in PR:out[m][b]=np.mean([f1m(yt[i],PR[m][s][i]) for s in range(len(seeds))])
    return out
COMPS={"M2-M1 (embedding news gain)":("M1","M2"),"M6-M1 (sentiment news gain)":("M1","M6"),"M3-M2 (regime effect)":("M2","M3"),"M3-M1 (news+regime vs price)":("M1","M3")}
def summarize_group(label,sub):
    bs=boot_group(sub);lv=[];df_=[]
    for m in ["M0r","M1","M2","M3","M6","M4","M5"]:
        lo,hi=np.percentile(bs[m],[2.5,97.5]);lv.append(dict(group=label,n=len(sub),model=m,f1=level(m,np.asarray(sub)),lo=lo,hi=hi))
    for name,(a,b) in COMPS.items():
        d=bs[b]-bs[a];lo,hi=np.percentile(d,[2.5,97.5]);df_.append(dict(group=label,n=len(sub),comparison=name,est=level(b,np.asarray(sub))-level(a,np.asarray(sub)),lo=lo,hi=hi))
    return lv,df_
groups=[("All test",np.arange(len(ids)))]+[(f"Regime: {REG_NAMES[r]}",np.where(rg==r)[0]) for r in range(3)]
s0,s1=pd.Timestamp(STRESS[0]),pd.Timestamp(STRESS[1]);inside=(dt>=s0)&(dt<=s1)
for lab,msk in [(f"Stress window {STRESS[0]}..{STRESS[1]}",inside),("Outside stress window",~inside)]:
    if msk.sum()>=10:groups.append((lab,np.where(msk)[0]))
L,D=[],[]
for lab,sub in groups:a,b=summarize_group(lab,sub);L+=a;D+=b
levels=pd.DataFrame(L).round(3);diffs=pd.DataFrame(D).round(3)
levels.to_csv(RES/"p3_levels_by_group.csv",index=False);diffs.to_csv(RES/"p3_diffs_by_group.csv",index=False)
for m in ["M1","M2","M3","M6"]:print(m,"predicted share Fall/Neutral/Rise:",np.round([np.mean(PR[m]==k) for k in range(3)],3))
print("true test share:",np.round([np.mean(yt==k) for k in range(3)],3))
def f1c(y,p):
    cm=np.bincount(y*3+p,minlength=9).reshape(3,3);tp=np.diag(cm);den=cm.sum(0)+cm.sum(1);return 2*tp/np.maximum(den,1)
for m in ["M1","M6"]:print(m,"per-class F1 (Fall/Neutral/Rise):",np.mean([f1c(yt,PR[m][s]) for s in range(len(seeds))],axis=0).round(3))
