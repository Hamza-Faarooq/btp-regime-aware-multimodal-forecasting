# Step 7 — Final Tables & Figures
# Generated from the attached BTP.ipynb; execute sequentially in one Colab runtime.

pd.set_option("display.width",200)
print("Macro-F1 levels with 95% bootstrap CI (P3 test; seed-averaged):")
print(levels.pivot(index="group",columns="model",values="f1").reindex([g for g,_ in groups]).round(3))
print("\nDifferences (positive = second model better), 95% CI from resampling test rows:")
print(diffs.assign(ci=lambda d:"["+d.lo.astype(str)+", "+d.hi.astype(str)+"]")[["group","n","comparison","est","ci"]].to_string(index=False))
fig,ax=plt.subplots(1,2,figsize=(14,4.3))
reg_groups=[g for g,_ in groups if g.startswith("Regime") or g=="All test"];ms=["M1","M2","M3","M6"];w=.2
for k,m in enumerate(ms):
    s=levels[levels.model==m].set_index("group").loc[reg_groups]
    ax[0].bar(np.arange(len(reg_groups))+(k-1.5)*w,s.f1,w,yerr=[s.f1-s.lo,s.hi-s.f1],capsize=2,label=NAME[m])
ax[0].axhline(levels[levels.model=="M0r"].f1.mean(),color="k",ls=":",lw=1,label="random guess")
ax[0].set_xticks(range(len(reg_groups)));ax[0].set_xticklabels([g.replace("Regime: ","") for g in reg_groups]);ax[0].set_ylim(.2,None);ax[0].set_ylabel("Test Macro-F1 (P3), 95% CI");ax[0].set_title("Macro-F1 by market regime");ax[0].legend(fontsize=8)
dg=[g for g,_ in groups if g!="All test"];cmp_show=list(COMPS)[:3];w=.26
for k,c in enumerate(cmp_show):
    s=diffs[diffs.comparison==c].set_index("group").loc[dg]
    ax[1].bar(np.arange(len(dg))+(k-1)*w,s.est,w,yerr=[s.est-s.lo,s.hi-s.est],capsize=2,label=c)
ax[1].axhline(0,color="k",lw=.8);ax[1].set_xticks(range(len(dg)));ax[1].set_xticklabels([g.replace("Regime: ","").replace("Stress window ","Stress\n") for g in dg],fontsize=8);ax[1].set_ylabel("Change in Macro-F1");ax[1].set_title("Gain from news / regime, by group (95% CI)");ax[1].legend(fontsize=7)
plt.tight_layout();plt.savefig(FIG/"fig_regime_news_value.png",dpi=200);plt.show()
