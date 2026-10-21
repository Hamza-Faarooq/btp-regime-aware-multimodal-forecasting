# =====================================================================
# BTP Steps 5-7: protocol audit, news value by regime, stress-window case study, figures
# Needs results/predictions_P1_P2_P3.parquet and metrics_P1_P2_P3.csv from 04_models.py (10 seeds).
# Set LAG below to match the step4 run you are analysing (0 = main analysis, 1 = robustness run). Cells are separated by "# %%".
# =====================================================================

# %% Cell 0: imports, config, helpers
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path

import os
BTP_ROOT = Path(os.environ.get("BTP_ROOT", "/content/drive/MyDrive/btp"))   # set BTP_ROOT to run outside Colab
ROOT = BTP_ROOT; RES = ROOT / "results"; FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)
LAG = 0                                  # must match 04_models.py / step2_regimes.py for the run being analysed
SUF = "" if LAG == 0 else f"_lag{LAG}"
STRESS = ("2020-02-01", "2020-04-30")          # stress-period case study window (by target date)
NBOOT = 1000
REG_NAMES = ["Low-vol", "Mid-vol", "High-vol"]
NAME = {"M0": "Majority", "M0r": "Random guess", "M1": "Price", "M2": "Price+News(emb)", "M3": "Price+News+Regime",
        "M4": "News only", "M5": "Regime only", "M6": "Price+Sentiment", "M7": "Gated regime fusion"}

def f1m(y, p):                                   # fast macro-F1 over classes 0,1,2 (class with no support and no predictions scores 0)
    cm = np.bincount(y * 3 + p, minlength=9).reshape(3, 3); tp = np.diag(cm).astype(float)
    den = cm.sum(0) + cm.sum(1)
    return float(np.mean(np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0)))

P = pd.read_parquet(RES / f"predictions_P1_P2_P3{SUF}.parquet"); M = pd.read_csv(RES / f"metrics_P1_P2_P3{SUF}.csv")
T = M[M.eval_split == "test"]
print("seeds per cell:", T.groupby(["protocol", "model"]).seed.nunique().unique().tolist())

# %% Cell 1: RQ1 -- evaluation-protocol audit (seed-level; CI = bootstrap over seeds)
tab = T.pivot_table(index="model", columns="protocol", values="macro_f1", aggfunc=["mean", "std"]).round(3)
print("TEST Macro-F1 by protocol (mean/std over seeds):\n", tab)

rng = np.random.default_rng(0)
def seed_gap(model, pa, pb, n=2000):
    a = T[(T.model == model) & (T.protocol == pa)].macro_f1.to_numpy(); b = T[(T.model == model) & (T.protocol == pb)].macro_f1.to_numpy()
    d = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n)]
    return a.mean() - b.mean(), *np.percentile(d, [2.5, 97.5])
rows = []
for m in ["M1", "M2", "M3", "M4", "M5", "M6", "M7"]:
    for pa in ["P1", "P2"]:
        e, lo, hi = seed_gap(m, pa, "P3"); rows.append(dict(model=m, gap=f"{pa} - P3", est=e, lo=lo, hi=hi))
gaps = pd.DataFrame(rows).round(3); print("\nOptimistic performance gap (positive = looser protocol scores higher):\n", gaps)
tab.to_csv(RES / f"audit_table{SUF}.csv"); gaps.to_csv(RES / f"audit_gaps{SUF}.csv", index=False)

fig, ax = plt.subplots(figsize=(11, 4.2)); ms = [m for m in ["M1", "M2", "M3", "M6", "M7", "M4", "M5"] if m in set(T.model)]; w = 0.26
for k, (p, c) in enumerate(zip(["P1", "P2", "P3"], ["#c44e52", "#dd8452", "#4c72b0"])):
    mu = [T[(T.model == m) & (T.protocol == p)].macro_f1.mean() for m in ms]; sd = [T[(T.model == m) & (T.protocol == p)].macro_f1.std() for m in ms]
    ax.bar(np.arange(len(ms)) + (k - 1) * w, mu, w, yerr=sd, capsize=2, color=c, label={"P1": "P1 random split, fit on all", "P2": "P2 chronological, fit on all", "P3": "P3 strict (train-only fit)"}[p])
ax.axhline(T[T.model == "M0r"].macro_f1.mean(), color="k", ls=":", lw=1, label="random-guess floor")
ax.set_xticks(range(len(ms))); ax.set_xticklabels([NAME[m] for m in ms], rotation=15); ax.set_ylabel("Test Macro-F1 (mean +/- sd over seeds)")
ax.set_ylim(0.2, None); ax.legend(fontsize=8); ax.set_title("Evaluation protocol vs apparent performance"); plt.tight_layout(); plt.savefig(FIG / f"fig_protocol_audit{SUF}.png", dpi=200); plt.show()

# %% Cell 2: RQ2 -- strict-protocol (P3) test predictions arranged as (seeds x rows)
p3 = P[(P.protocol == "P3") & (P.part == "test")]
ids = np.sort(p3.id.unique()); seeds = sorted(p3.seed.unique())
base = p3.drop_duplicates("id").set_index("id").loc[ids]
yt = base.y_true.to_numpy().astype(int); rg = base.regime_p3.to_numpy().astype(int); dt = pd.to_datetime(base.date.to_numpy())
PR = {m: np.stack([p3[(p3.model == m) & (p3.seed == s)].set_index("id").loc[ids, "y_pred"].to_numpy().astype(int) for s in seeds])
      for m in NAME if m != "M0"}
print("P3 test rows:", len(ids), "| seeds:", len(seeds), "| rows per regime:", {REG_NAMES[r]: int((rg == r).sum()) for r in range(3)})

def level(m, idx): return float(np.mean([f1m(yt[idx], PR[m][s][idx]) for s in range(len(seeds))]))
def boot_group(sub, n=NBOOT, seed=0):
    r = np.random.default_rng(seed); sub = np.asarray(sub); out = {m: np.empty(n) for m in PR}
    for b in range(n):
        i = sub[r.integers(0, len(sub), len(sub))]
        for m in PR: out[m][b] = np.mean([f1m(yt[i], PR[m][s][i]) for s in range(len(seeds))])
    return out
COMPS = {"M2-M1 (embedding news gain)": ("M1", "M2"), "M6-M1 (sentiment news gain)": ("M1", "M6"),
         "M3-M2 (regime effect)": ("M2", "M3"), "M3-M1 (news+regime vs price)": ("M1", "M3"),
         "M7-M3 (gated vs concatenated regime)": ("M3", "M7"), "M7-M1 (gated vs price only)": ("M1", "M7")}
COMPS = {k: v for k, v in COMPS.items() if v[0] in PR and v[1] in PR}   # skip comparisons whose models are absent
def summarize_group(label, sub):
    bs = boot_group(sub); lv, df_ = [], []
    for m in [m for m in ["M0r", "M1", "M2", "M3", "M6", "M7", "M4", "M5"] if m in PR]:
        lo, hi = np.percentile(bs[m], [2.5, 97.5]); lv.append(dict(group=label, n=len(sub), model=m, f1=level(m, np.asarray(sub)), lo=lo, hi=hi))
    for name, (a, b) in COMPS.items():
        d = bs[b] - bs[a]; lo, hi = np.percentile(d, [2.5, 97.5])
        df_.append(dict(group=label, n=len(sub), comparison=name, est=level(b, np.asarray(sub)) - level(a, np.asarray(sub)), lo=lo, hi=hi))
    return lv, df_

groups = [("All test", np.arange(len(ids)))] + [(f"Regime: {REG_NAMES[r]}", np.where(rg == r)[0]) for r in range(3)]
s0, s1 = pd.Timestamp(STRESS[0]), pd.Timestamp(STRESS[1]); inside = ((dt >= s0) & (dt <= s1))
for lab, msk in [(f"Stress window {STRESS[0]}..{STRESS[1]}", inside), ("Outside stress window", ~inside)]:
    if msk.sum() >= 10: groups.append((lab, np.where(msk)[0]))
    else: print("skipping", lab, "- only", int(msk.sum()), "rows")
L, D = [], []
for lab, sub in groups:
    a, b = summarize_group(lab, sub); L += a; D += b
levels = pd.DataFrame(L).round(3); diffs = pd.DataFrame(D).round(3)
levels.to_csv(RES / f"p3_levels_by_group{SUF}.csv", index=False); diffs.to_csv(RES / f"p3_diffs_by_group{SUF}.csv", index=False)

# %% Cell 3: print the result tables (paste these back)
pd.set_option("display.width", 200)
print("Macro-F1 levels with 95% bootstrap CI (P3 test; seed-averaged):")
print(levels.pivot(index="group", columns="model", values="f1").reindex([g for g, _ in groups]).round(3))
print("\nDifferences (positive = second model better), 95% CI from resampling test rows:")
print(diffs.assign(ci=lambda d: "[" + d.lo.astype(str) + ", " + d.hi.astype(str) + "]")[["group", "n", "comparison", "est", "ci"]].to_string(index=False))

# %% Cell 4: figures
fig, ax = plt.subplots(1, 2, figsize=(14, 4.3))
reg_groups = [g for g, _ in groups if g.startswith("Regime") or g == "All test"]; ms = [m for m in ["M1", "M2", "M3", "M6", "M7"] if m in PR]; w = 0.16
for k, m in enumerate(ms):
    s = levels[levels.model == m].set_index("group").loc[reg_groups]
    ax[0].bar(np.arange(len(reg_groups)) + (k - (len(ms) - 1) / 2) * w, s.f1, w, yerr=[s.f1 - s.lo, s.hi - s.f1], capsize=2, label=NAME[m])
ax[0].axhline(levels[levels.model == "M0r"].f1.mean(), color="k", ls=":", lw=1, label="random guess")
ax[0].set_xticks(range(len(reg_groups))); ax[0].set_xticklabels([g.replace("Regime: ", "") for g in reg_groups]); ax[0].set_ylim(0.2, None)
ax[0].set_ylabel("Test Macro-F1 (P3), 95% CI"); ax[0].set_title("Macro-F1 by market regime"); ax[0].legend(fontsize=8)

dg = [g for g, _ in groups if g != "All test"]; cmp_show = list(COMPS)[:3]; w = 0.26
for k, c in enumerate(cmp_show):
    s = diffs[diffs.comparison == c].set_index("group").loc[dg]
    ax[1].bar(np.arange(len(dg)) + (k - 1) * w, s.est, w, yerr=[s.est - s.lo, s.hi - s.est], capsize=2, label=c)
ax[1].axhline(0, color="k", lw=.8); ax[1].set_xticks(range(len(dg))); ax[1].set_xticklabels([g.replace("Regime: ", "").replace("Stress window ", "Stress\n") for g in dg], fontsize=8)
ax[1].set_ylabel("Change in Macro-F1"); ax[1].set_title("Gain from news / regime, by group (95% CI)"); ax[1].legend(fontsize=7)
plt.tight_layout(); plt.savefig(FIG / f"fig_regime_news_value{SUF}.png", dpi=200); plt.show()

# %% Cell 5: gate analysis for M7 -- how much weight on price (g near 1) vs news (g near 0), by regime
g7 = P[(P.protocol == "P3") & (P.model == "M7")]
if len(g7):
    def gate_tab(part):
        d = g7[g7.part == part]
        t = d.groupby(["seed", "regime_p3"]).gate_price.mean().unstack()               # seed x regime
        return t, t.agg(["mean", "std"]).round(3)
    for part in ["train", "test"]:
        t, summ = gate_tab(part); summ.columns = [REG_NAMES[c] for c in summ.columns]
        print(f"Mean price-gate by regime, P3 {part} rows (mean/sd over {len(t)} seeds):\n", summ)
    t_te, _ = gate_tab("test")
    d_hl = t_te[2] - t_te[0]
    print(f"\nHigh-vol minus Low-vol gate (test): mean {d_hl.mean():.3f}, sd {d_hl.std():.3f}, same sign in {int((np.sign(d_hl) == np.sign(d_hl.mean())).sum())}/{len(d_hl)} seeds")
    # share of gate variance explained by the regime label (eta^2), per seed
    eta = []
    for s_, d in g7[g7.part == "test"].groupby("seed"):
        tot = d.gate_price.var(ddof=0); bet = d.groupby("regime_p3").gate_price.transform("mean").var(ddof=0); eta.append(bet / tot if tot > 0 else np.nan)
    print(f"Share of row-level gate variance explained by regime: mean {np.nanmean(eta):.3f} (sd {np.nanstd(eta):.3f})  [0 = regime irrelevant, 1 = gate depends only on regime]")
    print("Overall mean price-gate (test):", round(float(g7[g7.part == "test"].gate_price.mean()), 3), "| row-level sd:", round(float(g7[g7.part == "test"].gate_price.std()), 3))
    sw = pd.to_datetime(g7.date); dts_ = g7[(g7.part == "test")]
    inw = (pd.to_datetime(dts_.date) >= s0) & (pd.to_datetime(dts_.date) <= s1)
    print("Mean price-gate in stress window:", round(float(dts_[inw].gate_price.mean()), 3), "| outside:", round(float(dts_[~inw].gate_price.mean()), 3))

    fig, ax = plt.subplots(figsize=(6, 3.8)); w = 0.35
    for k, (part, c) in enumerate([("train", "#9aa5b1"), ("test", "#4c72b0")]):
        t, _ = gate_tab(part)
        ax.bar(np.arange(3) + (k - .5) * w, t.mean(), w, yerr=t.std(), capsize=3, color=c, label=part)
    ax.axhline(0.5, color="k", ls=":", lw=1); ax.set_xticks(range(3)); ax.set_xticklabels(REG_NAMES); ax.set_ylim(0, 1)
    ax.set_ylabel("mean gate on price branch (1 = price, 0 = news)"); ax.set_title("M7 gate by regime (P3, mean +/- sd over seeds)"); ax.legend()
    plt.tight_layout(); plt.savefig(FIG / f"fig_gate_by_regime{SUF}.png", dpi=200); plt.show()
else:
    print("no M7 predictions found; rerun 04_models.py run_grid with the updated file")

# %% Cell 6: LAG robustness -- run this after BOTH the LAG=0 and LAG=1 analyses have been saved
f0, f1 = RES / "p3_diffs_by_group.csv", RES / "p3_diffs_by_group_lag1.csv"
l0, l1 = RES / "p3_levels_by_group.csv", RES / "p3_levels_by_group_lag1.csv"
if all(p.exists() for p in [f0, f1, l0, l1]):
    A, B = pd.read_csv(l0), pd.read_csv(l1)
    cmpl = A[A.group == "All test"].set_index("model")[["f1"]].join(B[B.group == "All test"].set_index("model")[["f1"]], lsuffix="_lag0", rsuffix="_lag1").round(3)
    print("Test Macro-F1, strict protocol, all test rows:\n", cmpl)
    D0, D1 = pd.read_csv(f0), pd.read_csv(f1)
    m = D0.merge(D1, on=["group", "comparison"], suffixes=("_lag0", "_lag1"))
    m["lag0"] = m.est_lag0.round(3).astype(str) + " [" + m.lo_lag0.round(3).astype(str) + ", " + m.hi_lag0.round(3).astype(str) + "]"
    m["lag1"] = m.est_lag1.round(3).astype(str) + " [" + m.lo_lag1.round(3).astype(str) + ", " + m.hi_lag1.round(3).astype(str) + "]"
    m["same_sign_and_ci_verdict"] = [(np.sign(a) == np.sign(b)) and ((lo0 > 0 or hi0 < 0) == (lo1 > 0 or hi1 < 0))
                                      for a, b, lo0, hi0, lo1, hi1 in zip(m.est_lag0, m.est_lag1, m.lo_lag0, m.hi_lag0, m.lo_lag1, m.hi_lag1)]
    print("\nKey differences, LAG=0 vs LAG=1 (estimate [95% CI]):")
    print(m[m.group == "All test"][["comparison", "lag0", "lag1", "same_sign_and_ci_verdict"]].to_string(index=False))
    print(f"\nAcross all {len(m)} group x comparison cells: same sign and same CI-excludes-zero verdict in {int(m.same_sign_and_ci_verdict.sum())}")
    m.to_csv(RES / "lag_robustness.csv", index=False)
else:
    print("Run Cells 0-5 once with LAG = 0 and once with LAG = 1 (after the matching step2/step4 runs), then rerun this cell.")
