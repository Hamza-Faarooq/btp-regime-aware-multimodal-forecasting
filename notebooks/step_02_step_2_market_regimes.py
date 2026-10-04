# Step 2 — Market Regimes
# Generated from the attached BTP.ipynb; execute sequentially in one Colab runtime.

# ===== Original notebook cell 3 =====
# =====================================================================
# BTP Step 2: market regimes (3-state KMeans on rolling return + volatility)
# Run in Colab after Step 1 has saved clean.parquet and daily_series.parquet.
# Cells are separated by "# %%".
# =====================================================================

# %% Cell 0: imports and config
import itertools, joblib
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

ROOT = Path("/content/drive/MyDrive/btp"); DATA = ROOT / "data"; FIG = ROOT / "figures"; RES = ROOT / "results"
FIG.mkdir(exist_ok=True); RES.mkdir(exist_ok=True)
LAG = 0          # MUST match LAG in Step 1 Cell 6 (0 = window ends at target date t)
W = 20           # rolling window (trading days) for regime features
K = 3            # number of regimes
KM_SEED = 0      # fixed: the regime labelling is part of data preparation, not a per-model seed
NAMES = ["Low-vol", "Mid-vol", "High-vol"]   # named by volatility rank only, never by guesswork
COLORS = ["tab:green", "tab:orange", "tab:red"]

# %% Cell 1: load Step 1 outputs
df = pd.read_parquet(DATA / "clean.parquet").reset_index(drop=True)
daily = pd.read_parquet(DATA / "daily_series.parquet").sort_values("date").reset_index(drop=True)
print(df.shape, daily.shape)

# %% Cell 2: regime features (each uses only prices up to its own date)
daily["ret"] = daily["close"].pct_change()
daily["ret20"] = daily["close"] / daily["close"].shift(W) - 1          # 20-day return
daily["vol20"] = daily["ret"].rolling(W).std()                         # 20-day volatility of daily returns
daily["lvol20"] = np.log(daily["vol20"])                               # log: volatility is right-skewed, so clusters are not dominated by crashes
FEAT = ["ret20", "lvol20"]

# last daily row each target row is allowed to use (same rule as the price windows in Step 1)
idx = daily["date"].searchsorted(df["date"].values, side="right") - 1 - LAG
assert (idx >= 0).all()
assert (daily["date"].to_numpy()[idx] <= df["date"].to_numpy()).all(), "LEAK: regime feature uses a future day"
for c in ["ret20", "vol20", "lvol20"]:
    df[c] = daily[c].to_numpy()[idx]
print("NaNs in regime features:", int(df[FEAT].isna().sum().sum()))

# %% Cell 3: the fit function -- takes the rows to fit on, which is what the leakage audit varies
def fit_regimes(X, seed=KM_SEED):
    sc = StandardScaler().fit(X)
    km = KMeans(n_clusters=K, n_init=10, random_state=seed).fit(sc.transform(X))
    centers = sc.inverse_transform(km.cluster_centers_)
    order = np.argsort(centers[:, 1])                  # rank clusters by mean log-volatility
    remap = np.empty(K, dtype=int); remap[order] = np.arange(K)
    return dict(sc=sc, km=km, remap=remap)

def assign_regimes(model, X):
    return model["remap"][model["km"].predict(model["sc"].transform(X))]

is_train = (df["split"] == "train").to_numpy()
X_all = df[FEAT].to_numpy()
m_p3 = fit_regimes(X_all[is_train])      # P3 (strict): fit on training rows only
m_p12 = fit_regimes(X_all)               # P1/P2: fit on all rows (deliberately leaky)
df["regime_p3"] = assign_regimes(m_p3, X_all)
df["regime_all"] = assign_regimes(m_p12, X_all)

agree = (df["regime_p3"] == df["regime_all"])
print("share of rows with the same regime under train-only vs all-data fit:", round(agree.mean(), 4))
print(agree.groupby(df["split"]).mean().round(4).to_dict())

# %% Cell 4: inspect the regimes (paste this output back)
def regime_stats(d, col):
    g = d.groupby(col).agg(n=("ret20", "size"), mean_ret20=("ret20", "mean"), mean_vol20=("vol20", "mean"),
                           mean_next_day_ret=("pct_change", "mean"))
    g["share"] = g["n"] / g["n"].sum()
    g.index = [NAMES[i] for i in g.index]
    return g.round(4)

stats_train = regime_stats(df[is_train], "regime_p3")
print("Regime stats on TRAIN rows (P3 fit):
", stats_train)
print("
Regime share by split (P3 fit):
", pd.crosstab(df["split"], df["regime_p3"].map(dict(enumerate(NAMES))), normalize="index").round(3))
print("
Label mix within each regime (P3 fit), all splits:
",
      pd.crosstab(df["regime_p3"].map(dict(enumerate(NAMES))), df["label"], normalize="index").round(3))

# %% Cell 5: persistence -- transition matrix and run lengths on the DAILY series (P3 model)
ok = daily[FEAT].notna().all(axis=1).to_numpy()
daily["regime_p3"] = np.nan
daily.loc[ok, "regime_p3"] = assign_regimes(m_p3, daily.loc[ok, FEAT].to_numpy())
seq = daily["regime_p3"].dropna().astype(int).to_numpy()
T = pd.crosstab(pd.Series(seq[:-1], name="from"), pd.Series(seq[1:], name="to"), normalize="index").round(3)
T.index = [NAMES[i] for i in T.index]; T.columns = [NAMES[i] for i in T.columns]
print("Transition matrix (daily, P3 model):
", T)
runs = pd.DataFrame([(NAMES[k], len(list(g))) for k, g in itertools.groupby(seq)], columns=["regime", "run_len"])
print("
Run length in trading days:
", runs.groupby("regime")["run_len"].agg(["count", "mean", "median", "max"]).round(1))

# %% Cell 6: figures
fig, ax = plt.subplots(1, 2, figsize=(15, 4.4))
d = daily.dropna(subset=["regime_p3"])
for r in range(K):
    s = d[d["regime_p3"] == r]
    ax[0].scatter(s["date"], s["close"], s=3, color=COLORS[r], label=NAMES[r])
for sp, c in [("valid", "k"), ("test", "k")]:
    ax[0].axvline(df.loc[df["split"] == sp, "date"].min(), color=c, ls="--", lw=.8)
ax[0].set_title("SPY close coloured by regime (train-only fit); dashed = valid/test start")
ax[0].legend(markerscale=4)

tr = df[is_train]
for r in range(K):
    s = tr[tr["regime_p3"] == r]
    ax[1].scatter(s["ret20"], s["lvol20"], s=6, color=COLORS[r], alpha=.6, label=NAMES[r])
cen = m_p3["sc"].inverse_transform(m_p3["km"].cluster_centers_)
ax[1].scatter(cen[:, 0], cen[:, 1], marker="X", s=140, c="k", label="centres")
ax[1].set_xlabel("20-day return"); ax[1].set_ylabel("log 20-day volatility"); ax[1].set_title("Training rows in regime-feature space")
ax[1].legend()
plt.tight_layout(); plt.savefig(FIG / "regimes_overview.png", dpi=200); plt.show()

# %% Cell 7: save
df[["id", "ret20", "vol20", "lvol20", "regime_p3", "regime_all"]].to_parquet(DATA / "regimes.parquet")
stats_train.to_csv(RES / "regime_stats_train.csv"); T.to_csv(RES / "regime_transition_matrix.csv")
joblib.dump(m_p3, DATA / "regime_model_p3.joblib"); joblib.dump(m_p12, DATA / "regime_model_p12.joblib")
print("saved regimes.parquet, stats, transition matrix, models")
