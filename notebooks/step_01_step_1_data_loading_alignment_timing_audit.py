# Step 1 — Data Loading, Alignment & Timing Audit
# Generated from the attached BTP.ipynb; execute sequentially in one Colab runtime.

# ===== Original notebook cell 1 =====
# =====================================================================
# BTP Step 1: load NIFTY, build clean table, lagged windows, timing audit
# Run in Google Colab, cell by cell (cells are separated by "# %%").
# Colab setup (run once in its own cell):  !pip -q install datasets pyarrow
# =====================================================================

# %% Cell 0: imports and config
import re, io, json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

LOOKBACK = 10
OUT = Path("/content/drive/MyDrive/btp/data"); OUT.mkdir(parents=True, exist_ok=True)
FIG = Path("/content/drive/MyDrive/btp/figures"); FIG.mkdir(parents=True, exist_ok=True)

HEADER = ("date,open,high,low,close,adj_close,volume,pct_change,macd,boll_ub,boll_lb,"
          "rsi_30,cci_30,dx_30,close_30_sma,close_60_sma")
FEATS = ["open","high","low","close","volume","pct_change","macd","boll_ub","boll_lb",
         "rsi_30","cci_30","dx_30","close_30_sma","close_60_sma"]
SPLITS = ["train", "valid", "test"]
DATE_ROW = re.compile(r"^\s*(\d{4}-\d{2}-\d{2}),")

# %% Cell 1: load and INSPECT the raw format
from datasets import load_dataset
ds = load_dataset("raeidsaqur/NIFTY")
print(ds)
r = ds["train"][0]
print("keys:", list(r.keys()))
print("context col:", repr(r["context"][:200]))
print("PROMPT (first 1800 chars):\n", repr(r["conversations"][0]["value"][:1800]))
print("NEWS (first 500 chars):\n", repr(r["news"][:500]))
print("label / pct_change:", r["label"], r["pct_change"])

# %% Cell 2: parsing helpers
def get_prompt(r):
    return next(c["value"] for c in r["conversations"] if c["role"] == "user")

def parse_context(prompt, header=HEADER):
    rows = [l.strip() for l in prompt.splitlines() if DATE_ROW.match(l)]
    if not rows:
        return None
    df_temp = pd.read_csv(io.StringIO(header + "\n" + "\n".join(rows)))
    df_temp["date"] = pd.to_datetime(df_temp["date"])
    return df_temp.sort_values("date").reset_index(drop=True)

def label_from_pct(p, thr=0.005):
    return "Rise" if p > thr else ("Fall" if p < -thr else "Neutral")

# %% Cell 3: build the clean table
def build(ds):
    recs, ctxs, bad = [], [], []
    for split in SPLITS:
        for r in ds[split]:
            try:
                c = parse_context(get_prompt(r), HEADER)
            except Exception as e:
                c, err = None, str(e)
            if c is None or len(c) == 0:
                bad.append(r["id"]); continue
            heads = [h.strip() for h in r["news"].split("\n") if h.strip()]
            recs.append(dict(id=r["id"], date=pd.to_datetime(r["date"]), split=split,
                             label=r["label"], pct_change=float(r["pct_change"]),
                             news="\n".join(heads), n_headlines=len(heads),
                             ctx_len=len(c), ctx_first=c["date"].iloc[0], ctx_last=c["date"].iloc[-1]))
            ctxs.append(c)
    if bad:
        raise RuntimeError(f"Could not parse market context for {len(bad)} rows, e.g. {bad[:5]}. "
                           "Paste the Cell 1 output to your assistant so the parser can be fixed.")
    df_temp = pd.DataFrame(recs)
    return df_temp, ctxs

df, ctxs = build(ds)
print(df.shape); print(df.groupby("split").date.agg(["min", "max", "count"]))
print("mean headlines per day:", df.n_headlines.mean(),
      "(about 1 means the headline separator is not a newline: check Cell 1)")

# %% Cell 4: timing audit
assert (df.ctx_last < df.date).all(), "LEAK: some context windows include the target day"
print("OK: latest input date is strictly before target date for all", len(df), "rows")
print("context lengths:", df.ctx_len.value_counts().to_dict())
print("gap (calendar days) between last input and target:", (df.date - df.ctx_last).dt.days.value_counts().sort_index().to_dict())
for i in np.random.default_rng(0).choice(len(df), 5, replace=False):
    print(f"target {df.date[i].date()}  | last input {df.ctx_last[i].date()}  | label {df.label[i]}  | pct {df['pct_change'][i]:+.4f}")

mism = (df["pct_change"].map(label_from_pct) != df.label)
print("label vs +/-0.5% rule mismatches:", int(mism.sum()), "of", len(df))

# %% Cell 5: daily series rebuilt from the context windows + contiguity check
daily = (pd.concat(ctxs).drop_duplicates("date").sort_values("date").reset_index(drop=True))
print("unique trading days recovered from contexts:", len(daily), "|", daily.date.min().date(), "to", daily.date.max().date())
from pandas.tseries.holiday import USFederalHolidayCalendar
hol = USFederalHolidayCalendar().holidays(daily.date.min(), daily.date.max())
expected = pd.bdate_range(daily.date.min(), daily.date.max()).difference(hol)
missing = expected.difference(pd.DatetimeIndex(daily.date))
print(f"expected (approx) trading days: {len(expected)} | recovered: {len(daily)} | missing: {len(missing)}")
print("sample missing days:", [d.date().isoformat() for d in missing[:15]])
td = df.sort_values("date").date
print("target rows:", len(td), "vs approx trading days in the same range:",
      len(pd.bdate_range(td.min(), td.max()).difference(USFederalHolidayCalendar().holidays(td.min(), td.max()))))
gaps = pd.Series(np.busday_count(td.values[:-1].astype("datetime64[D]"), td.values[1:].astype("datetime64[D]")))
print("target-date gaps in business days (1 = consecutive):", gaps.value_counts().sort_index().to_dict())
m = df.merge(daily[["date", "pct_change"]].rename(columns={"pct_change": "pct_ctx"}), on="date", how="inner")
if len(m):
    print(f"target pct_change vs context pct_change on {len(m)} overlapping dates: "
          f"share within 1e-3 = {(np.abs(m['pct_change'] - m['pct_ctx']) < 1e-3).mean():.3f}")

# %% Cell 5b: DIAGNOSTIC
allc = pd.concat(ctxs)
print("max disagreement in close across duplicate context rows:", allc.groupby("date")["close"].agg(lambda s: s.max() - s.min()).max())
d = daily.set_index("date")
d["ret_cc"] = d["close"].pct_change()
d["ret_oc"] = d["close"] / d["open"] - 1
cmp = pd.DataFrame({"target": df.set_index("date")["pct_change"]})
cmp["cc_same_day"] = d["ret_cc"].reindex(cmp.index)
cmp["cc_next_day"] = d["ret_cc"].shift(-1).reindex(cmp.index)
cmp["cc_prev_day"] = d["ret_cc"].shift(1).reindex(cmp.index)
cmp["oc_same_day"] = d["ret_oc"].reindex(cmp.index)
cmp["ctx_pct_same_day"] = d["pct_change"].reindex(cmp.index)
print(cmp.corr()["target"].round(3))
for c in cmp.columns[1:]:
    print(f"{c:18s} share within 1e-3 of target: {(np.abs(cmp['target'] - cmp[c]) < 1e-3).mean():.3f}")
print(cmp.describe().loc[["mean", "std", "min", "max"]].round(4))
print(cmp.head(6).round(4))
print("label-rule mismatches:"); print(df.loc[mism, ["date", "label", "pct_change"]])

# %% Cell 6: lagged windows
LAG = 0
REGIME_WINDOW = 20
daily = daily.sort_values("date").reset_index(drop=True)
daily["ret"] = daily["close"].pct_change()
FEATS2 = [f if f != "pct_change" else "ret" for f in FEATS]
dts = df["date"].values
nxt = daily["date"].searchsorted(dts, side="right")
pos = nxt - LAG
keep = pos >= REGIME_WINDOW + 1
print(f"dropped {int((~keep).sum())} early rows without 20 days of history:", df.loc[~keep, "date"].dt.date.astype(str).tolist())
df = df[keep].reset_index(drop=True); pos = pos[keep]; nxt = nxt[keep]
X = np.stack([daily.iloc[p - LOOKBACK:p][FEATS2].to_numpy(dtype=float) for p in pos])
last_in = pd.DatetimeIndex([daily["date"].iloc[p - 1] for p in pos])
first_in = pd.DatetimeIndex([daily["date"].iloc[p - LOOKBACK] for p in pos])
tgt = pd.DatetimeIndex(df["date"])
assert (last_in <= tgt).all() and (LAG == 0 or (last_in < tgt).all()), "LEAK: window extends past the allowed date"
has_end = nxt < len(daily)
lab_end = daily["date"].to_numpy()[np.minimum(nxt, len(daily) - 1)]
assert (last_in.values[has_end] < lab_end[has_end]).all(), "LEAK: window reaches the day the label move ends on"
print("X shape:", X.shape, "| NaNs:", int(np.isnan(X).sum()))
y = df.label.map({"Fall": 0, "Neutral": 1, "Rise": 2}).to_numpy()
df.to_parquet(OUT / "clean.parquet"); daily.to_parquet(OUT / "daily_series.parquet")
np.save(OUT / "X_price_windows.npy", X); np.save(OUT / "y.npy", y)
print("saved to", OUT)

# %% Cell 7: EDA
print(pd.crosstab(df.split, df.label, normalize="index").round(3))
print(df.groupby("split").n_headlines.describe()[["mean", "min", "max"]])
print("missing values per column:\n", df.isna().sum()[lambda s: s > 0])
fig, ax = plt.subplots(1, 3, figsize=(15, 3.6))
pd.crosstab(df.split, df.label).loc[SPLITS].plot.bar(ax=ax[0], rot=0); ax[0].set_title("Class counts by split")
ax[1].plot(df.date, df.n_headlines, lw=.6); ax[1].set_title("Headlines per day")
ax[2].plot(daily.date, daily.close, lw=.8); ax[2].set_title("SPY close (recovered from contexts)")
for s, c in zip(["valid", "test"], ["orange", "red"]):
    ax[2].axvline(df[df.split == s].date.min(), color=c, ls="--", lw=.8)
plt.tight_layout(); plt.savefig(FIG / "eda_overview.png", dpi=200); plt.show()
