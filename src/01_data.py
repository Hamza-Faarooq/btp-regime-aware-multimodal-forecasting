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

LOOKBACK = 10                      # previous trading days fed to the LSTM
import os
BTP_ROOT = Path(os.environ.get("BTP_ROOT", "/content/drive/MyDrive/btp"))   # set BTP_ROOT to run outside Colab
OUT = BTP_ROOT / "data"; OUT.mkdir(parents=True, exist_ok=True)
FIG = BTP_ROOT / "figures"; FIG.mkdir(parents=True, exist_ok=True)

HEADER = ("date,open,high,low,close,adj_close,volume,pct_change,macd,boll_ub,boll_lb,"
          "rsi_30,cci_30,dx_30,close_30_sma,close_60_sma")
# Model inputs: adj_close dropped (duplicate of close). pct_change inside the context
# is the PREVIOUS days' return, which is allowed.
FEATS = ["open","high","low","close","volume","pct_change","macd","boll_ub","boll_lb",
         "rsi_30","cci_30","dx_30","close_30_sma","close_60_sma"]
SPLITS = ["train", "valid", "test"]
DATE_ROW = re.compile(r"^\s*(\d{4}-\d{2}-\d{2}),")

# %% Cell 1: load and INSPECT the raw format (paste this cell's output back to your assistant)
from datasets import load_dataset
ds = load_dataset("raeidsaqur/NIFTY")      # if it fails, try revision="refs/convert/parquet"
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
    """Pull the dated market rows out of the prompt text."""
    rows = [l.strip() for l in prompt.splitlines() if DATE_ROW.match(l)]
    if not rows:
        return None
    df = pd.read_csv(io.StringIO(header + "\n" + "\n".join(rows)))
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)

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
    df = pd.DataFrame(recs)
    return df, ctxs

df, ctxs = build(ds)
print(df.shape); print(df.groupby("split").date.agg(["min", "max", "count"]))
print("mean headlines per day:", df.n_headlines.mean(),
      "(about 1 means the headline separator is not a newline: check Cell 1)")

# %% Cell 4: timing audit (the leakage guard)
# 4a. every input window must end strictly before the target date
assert (df.ctx_last < df.date).all(), "LEAK: some context windows include the target day"
print("OK: latest input date is strictly before target date for all", len(df), "rows")
print("context lengths:", df.ctx_len.value_counts().to_dict())
print("gap (calendar days) between last input and target:", (df.date - df.ctx_last).dt.days.value_counts().sort_index().to_dict())
for i in np.random.default_rng(0).choice(len(df), 5, replace=False):
    print(f"target {df.date[i].date()}  | last input {df.ctx_last[i].date()}  | label {df.label[i]}  | pct {df['pct_change'][i]:+.4f}")

# 4b. label rule check (card says Neutral if |change| < 0.50%)
mism = (df["pct_change"].map(label_from_pct) != df.label)
print("label vs +/-0.5% rule mismatches:", int(mism.sum()), "of", len(df))

# %% Cell 5: daily series rebuilt from the context windows + contiguity check
daily = (pd.concat(ctxs).drop_duplicates("date").sort_values("date").reset_index(drop=True))
print("unique trading days recovered from contexts:", len(daily), "|", daily.date.min().date(), "to", daily.date.max().date())

# approximate expected trading days: business days minus US federal holidays (approximation only!)
from pandas.tseries.holiday import USFederalHolidayCalendar
hol = USFederalHolidayCalendar().holidays(daily.date.min(), daily.date.max())
expected = pd.bdate_range(daily.date.min(), daily.date.max()).difference(hol)
missing = expected.difference(pd.DatetimeIndex(daily.date))
print(f"expected (approx) trading days: {len(expected)} | recovered: {len(daily)} | missing: {len(missing)}")
print("sample missing days:", [d.date().isoformat() for d in missing[:15]])

# how gappy are the TARGET dates themselves?
td = df.sort_values("date").date
print("target rows:", len(td), "vs approx trading days in the same range:",
      len(pd.bdate_range(td.min(), td.max()).difference(USFederalHolidayCalendar().holidays(td.min(), td.max()))))
gaps = pd.Series(np.busday_count(td.values[:-1].astype("datetime64[D]"), td.values[1:].astype("datetime64[D]")))
print("target-date gaps in business days (1 = consecutive):", gaps.value_counts().sort_index().to_dict())

# does the target's pct_change agree with the daily series (same return definition)?
m = df.merge(daily[["date", "pct_change"]].rename(columns={"pct_change": "pct_ctx"}), on="date", how="inner")
if len(m):
    print(f"target pct_change vs context pct_change on {len(m)} overlapping dates: "
          f"share within 1e-3 = {(np.abs(m['pct_change'] - m['pct_ctx']) < 1e-3).mean():.3f}")

# %% Cell 5b: DIAGNOSTIC - what does the label's pct_change actually measure? (paste output back)
allc = pd.concat(ctxs)
print("max disagreement in close across duplicate context rows:", allc.groupby("date")["close"].agg(lambda s: s.max() - s.min()).max())

d = daily.set_index("date")
d["ret_cc"] = d["close"].pct_change()          # close-to-close, from our recovered series
d["ret_oc"] = d["close"] / d["open"] - 1       # open-to-close
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

# %% Cell 6: lagged windows built from the recovered DAILY series (contexts only hold 4-8 rows)
# FINDING from Cell 5b: the label pct_change for date t equals close(t+1)/close(t) - 1, i.e. the NEXT trading day's move.
# So close(t) is known before the move being predicted, and a window ending at t is legitimate.
# (The benchmark's own context stops at t-1, so it leaves day t's market data out. LAG=1 reproduces that convention.)
LAG = 0          # 0: window ends at date t (default) | 1: window ends at t-1 (robustness check)
REGIME_WINDOW = 20
daily = daily.sort_values("date").reset_index(drop=True)
daily["ret"] = daily["close"].pct_change()     # our own close-to-close return, used as the return feature
FEATS2 = [f if f != "pct_change" else "ret" for f in FEATS]

dts = df["date"].values
nxt = daily["date"].searchsorted(dts, side="right")      # first daily row AFTER t = day the label's move ends on
pos = nxt - LAG                                          # window = daily.iloc[pos-LOOKBACK : pos]

# alignment check: label == close-to-close return of the next trading day
okn = nxt < len(daily)
share = (np.abs(daily["ret"].to_numpy()[nxt[okn]] - df["pct_change"].to_numpy()[okn]) < 1e-3).mean()
print(f"label equals next-trading-day close-to-close return on {okn.sum()} rows: share = {share:.4f}")
assert share > 0.99, "label alignment differs from what the pipeline assumes"

keep = pos >= REGIME_WINDOW + 1                          # same rows for every model (enough history for 20-day regime features)
print(f"dropped {int((~keep).sum())} early rows without 20 days of history:", df.loc[~keep, "date"].dt.date.astype(str).tolist())
df = df[keep].reset_index(drop=True); pos = pos[keep]; nxt = nxt[keep]

X = np.stack([daily.iloc[p - LOOKBACK:p][FEATS2].to_numpy(dtype=float) for p in pos])
last_in = pd.DatetimeIndex([daily["date"].iloc[p - 1] for p in pos]); first_in = pd.DatetimeIndex([daily["date"].iloc[p - LOOKBACK] for p in pos])
tgt = pd.DatetimeIndex(df["date"])
assert (last_in <= tgt).all() and (LAG == 0 or (last_in < tgt).all()), "LEAK: window extends past the allowed date"
has_end = nxt < len(daily)                               # label end-day row exists in our series (not for the very last dates)
lab_end = daily["date"].to_numpy()[np.minimum(nxt, len(daily) - 1)]
assert (last_in.values[has_end] < lab_end[has_end]).all(), "LEAK: window reaches the day the label move ends on"
print("X shape:", X.shape, "| NaNs:", int(np.isnan(X).sum()), "| max calendar span of a 10-row window (days):", int((last_in - first_in).days.max()))
print("rows per split after dropping:", df.split.value_counts().to_dict())
y = df.label.map({"Fall": 0, "Neutral": 1, "Rise": 2}).to_numpy()

df.to_parquet(OUT / "clean.parquet"); daily.to_parquet(OUT / "daily_series.parquet")
np.save(OUT / "X_price_windows.npy", X); np.save(OUT / "y.npy", y)
print("saved to", OUT)

# %% Cell 7: EDA (class balance, news volume, price history)
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
