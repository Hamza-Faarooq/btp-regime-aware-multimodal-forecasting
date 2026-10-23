# =====================================================================
# BTP Step 4: models M0-M7 and the training/evaluation harness
# Cells are separated by "# %%". Runs fine on CPU.
#   M0 majority | M0r uniform random | M1 price LSTM | M2 price+news | M3 price+news+regime | M4 news only | M5 regime only
#   M6 price + FinBERT sentiment (3 numbers) -- low-capacity news channel, ADDED after seeing that the 768-d news input overfits
#   M7 regime-conditioned GATED fusion of price and news (the architecture proposed in the original BTP plan).
#      Added after the main results; it was specified (not tuned) before being run. Report it whatever it shows.
# =====================================================================

# %% Cell 0: imports and FROZEN hyperparameters (do not tune on test)
import copy, time, warnings
import numpy as np, pandas as pd, torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score
warnings.filterwarnings("ignore", category=UserWarning)

import os
BTP_ROOT = Path(os.environ.get("BTP_ROOT", "/content/drive/MyDrive/btp"))   # set BTP_ROOT to run outside Colab
ROOT = BTP_ROOT; DATA = ROOT / "data"; RES = ROOT / "results"; RES.mkdir(exist_ok=True)
SEEDS = list(range(10))                    # final analysis uses 10 seeds
LOOKBACK, LAG = 10, 0                      # LAG=0: window ends at target date t (main) | LAG=1: ends at t-1 (robustness); must match step2_regimes.py
SUF = "" if LAG == 0 else f"_lag{LAG}"      # LAG=0 keeps the original file names
H, DROPOUT, NEWS_OUT, REG_EMB = 64, 0.2, 64, 8
LR, WD, BATCH, MAX_EPOCHS, PATIENCE = 1e-3, 1e-4, 64, 50, 8
CLIP = 5.0                                 # standardised inputs are clipped to +/-5 sd (COVID-era extremes)
SPLITS = ["train", "valid", "test"]

# %% Cell 1: load everything and build SCALE-FREE price features
df = pd.read_parquet(DATA / "clean.parquet").reset_index(drop=True)
daily = pd.read_parquet(DATA / "daily_series.parquet").sort_values("date").reset_index(drop=True)
news_emb = np.load(DATA / "news_emb.npy"); news_sent = np.load(DATA / "news_sent.npy"); meta = pd.read_parquet(DATA / "news_meta.parquet")
y = np.load(DATA / "y.npy")
assert len(news_emb) == len(df) == len(news_sent) == len(y) and (meta["id"].values == df["id"].values).all(), "news rows not aligned with clean table"
reg_all = pd.read_parquet(DATA / f"regimes{SUF}.parquet").set_index("id")
keep = df["id"].isin(reg_all.index).to_numpy()            # LAG>=1 can drop the earliest row(s) (no full regime history)
if not keep.all():
    print("dropping", int((~keep).sum()), "row(s) absent from the regime file:", df.loc[~keep, "date"].dt.date.astype(str).tolist())
df = df[keep].reset_index(drop=True); news_emb, news_sent, y = news_emb[keep], news_sent[keep], y[keep]
reg = reg_all.loc[df["id"]].reset_index()
assert (y == df["label"].map({"Fall": 0, "Neutral": 1, "Rise": 2}).to_numpy()).all()

# Raw price levels (open/high/low/close, Bollinger bands, SMAs) drift from ~100 to ~340 over the sample, so test-period
# inputs would sit far outside the training range. We therefore use scale-free versions: returns, ratios, bounded indicators.
d = daily; f = pd.DataFrame({"date": d["date"]})
f["ret"] = d["close"].pct_change()
f["oc"] = d["close"] / d["open"] - 1
f["hl"] = (d["high"] - d["low"]) / d["close"]
lv = np.log(d["volume"].replace(0, np.nan)); f["lvol_rel"] = lv - lv.rolling(20, min_periods=5).mean()
f["macd_n"] = d["macd"] / d["close"]
f["rsi"] = d["rsi_30"] / 100; f["cci"] = d["cci_30"] / 100; f["dx"] = d["dx_30"] / 100
f["bollpos"] = (d["close"] - d["boll_lb"]) / (d["boll_ub"] - d["boll_lb"])
f["bollw"] = (d["boll_ub"] - d["boll_lb"]) / d["close"]
f["c_sma30"] = d["close"] / d["close_30_sma"] - 1; f["c_sma60"] = d["close"] / d["close_60_sma"] - 1
FEAT = [c for c in f.columns if c != "date"]
F = f[FEAT].replace([np.inf, -np.inf], np.nan).to_numpy(float)

idx = daily["date"].searchsorted(df["date"].values, side="right") - 1 - LAG        # last day each window may use
assert (daily["date"].to_numpy()[idx] <= df["date"].to_numpy()).all(), "LEAK"
X = np.stack([F[i - LOOKBACK + 1:i + 1] for i in idx])
bad = np.isnan(X).any(axis=(0, 1))
assert not bad.any(), f"NaNs in features: {[c for c, b in zip(FEAT, bad) if b]}"
if LAG == 0:                                                                        # Step 1 saved windows only for LAG=0
    X_old = np.load(DATA / "X_price_windows.npy")[keep]                             # same rows, raw features
    assert np.allclose(X[:, :, 0], X_old[:, :, 5], atol=1e-6), "window rows differ from Step 1"
print("X", X.shape, "| features:", FEAT)
print("class counts (all rows):", np.bincount(y).tolist(), "| regimes available:", [c for c in reg.columns if c.startswith("regime")])

# %% Cell 2: protocols (the thing the leakage audit varies)
def get_split(protocol, seed):
    if protocol in ("P2", "P3"):                       # official chronological split
        return df["split"].to_numpy()
    rng = np.random.default_rng(1000 + seed)           # P1: random partition, same sizes (deliberately invalid)
    perm = rng.permutation(len(df)); n_tr = int((df["split"] == "train").sum()); n_va = int((df["split"] == "valid").sum())
    sp = np.empty(len(df), dtype=object)
    sp[perm[:n_tr]] = "train"; sp[perm[n_tr:n_tr + n_va]] = "valid"; sp[perm[n_tr + n_va:]] = "test"
    return sp

def prepare(protocol, seed):
    sp = get_split(protocol, seed)
    fit = (sp == "train") if protocol == "P3" else np.ones(len(sp), bool)   # P3: fit scalers on train only; P1/P2: on everything
    flat = X[fit].reshape(-1, X.shape[-1]); mu, sd = flat.mean(0), flat.std(0) + 1e-8
    Xs = np.clip((X - mu) / sd, -CLIP, CLIP).astype(np.float32)
    nm, ns = news_emb[fit].mean(0), news_emb[fit].std(0) + 1e-8
    Ns = np.clip((news_emb - nm) / ns, -CLIP, CLIP).astype(np.float32)
    sm, sdv = news_sent[fit].mean(0), news_sent[fit].std(0) + 1e-8
    Ss = np.clip((news_sent - sm) / sdv, -CLIP, CLIP).astype(np.float32)
    R = reg["regime_p3" if protocol == "P3" else "regime_all"].to_numpy().astype(np.int64)
    return sp, Xs, Ns, Ss, R

# %% Cell 3: models and training
MODELS = {"M1": (1, 0, 0), "M2": (1, 1, 0), "M3": (1, 1, 1), "M4": (0, 1, 0), "M5": (0, 0, 1), "M6": (1, 1, 0)}   # (price, news, regime)
NEWS_SRC = {"M6": "sent"}                                       # every other news model uses the 768-d embedding

class Net(nn.Module):
    def __init__(self, use_p, use_n, use_r, nfeat, ndim):
        super().__init__(); self.use_p, self.use_n, self.use_r = use_p, use_n, use_r; d = 0
        if use_p: self.lstm = nn.LSTM(nfeat, H, batch_first=True); d += H
        if use_n: self.news = nn.Sequential(nn.Linear(ndim, NEWS_OUT), nn.ReLU(), nn.Dropout(DROPOUT)); d += NEWS_OUT
        if use_r: self.emb = nn.Embedding(3, REG_EMB); d += REG_EMB
        self.drop = nn.Dropout(DROPOUT)
        self.head = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Dropout(DROPOUT), nn.Linear(64, 3))
    def forward(self, x, n, r):
        parts = []
        if self.use_p: _, (h, _) = self.lstm(x); parts.append(self.drop(h[-1]))
        if self.use_n: parts.append(self.news(n))
        if self.use_r: parts.append(self.emb(r))
        return self.head(torch.cat(parts, 1))

assert NEWS_OUT == H, "GatedNet mixes the price and news embeddings element-wise, so they need the same width"

class GatedNet(nn.Module):
    """M7: regime-conditioned gated fusion.
         g = sigmoid(W [z_price ; z_news ; e_regime])        (one gate value per embedding dimension)
         z = g * z_price + (1 - g) * z_news                  (g near 1 = lean on price, near 0 = lean on news)
         logits = MLP([z ; e_regime])"""
    def __init__(self, nfeat, ndim):
        super().__init__()
        self.lstm = nn.LSTM(nfeat, H, batch_first=True)
        self.news = nn.Sequential(nn.Linear(ndim, NEWS_OUT), nn.ReLU(), nn.Dropout(DROPOUT))
        self.emb = nn.Embedding(3, REG_EMB)
        self.gate = nn.Linear(H + NEWS_OUT + REG_EMB, H)
        self.drop = nn.Dropout(DROPOUT)
        self.head = nn.Sequential(nn.Linear(H + REG_EMB, 64), nn.ReLU(), nn.Dropout(DROPOUT), nn.Linear(64, 3))
    def forward(self, x, n, r, return_gate=False):
        _, (h, _) = self.lstm(x); zp = self.drop(h[-1]); zn = self.news(n); er = self.emb(r)
        g = torch.sigmoid(self.gate(torch.cat([zp, zn, er], 1)))
        out = self.head(torch.cat([g * zp + (1 - g) * zn, er], 1))
        return (out, g) if return_gate else out

def set_seed(s):
    np.random.seed(s); torch.manual_seed(s)

def score(yt, yp):
    return dict(accuracy=accuracy_score(yt, yp), macro_f1=f1_score(yt, yp, average="macro", labels=[0, 1, 2], zero_division=0),
                balanced_acc=balanced_accuracy_score(yt, yp))

def run(model_id, protocol, seed):
    set_seed(seed); sp, Xs, Ns, Ss, R = prepare(protocol, seed)
    gate = np.full(len(y), np.nan, np.float32)                        # mean price-gate per row (M7 only)
    ix = {s: np.where(sp == s)[0] for s in SPLITS}; ytr = y[ix["train"]]
    if model_id == "M0":                                              # majority class of the training partition
        proba = np.zeros((len(y), 3), np.float32); proba[:, np.bincount(ytr, minlength=3).argmax()] = 1.0
    elif model_id == "M0r":                                           # uniform random guesses: the Macro-F1 floor for a balanced guesser
        proba = np.eye(3, dtype=np.float32)[np.random.default_rng(seed).integers(0, 3, len(y))]
    else:
        Nin = Ss if NEWS_SRC.get(model_id, "emb") == "sent" else Ns
        T = torch.as_tensor; xt, nt, rt, yt = T(Xs), T(Nin), T(R), T(y)
        cw = torch.tensor(len(ytr) / (3 * np.maximum(np.bincount(ytr, minlength=3), 1)), dtype=torch.float32)
        net = GatedNet(Xs.shape[-1], Nin.shape[1]) if model_id == "M7" else Net(*MODELS[model_id], Xs.shape[-1], Nin.shape[1])
        opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WD)
        lossf = nn.CrossEntropyLoss(weight=cw); rng = np.random.default_rng(seed)
        best_f1, best_state, bad = -1.0, None, 0
        for ep in range(MAX_EPOCHS):
            net.train(); perm = rng.permutation(ix["train"])
            for i in range(0, len(perm), BATCH):
                b = T(perm[i:i + BATCH]); opt.zero_grad(); lossf(net(xt[b], nt[b], rt[b]), yt[b]).backward(); opt.step()
            net.eval(); v = T(ix["valid"])
            with torch.no_grad(): pv = net(xt[v], nt[v], rt[v]).argmax(1).numpy()
            f1 = f1_score(y[ix["valid"]], pv, average="macro", labels=[0, 1, 2], zero_division=0)   # early stopping on VALID only
            if f1 > best_f1: best_f1, best_state, bad = f1, copy.deepcopy(net.state_dict()), 0
            else:
                bad += 1
                if bad >= PATIENCE: break
        net.load_state_dict(best_state); net.eval()
        with torch.no_grad():
            if model_id == "M7":
                lg, g = net(xt, nt, rt, return_gate=True); proba = torch.softmax(lg, 1).numpy(); gate = g.mean(1).numpy()
            else:
                proba = torch.softmax(net(xt, nt, rt), 1).numpy()
    pred = proba.argmax(1)
    mrows = [dict(protocol=protocol, model=model_id, seed=seed, eval_split=s, n=len(ix[s]), **score(y[ix[s]], pred[ix[s]])) for s in SPLITS]
    prow = pd.DataFrame(dict(protocol=protocol, model=model_id, seed=seed, id=df["id"], date=df["date"], part=sp, y_true=y, y_pred=pred,
                             p_fall=proba[:, 0], p_neutral=proba[:, 1], p_rise=proba[:, 2], regime_used=R, regime_p3=reg["regime_p3"].to_numpy(), gate_price=gate))
    return mrows, prow

# %% Cell 4: run a grid and summarise
def run_grid(protocols, models=("M0", "M0r", "M1", "M2", "M3", "M4", "M5", "M6"), seeds=SEEDS):
    mrows, preds, t0 = [], [], time.time()
    for p in protocols:
        for m in models:
            for s in seeds:
                r, pr = run(m, p, s); mrows += r; preds.append(pr)
                te = [x for x in r if x["eval_split"] == "test"][0]
                print(f"{p} {m} seed{s}: test macro-F1 {te['macro_f1']:.3f} acc {te['accuracy']:.3f}  ({time.time() - t0:.0f}s)")
    tag = "_".join(protocols) + SUF; M = pd.DataFrame(mrows); P = pd.concat(preds, ignore_index=True)
    M.to_csv(RES / f"metrics_{tag}.csv", index=False); P.to_parquet(RES / f"predictions_{tag}.parquet")
    return M, P

def summarize(M, protocol, split="test"):
    t = M[(M.protocol == protocol) & (M.eval_split == split)]
    g = t.groupby("model")[["macro_f1", "accuracy", "balanced_acc"]].agg(["mean", "std"]).round(3)
    return g

def gate_summary(P, protocol="P3", part="test"):
    """Mean price-gate by regime: per-seed regime means, then mean and sd across seeds."""
    g = P[(P.model == "M7") & (P.protocol == protocol) & (P.part == part)]
    return g.groupby(["seed", "regime_used"]).gate_price.mean().unstack().agg(["mean", "std"]).round(3)

# --- run all three protocols: 10 seeds x 8 models (M0, M0r, M1-M6) x 3 protocols = 240 model-protocol-seed evaluations
# (180 trained networks M1-M6 + 60 non-trained baseline evaluations M0/M0r), ~5 min on CPU in the recorded run.
# M7 is implemented but was never run on real data; pass models=(..., "M7") to run_grid to include it (270 evaluations).
# With LAG=0, M0-M6 reproduce the reported 10-seed numbers; M7 is optional and was not part of the reported results.
if __name__ == "__main__":
    M_all, P_all = run_grid(["P1", "P2", "P3"])
    for p in ["P3", "P2", "P1"]:
        print(p, "TEST\n", summarize(M_all, p, "test"))
    print("P3 VALID\n", summarize(M_all, "P3", "valid")); print("P3 TRAIN\n", summarize(M_all, "P3", "train"))
    if (P_all.model == "M7").any():
        print("M7 gate on P3 test rows (columns = regime 0/1/2 = Low/Mid/High-vol):\n", gate_summary(P_all))
