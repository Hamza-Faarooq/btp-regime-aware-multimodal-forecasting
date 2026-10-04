# Step 4 — Models, Training & Evaluation
# Generated from the attached BTP.ipynb; execute sequentially in one Colab runtime.

# ===== Original notebook cell 5 =====
import copy, time, warnings
import numpy as np, pandas as pd, torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score
warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path("/content/drive/MyDrive/btp"); DATA = ROOT / "data"; RES = ROOT / "results"; RES.mkdir(exist_ok=True)
SEEDS = list(range(10))
LOOKBACK, LAG = 10, 0
H, DROPOUT, NEWS_OUT, REG_EMB = 64, 0.2, 64, 8
LR, WD, BATCH, MAX_EPOCHS, PATIENCE = 1e-3, 1e-4, 64, 50, 8
CLIP = 5.0
SPLITS = ["train", "valid", "test"]

# %% Cell 1
df = pd.read_parquet(DATA / "clean.parquet").reset_index(drop=True)
daily = pd.read_parquet(DATA / "daily_series.parquet").sort_values("date").reset_index(drop=True)
reg = pd.read_parquet(DATA / "regimes.parquet").set_index("id").loc[df["id"]].reset_index()
news_emb = np.load(DATA / "news_emb.npy"); news_sent = np.load(DATA / "news_sent.npy"); meta = pd.read_parquet(DATA / "news_meta.parquet")
assert len(news_emb) == len(df) == len(news_sent) and (meta["id"].values == df["id"].values).all()
y = np.load(DATA / "y.npy")
assert (y == df["label"].map({"Fall": 0, "Neutral": 1, "Rise": 2}).to_numpy()).all()

d = daily; f = pd.DataFrame({"date": d["date"]})
f["ret"] = d["close"].pct_change(); f["oc"] = d["close"] / d["open"] - 1; f["hl"] = (d["high"] - d["low"]) / d["close"]
lv = np.log(d["volume"].replace(0, np.nan)); f["lvol_rel"] = lv - lv.rolling(20, min_periods=5).mean()
f["macd_n"] = d["macd"] / d["close"]; f["rsi"] = d["rsi_30"] / 100; f["cci"] = d["cci_30"] / 100; f["dx"] = d["dx_30"] / 100
f["bollpos"] = (d["close"] - d["boll_lb"]) / (d["boll_ub"] - d["boll_lb"]); f["bollw"] = (d["boll_ub"] - d["boll_lb"]) / d["close"]
f["c_sma30"] = d["close"] / d["close_30_sma"] - 1; f["c_sma60"] = d["close"] / d["close_60_sma"] - 1
FEAT = [c for c in f.columns if c != "date"]; F = f[FEAT].replace([np.inf, -np.inf], np.nan).to_numpy(float)
idx = daily["date"].searchsorted(df["date"].values, side="right") - 1 - LAG
assert (daily["date"].to_numpy()[idx] <= df["date"].to_numpy()).all()
X = np.stack([F[i - LOOKBACK + 1:i + 1] for i in idx])
bad = np.isnan(X).any(axis=(0, 1)); assert not bad.any(), f"NaNs in features: {[c for c, b in zip(FEAT, bad) if b]}"
X_old = np.load(DATA / "X_price_windows.npy"); assert np.allclose(X[:, :, 0], X_old[:, :, 5], atol=1e-6)
print("X", X.shape, "| features:", FEAT)

# %% Cell 2: protocols
def get_split(protocol, seed):
    if protocol in ("P2", "P3"): return df["split"].to_numpy()
    rng = np.random.default_rng(1000 + seed); perm = rng.permutation(len(df))
    n_tr = int((df["split"] == "train").sum()); n_va = int((df["split"] == "valid").sum())
    sp = np.empty(len(df), dtype=object); sp[perm[:n_tr]]="train"; sp[perm[n_tr:n_tr+n_va]]="valid"; sp[perm[n_tr+n_va:]]="test"; return sp

def prepare(protocol, seed):
    sp = get_split(protocol, seed); fit = (sp == "train") if protocol == "P3" else np.ones(len(sp), bool)
    flat = X[fit].reshape(-1, X.shape[-1]); mu, sd = flat.mean(0), flat.std(0)+1e-8; Xs=np.clip((X-mu)/sd,-CLIP,CLIP).astype(np.float32)
    nm, ns = news_emb[fit].mean(0), news_emb[fit].std(0)+1e-8; Ns=np.clip((news_emb-nm)/ns,-CLIP,CLIP).astype(np.float32)
    sm, sdv = news_sent[fit].mean(0), news_sent[fit].std(0)+1e-8; Ss=np.clip((news_sent-sm)/sdv,-CLIP,CLIP).astype(np.float32)
    R=reg["regime_p3" if protocol=="P3" else "regime_all"].to_numpy().astype(np.int64); return sp,Xs,Ns,Ss,R

# %% Cell 3: models and training
MODELS={"M1":(1,0,0),"M2":(1,1,0),"M3":(1,1,1),"M4":(0,1,0),"M5":(0,0,1),"M6":(1,1,0)}
NEWS_SRC={"M6":"sent"}
class Net(nn.Module):
    def __init__(self,use_p,use_n,use_r,nfeat,ndim):
        super().__init__(); self.use_p,self.use_n,self.use_r=use_p,use_n,use_r; d=0
        if use_p:self.lstm=nn.LSTM(nfeat,H,batch_first=True); d+=H
        if use_n:self.news=nn.Sequential(nn.Linear(ndim,NEWS_OUT),nn.ReLU(),nn.Dropout(DROPOUT)); d+=NEWS_OUT
        if use_r:self.emb=nn.Embedding(3,REG_EMB); d+=REG_EMB
        self.drop=nn.Dropout(DROPOUT); self.head=nn.Sequential(nn.Linear(d,64),nn.ReLU(),nn.Dropout(DROPOUT),nn.Linear(64,3))
    def forward(self,x,n,r):
        parts=[]
        if self.use_p:_,(h,_)=self.lstm(x);parts.append(self.drop(h[-1]))
        if self.use_n:parts.append(self.news(n))
        if self.use_r:parts.append(self.emb(r))
        return self.head(torch.cat(parts,1))
def set_seed(s): np.random.seed(s);torch.manual_seed(s)
def score(yt,yp): return dict(accuracy=accuracy_score(yt,yp),macro_f1=f1_score(yt,yp,average="macro",labels=[0,1,2],zero_division=0),balanced_acc=balanced_accuracy_score(yt,yp))
def run(model_id,protocol,seed):
    set_seed(seed);sp,Xs,Ns,Ss,R=prepare(protocol,seed);ix={s:np.where(sp==s)[0] for s in SPLITS};ytr=y[ix["train"]]
    if model_id=="M0": proba=np.zeros((len(y),3),np.float32);proba[:,np.bincount(ytr,minlength=3).argmax()]=1.0
    elif model_id=="M0r": proba=np.eye(3,dtype=np.float32)[np.random.default_rng(seed).integers(0,3,len(y))]
    else:
        Nin=Ss if NEWS_SRC.get(model_id,"emb")=="sent" else Ns;T=torch.as_tensor;xt,nt,rt,yt=T(Xs),T(Nin),T(R),T(y)
        cw=torch.tensor(len(ytr)/(3*np.maximum(np.bincount(ytr,minlength=3),1)),dtype=torch.float32)
        net=Net(*MODELS[model_id],Xs.shape[-1],Nin.shape[1]);opt=torch.optim.Adam(net.parameters(),lr=LR,weight_decay=WD);lossf=nn.CrossEntropyLoss(weight=cw);rng=np.random.default_rng(seed)
        best_f1,best_state,bad=-1.0,None,0
        for ep in range(MAX_EPOCHS):
            net.train();perm=rng.permutation(ix["train"])
            for i in range(0,len(perm),BATCH):
                b=T(perm[i:i+BATCH]);opt.zero_grad();lossf(net(xt[b],nt[b],rt[b]),yt[b]).backward();opt.step()
            net.eval();v=T(ix["valid"])
            with torch.no_grad():pv=net(xt[v],nt[v],rt[v]).argmax(1).numpy()
            f1=f1_score(y[ix["valid"]],pv,average="macro",labels=[0,1,2],zero_division=0)
            if f1>best_f1:best_f1,best_state,bad=f1,copy.deepcopy(net.state_dict()),0
            else:bad+=1
            if bad>=PATIENCE:break
        net.load_state_dict(best_state);net.eval()
        with torch.no_grad():proba=torch.softmax(net(xt,nt,rt),1).numpy()
    pred=proba.argmax(1);mrows=[dict(protocol=protocol,model=model_id,seed=seed,eval_split=s,n=len(ix[s]),**score(y[ix[s]],pred[ix[s]])) for s in SPLITS]
    prow=pd.DataFrame(dict(protocol=protocol,model=model_id,seed=seed,id=df["id"],date=df["date"],part=sp,y_true=y,y_pred=pred,p_fall=proba[:,0],p_neutral=proba[:,1],p_rise=proba[:,2],regime_used=R,regime_p3=reg["regime_p3"].to_numpy()))
    return mrows,prow

# %% Cell 4
def run_grid(protocols,models=("M0","M0r","M1","M2","M3","M4","M5","M6"),seeds=SEEDS):
    mrows,preds,t0=[],[],time.time()
    for p in protocols:
        for m in models:
            for s in seeds:
                r,pr=run(m,p,s);mrows+=r;preds.append(pr);te=[x for x in r if x["eval_split"]=="test"][0]
                print(f"{p} {m} seed{s}: test macro-F1 {te['macro_f1']:.3f} acc {te['accuracy']:.3f}  ({time.time()-t0:.0f}s)")
    tag="_".join(protocols);M=pd.DataFrame(mrows);P=pd.concat(preds,ignore_index=True)
    M.to_csv(RES/f"metrics_{tag}.csv",index=False);P.to_parquet(RES/f"predictions_{tag}.parquet");return M,P
def summarize(M,protocol,split="test"):
    t=M[(M.protocol==protocol)&(M.eval_split==split)];return t.groupby("model")[["macro_f1","accuracy","balanced_acc"]].agg(["mean","std"]).round(3)

M_all,P_all=run_grid(["P1","P2","P3"])
for p in ["P3","P2","P1"]: print(p,"TEST\n",summarize(M_all,p,"test"))
print("P3 VALID\n",summarize(M_all,"P3","valid"));print("P3 TRAIN\n",summarize(M_all,"P3","train"))
