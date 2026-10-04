# Step 3 — Frozen FinBERT News Embeddings
import time
import numpy as np, pandas as pd, torch
from pathlib import Path
ROOT=Path("/content/drive/MyDrive/btp"); DATA=ROOT/"data"
CAP=50; SAMPLE_SEED=1234; MAX_LEN=64; BS=256
device="cuda" if torch.cuda.is_available() else "cpu"
df=pd.read_parquet(DATA/"clean.parquet").reset_index(drop=True)
rng=np.random.default_rng(SAMPLE_SEED); rows=[]; texts=[]; n_used=np.zeros(len(df),dtype=int)
for i,txt in enumerate(df["news"]):
    hs=[h.strip() for h in txt.split("\n") if h.strip()]
    if len(hs)>CAP:
        keep=np.sort(rng.choice(len(hs),CAP,replace=False));hs=[hs[j] for j in keep]
    n_used[i]=len(hs);rows += [i]*len(hs);texts += hs
rows=np.array(rows)
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tok=AutoTokenizer.from_pretrained("ProsusAI/finbert")
model=AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device).eval()
@torch.no_grad()
def encode(texts):
    order=np.argsort([len(t) for t in texts]);emb=np.zeros((len(texts),768),np.float32);sent=np.zeros((len(texts),3),np.float32)
    for s in range(0,len(texts),BS):
        ids=order[s:s+BS];b=tok([texts[j] for j in ids],padding=True,truncation=True,max_length=MAX_LEN,return_tensors="pt").to(device)
        out=model(**b,output_hidden_states=True);h=out.hidden_states[-1].float();m=b["attention_mask"].unsqueeze(-1).float()
        emb[ids]=((h*m).sum(1)/m.sum(1)).cpu().numpy();sent[ids]=torch.softmax(out.logits.float(),-1).cpu().numpy()
    return emb,sent
t0=time.time();emb_h,sent_h=encode(texts);print(f"embedded {len(texts)} headlines in {time.time()-t0:.0f}s")
N=len(df);news_emb=pd.DataFrame(emb_h).groupby(rows).mean().reindex(range(N)).fillna(0).to_numpy(np.float32);news_sent=pd.DataFrame(sent_h).groupby(rows).mean().reindex(range(N)).fillna(0).to_numpy(np.float32)
np.save(DATA/"news_emb.npy",news_emb);np.save(DATA/"news_sent.npy",news_sent)
pd.DataFrame({"id":df["id"],"n_headlines":df["n_headlines"],"n_used":n_used}).to_parquet(DATA/"news_meta.parquet")
pd.Series(model.config.id2label).to_json(DATA/"finbert_label_order.json")
print("saved news_emb.npy",news_emb.shape,"| news_sent.npy",news_sent.shape)
