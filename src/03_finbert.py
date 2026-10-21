# =====================================================================
# BTP Step 3: frozen FinBERT embeddings of each day's headlines (run ONCE, then never again)
# Colab: switch runtime to a T4 GPU first (Runtime > Change runtime type), then re-run the
# Drive-mount and pip-install cells from Step 0 (the runtime restarts).
# Cells are separated by "# %%".
# =====================================================================

# %% Cell 0: imports and config
import time
import numpy as np, pandas as pd, torch
from pathlib import Path

import os
BTP_ROOT = Path(os.environ.get("BTP_ROOT", "/content/drive/MyDrive/btp"))   # set BTP_ROOT to run outside Colab
ROOT = BTP_ROOT; DATA = ROOT / "data"
CAP = 50              # max headlines embedded per day (random sample, fixed seed)
SAMPLE_SEED = 1234
MAX_LEN = 64          # tokens per headline
BS = 256              # batch size (lower to 64 if the GPU runs out of memory)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device, "(CPU works but is slow; consider CAP = 20)" if device == "cpu" else "")

# %% Cell 1: choose the headlines to embed
df = pd.read_parquet(DATA / "clean.parquet").reset_index(drop=True)   # same row order as the price windows
rng = np.random.default_rng(SAMPLE_SEED)
rows, texts = [], []
n_used = np.zeros(len(df), dtype=int)
for i, txt in enumerate(df["news"]):
    hs = [h.strip() for h in txt.split("\n") if h.strip()]
    if len(hs) > CAP:                                   # cap very heavy news days; sample is random, not "first 50"
        keep = np.sort(rng.choice(len(hs), CAP, replace=False)); hs = [hs[j] for j in keep]
    n_used[i] = len(hs); rows += [i] * len(hs); texts += hs
rows = np.array(rows)
print(len(texts), "headlines to embed | days with 0 headlines:", int((n_used == 0).sum()))
print("mean headlines used per day, by split:", pd.Series(n_used).groupby(df["split"].values).mean().round(1).to_dict())

# %% Cell 2: load frozen FinBERT
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device).eval()
print("label order:", model.config.id2label)

# %% Cell 3: embed (one pass gives both a 768-d embedding and 3 sentiment probabilities)
@torch.no_grad()
def encode(texts):
    order = np.argsort([len(t) for t in texts])           # similar lengths together = less padding
    emb = np.zeros((len(texts), 768), np.float32); sent = np.zeros((len(texts), 3), np.float32)
    for s in range(0, len(texts), BS):
        ids = order[s:s + BS]
        b = tok([texts[j] for j in ids], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(device)
        with torch.autocast(device_type=device, dtype=torch.float16 if device == "cuda" else torch.bfloat16, enabled=(device == "cuda")):
            out = model(**b, output_hidden_states=True)
        h = out.hidden_states[-1].float(); m = b["attention_mask"].unsqueeze(-1).float()
        emb[ids] = ((h * m).sum(1) / m.sum(1)).cpu().numpy()             # mean over tokens (padding masked)
        sent[ids] = torch.softmax(out.logits.float(), -1).cpu().numpy()
    return emb, sent

t0 = time.time()
emb_h, sent_h = encode(texts)
print(f"embedded {len(texts)} headlines in {time.time() - t0:.0f}s")

# %% Cell 4: average headlines per day, save
N = len(df)
news_emb = pd.DataFrame(emb_h).groupby(rows).mean().reindex(range(N)).fillna(0).to_numpy(np.float32)
news_sent = pd.DataFrame(sent_h).groupby(rows).mean().reindex(range(N)).fillna(0).to_numpy(np.float32)
assert news_emb.shape == (N, 768) and news_sent.shape == (N, 3)

np.save(DATA / "news_emb.npy", news_emb); np.save(DATA / "news_sent.npy", news_sent)
pd.DataFrame({"id": df["id"], "n_headlines": df["n_headlines"], "n_used": n_used}).to_parquet(DATA / "news_meta.parquet")
pd.Series(model.config.id2label).to_json(DATA / "finbert_label_order.json")
print("saved news_emb.npy", news_emb.shape, "| news_sent.npy", news_sent.shape)

# %% Cell 5: sanity checks (paste this output back)
print("NaNs:", int(np.isnan(news_emb).sum()), "| mean L2 norm:", float(np.linalg.norm(news_emb, axis=1).mean().round(2)))
lab = {v.lower(): k for k, v in model.config.id2label.items()}
senti = pd.Series(news_sent[:, lab["positive"]] - news_sent[:, lab["negative"]], index=df["date"].values)
print("mean (positive - negative) FinBERT sentiment by split:", senti.groupby(df["split"].values).mean().round(3).to_dict())
ym = senti.index.strftime("%Y-%m").to_numpy(); is_mar20 = ym == "2020-03"; is_test = (df["split"] == "test").to_numpy()
print("Mar 2020 mean sentiment:", round(senti[is_mar20].mean(), 3), "| rest of test:", round(senti[is_test & ~is_mar20].mean(), 3))
# do two different days look different? (cosine similarity of consecutive days vs random pairs)
cs = lambda a, b: float((a * b).sum(1).mean() / (np.linalg.norm(a, axis=1).mean() * np.linalg.norm(b, axis=1).mean()))
perm = np.random.default_rng(0).permutation(N)
print("avg cosine, consecutive days:", round(cs(news_emb[:-1], news_emb[1:]), 3), "| random pairs:", round(cs(news_emb, news_emb[perm]), 3))
