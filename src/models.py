"""Model definitions used in the executed BTP experiment."""
from __future__ import annotations
import torch
import torch.nn as nn

class MultimodalNet(nn.Module):
    """Small LSTM + optional news/regime fusion network matching the working notebook."""
    def __init__(self,use_price,use_news,use_regime,n_price_features,news_dim,hidden=64,news_out=64,regime_embed_dim=8,dropout=0.2):
        super().__init__(); self.use_price=use_price; self.use_news=use_news; self.use_regime=use_regime
        d=0
        if use_price: self.lstm=nn.LSTM(n_price_features,hidden,batch_first=True); d+=hidden
        if use_news: self.news=nn.Sequential(nn.Linear(news_dim,news_out),nn.ReLU(),nn.Dropout(dropout)); d+=news_out
        if use_regime: self.emb=nn.Embedding(3,regime_embed_dim); d+=regime_embed_dim
        self.drop=nn.Dropout(dropout)
        self.head=nn.Sequential(nn.Linear(d,64),nn.ReLU(),nn.Dropout(dropout),nn.Linear(64,3))
    def forward(self,x,n,r):
        parts=[]
        if self.use_price: _,(h,_)=self.lstm(x); parts.append(self.drop(h[-1]))
        if self.use_news: parts.append(self.news(n))
        if self.use_regime: parts.append(self.emb(r))
        return self.head(torch.cat(parts,1))

MODEL_SPECS={
    "M1":(1,0,0,"emb"),"M2":(1,1,0,"emb"),"M3":(1,1,1,"emb"),
    "M4":(0,1,0,"emb"),"M5":(0,0,1,"emb"),"M6":(1,1,0,"sent")
}
