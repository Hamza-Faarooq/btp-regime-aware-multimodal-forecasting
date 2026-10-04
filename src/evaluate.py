"""Metrics and bootstrap helpers used by the BTP analysis."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score

def classification_scores(y_true,y_pred):
    return {"accuracy":accuracy_score(y_true,y_pred),"macro_f1":f1_score(y_true,y_pred,average="macro",labels=[0,1,2],zero_division=0),"balanced_acc":balanced_accuracy_score(y_true,y_pred)}

def macro_f1_fast(y_true,y_pred):
    y_true=np.asarray(y_true,dtype=int); y_pred=np.asarray(y_pred,dtype=int)
    cm=np.bincount(y_true*3+y_pred,minlength=9).reshape(3,3); tp=np.diag(cm).astype(float); den=cm.sum(0)+cm.sum(1)
    return float(np.mean(np.where(den>0,2*tp/np.maximum(den,1),0.0)))

def bootstrap_metric_diff(y_true,pred_a,pred_b,n_boot=1000,seed=0):
    y_true=np.asarray(y_true); pred_a=np.asarray(pred_a); pred_b=np.asarray(pred_b); rng=np.random.default_rng(seed); out=np.empty(n_boot); n=len(y_true)
    for k in range(n_boot):
        idx=rng.integers(0,n,n); out[k]=macro_f1_fast(y_true[idx],pred_b[idx])-macro_f1_fast(y_true[idx],pred_a[idx])
    lo,hi=np.percentile(out,[2.5,97.5]); return float(out.mean()),float(lo),float(hi)
