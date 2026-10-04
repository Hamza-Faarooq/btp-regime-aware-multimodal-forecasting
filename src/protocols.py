"""Evaluation protocol helpers for the BTP audit."""
from __future__ import annotations
import numpy as np

def get_split(official_split, protocol, seed, train_size, valid_size):
    official_split=np.asarray(official_split)
    if protocol in ("P2","P3"): return official_split.copy()
    rng=np.random.default_rng(1000+seed); perm=rng.permutation(len(official_split))
    out=np.empty(len(official_split),dtype=object)
    out[perm[:train_size]]="train"; out[perm[train_size:train_size+valid_size]]="valid"; out[perm[train_size+valid_size:]]="test"
    return out

def preprocessing_fit_mask(split_membership, protocol):
    sp=np.asarray(split_membership)
    return (sp=="train") if protocol=="P3" else np.ones(len(sp),dtype=bool)
