#!/usr/bin/env python
"""Subsample the 15 cells/pert raw h5ad to 10 cells/pert + 2000 ctrl."""
import time
import anndata as ad
import numpy as np

t0 = time.time()
adata = ad.read_h5ad("/workspace/data/gwps_k562_raw.h5ad")
print(f"[raw] {adata.shape} ({time.time()-t0:.0f}s)", flush=True)
rng = np.random.default_rng(42)
cond = adata.obs["condition"].values
keep = np.zeros(len(cond), dtype=bool)
for c in np.unique(cond):
    rows = np.where(cond == c)[0]
    n_keep = 2000 if c == "ctrl" else min(10, len(rows))
    keep[rng.choice(rows, size=n_keep, replace=False)] = True
sub = adata[keep].copy()
print(f"[sub] {sub.shape} ({time.time()-t0:.0f}s)", flush=True)
sub.write_h5ad("/workspace/data/gwps_k562_raw10.h5ad")
print(f"[done] ({time.time()-t0:.0f}s)", flush=True)
