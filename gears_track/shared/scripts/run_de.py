#!/usr/bin/env python
"""DE computation only (skip GEARS's in-process pyg build)."""
import time
import anndata as ad
from gears.data_utils import get_DE_genes, get_dropout_non_zero_genes

t0 = time.time()
adata = ad.read_h5ad("/workspace/data/gwps_k562_raw10.h5ad")
print(f"[raw] {adata.shape} ({time.time()-t0:.0f}s)", flush=True)
adata = get_DE_genes(adata)
print(f"[de] DE genes done ({time.time()-t0:.0f}s)", flush=True)
adata = get_dropout_non_zero_genes(adata)
print(f"[de] dropout genes done ({time.time()-t0:.0f}s)", flush=True)
import os
os.makedirs("/workspace/data/gwps_k562", exist_ok=True)
adata.write_h5ad("/workspace/data/gwps_k562/perturb_processed.h5ad")
print(f"[de] written ({time.time()-t0:.0f}s)", flush=True)
