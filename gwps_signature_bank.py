import time
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
from pathlib import Path

ROOT = Path.home() / 'vcc2026_cpu'
ARTIFACTS = ROOT / 'artifacts'
ARTIFACTS.mkdir(parents=True, exist_ok=True)

GWPS_PATH = ROOT / 'train' / 'K562_gwps_raw_singlecell_01.h5ad'
SEED = 2026
N_VALID_PER_GROUP = 40
MIN_CELLS_PER_GROUP = 15
SHRINKAGE = 0.65
PSEUDOCOUNT = 0.5
CLIP_LOG_FOLD = 1.0

print("Loading K562_gwps (backed)...", flush=True)
adata = ad.read_h5ad(GWPS_PATH, backed='r')
print("shape:", adata.shape, flush=True)

# var_names here are Ensembl IDs, not gene symbols - the real panel's gene_names.csv and
# pert_counts.csv both use symbols, so remap to var['gene_name'] for correct matching.
gene_symbols = adata.var['gene_name'].astype(str).to_numpy()
labels = adata.obs['gene'].astype(str).to_numpy()
CONTROL_LABEL = 'non-targeting'

print("Control cells:", int(np.sum(labels == CONTROL_LABEL)), flush=True)
print("Perturbations:", len(np.unique(labels[labels != CONTROL_LABEL])), flush=True)


def mean_rows_backed(adata, indices, block=512):
    indices = np.sort(np.asarray(indices, dtype=np.int64))
    out = np.zeros(adata.n_vars, dtype=np.float64)
    if not len(indices):
        raise ValueError('Empty subset')
    for i in range(0, len(indices), block):
        chunk = adata.X[indices[i:i + block], :]
        if sp.issparse(chunk):
            out += np.asarray(chunk.sum(axis=0)).ravel()
        else:
            out += np.asarray(chunk, dtype=np.float64).sum(axis=0)
    return out / len(indices)


def compute_bank(adata, lbls, control_label):
    groups = np.unique(lbls)
    idx_map = {g: np.flatnonzero(lbls == g) for g in groups}
    baseline = mean_rows_backed(adata, idx_map[control_label])
    effects = {}
    n_groups = len(groups)
    t0 = time.time()
    for gi, group in enumerate(groups):
        if group == control_label:
            continue
        kept = idx_map[group]
        if len(kept) < MIN_CELLS_PER_GROUP:
            continue
        mean = mean_rows_backed(adata, kept)
        effect = np.log((mean + PSEUDOCOUNT) / (baseline + PSEUDOCOUNT))
        effects[str(group)] = np.clip(effect, -CLIP_LOG_FOLD, CLIP_LOG_FOLD).astype(np.float32)
        if gi % 500 == 0:
            elapsed = time.time() - t0
            print(f"  {gi}/{n_groups} genes processed, {elapsed:.1f}s elapsed", flush=True)
    return baseline, effects


t_start = time.time()
baseline, effects = compute_bank(adata, labels, CONTROL_LABEL)
print(f"Trained signatures: {len(effects)}, time (min): {(time.time() - t_start) / 60:.1f}", flush=True)

np.savez_compressed(
    ARTIFACTS / 'signature_bank_k562gwps.npz',
    genes=np.asarray(gene_symbols),
    perts=np.asarray(list(effects)),
    effects=np.stack(list(effects.values())),
    control_mean=baseline,
    shrinkage=np.array([SHRINKAGE]),
)
print("Saved signature_bank_k562gwps.npz, size MB",
      round((ARTIFACTS / 'signature_bank_k562gwps.npz').stat().st_size / 1e6, 2), flush=True)
adata.file.close()

# --- Check overlap against the real 2026 panels ---
bank = np.load(ARTIFACTS / 'signature_bank_k562gwps.npz', allow_pickle=True)
source_pert_index = {x: i for i, x in enumerate(bank['perts'].astype(str))}

for name, directory in [('validation', ROOT / 'vcc_2026_controls'), ('final', ROOT / 'vcc_2026_controls_final')]:
    pert_list = pd.read_csv(directory / 'pert_counts.csv').iloc[:, 0].astype(str).tolist()
    overlap = sum(p in source_pert_index for p in pert_list)
    print(f"{name}: {overlap} / {len(pert_list)} perturbations have a trained K562_gwps signature", flush=True)

print("DONE", flush=True)
