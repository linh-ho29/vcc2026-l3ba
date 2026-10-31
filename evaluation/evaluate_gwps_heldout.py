import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import anndata as ad
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr

ROOT = Path.home() / 'vcc2026_cpu'
GWPS_PATH = ROOT / 'train' / 'K562_gwps_raw_singlecell_01.h5ad'

SEED = 2026
N_VALID_PER_GROUP = 40
MIN_CELLS_PER_GROUP = 15
SHRINKAGE = 0.65
PSEUDOCOUNT = 0.5
CLIP_LOG_FOLD = 1.0
CONTROL_LABEL = 'non-targeting'


def compute_vcc_metrics(y_true, y_pred, baseline_control):
    """
    Computes local approximations of the Virtual Cell Challenge metrics.
    y_true, y_pred, baseline_control shape: (n_cells, n_genes)
    """
    mse = mean_squared_error(y_true, y_pred)
    delta_true = y_true - baseline_control
    delta_pred = y_pred - baseline_control
    delta_true_flat = delta_true.flatten()
    delta_pred_flat = delta_pred.flatten()
    pearson_delta, _ = pearsonr(delta_true_flat, delta_pred_flat)
    spearman_delta, _ = spearmanr(delta_true_flat, delta_pred_flat)
    return {
        "MSE": mse,
        "Pearson_Delta": pearson_delta,
        "Spearman_Delta": spearman_delta,
    }


def mean_rows_backed(adata, indices, block=512):
    indices = np.sort(np.asarray(indices, dtype=np.int64))
    out = np.zeros(adata.n_vars, dtype=np.float64)
    for i in range(0, len(indices), block):
        chunk = adata.X[indices[i:i + block], :]
        if sp.issparse(chunk):
            out += np.asarray(chunk.sum(axis=0)).ravel()
        else:
            out += np.asarray(chunk, dtype=np.float64).sum(axis=0)
    return out / len(indices)


print("Loading K562_gwps (backed)...", flush=True)
adata = ad.read_h5ad(GWPS_PATH, backed='r')
labels = adata.obs['gene'].astype(str).to_numpy()
gene_symbols = adata.var['gene_name'].astype(str).to_numpy()
print("shape:", adata.shape, flush=True)

rng = np.random.default_rng(SEED)
groups = np.unique(labels)

# Split control cells: some for baseline (train), some held out for fairness
ctrl_idx = np.flatnonzero(labels == CONTROL_LABEL)
ctrl_holdout = rng.choice(ctrl_idx, size=N_VALID_PER_GROUP, replace=False)
ctrl_train = np.setdiff1d(ctrl_idx, ctrl_holdout, assume_unique=True)
baseline_control_vec = mean_rows_backed(adata, ctrl_train)
print(f"Baseline computed from {len(ctrl_train)} control cells", flush=True)

y_true_rows = []
y_pred_rows = []
tested_genes = []

t0 = time.time()
perturbed_groups = [g for g in groups if g != CONTROL_LABEL]
for gi, group in enumerate(perturbed_groups):
    ids = np.flatnonzero(labels == group)
    if len(ids) < MIN_CELLS_PER_GROUP + N_VALID_PER_GROUP:
        continue
    holdout = rng.choice(ids, size=N_VALID_PER_GROUP, replace=False)
    train = np.setdiff1d(ids, holdout, assume_unique=True)

    mean_train = mean_rows_backed(adata, train)
    effect = np.log((mean_train + PSEUDOCOUNT) / (baseline_control_vec + PSEUDOCOUNT))
    effect = np.clip(effect, -CLIP_LOG_FOLD, CLIP_LOG_FOLD)
    y_pred_row = baseline_control_vec * np.exp(SHRINKAGE * effect)

    y_true_row = mean_rows_backed(adata, holdout)

    y_true_rows.append(y_true_row)
    y_pred_rows.append(y_pred_row)
    tested_genes.append(group)

    if (gi + 1) % 500 == 0:
        print(f"  {gi + 1}/{len(perturbed_groups)} genes processed, {time.time() - t0:.1f}s elapsed, "
              f"{len(tested_genes)} usable so far", flush=True)

adata.file.close()
print(f"Total usable held-out genes: {len(tested_genes)}, time: {(time.time() - t0) / 60:.1f} min", flush=True)

y_true = np.stack(y_true_rows)
y_pred = np.stack(y_pred_rows)
baseline_control = np.tile(baseline_control_vec, (len(tested_genes), 1))

print("\n=== Signature-transfer model ===", flush=True)
metrics_model = compute_vcc_metrics(y_true, y_pred, baseline_control)
for k, v in metrics_model.items():
    print(f"  {k}: {v:.6f}", flush=True)

print("\n=== Naive 'no effect' baseline (predicts control for everything) ===", flush=True)
metrics_naive = compute_vcc_metrics(y_true, baseline_control, baseline_control)
for k, v in metrics_naive.items():
    print(f"  {k}: {v:.6f}", flush=True)

print("\n=== Per-gene MSE: model beats naive baseline ===", flush=True)
per_gene_mse_model = np.mean((y_true - y_pred) ** 2, axis=1)
per_gene_mse_naive = np.mean((y_true - baseline_control) ** 2, axis=1)
win_rate = np.mean(per_gene_mse_model < per_gene_mse_naive)
print(f"  model beats naive baseline in {100 * win_rate:.1f}% of {len(tested_genes)} held-out perturbations", flush=True)

np.savez_compressed(
    ROOT / 'artifacts' / 'heldout_eval_results.npz',
    y_true=y_true, y_pred=y_pred, baseline_control=baseline_control,
    tested_genes=np.asarray(tested_genes),
)
print("\nSaved heldout_eval_results.npz", flush=True)
print("DONE", flush=True)
