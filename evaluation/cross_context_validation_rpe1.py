"""
Cross-context validation: Replogle RPE1.

Unlike the D/E/F diagnostics (no ground truth exists there), RPE1 has its
OWN real measured perturbation responses. For genes RPE1 actually screened
in its own essential-gene library AND that K562_gwps also has a trained
signature for, this compares:
  - K562's effect, transferred to RPE1 via each gating mode, using RPE1's
    own REAL basal expression (detection/percentile) for the gate
  - RPE1's own REAL measured effect for that same gene (genuine ground truth)

This is actual cross-cell-type accuracy validation, not a prediction-
difference diagnostic. Confirms earlier: RPE1's own essential-gene library
has ZERO overlap with the competition's 300-gene panel, so this cannot and
does not tell us anything about panel coverage - it is purely a test of
whether gating improves true cross-context transfer accuracy in general.
"""
import json
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))
from expression_gating import (GateConfig, compute_context_gate_stats, compute_gate_weights,
                                compute_gate_weights_direction_aware)

REPO_ROOT = Path(__file__).parent.parent
RPE1_PATH = REPO_ROOT / "data" / "rpe1_normalized_singlecell_01.h5ad"
PRED_DIR = REPO_ROOT / "predictions"
OUT_PATH = REPO_ROOT / "vcc_agent_harness" / "experiments" / "rpe1_cross_validation_metrics.json"

SEED = 2026
SHRINKAGE = 0.65
CLIP_LOG_FOLD = 1.0
PSEUDOCOUNT = 0.5
CONTROL_LABEL = "non-targeting"
MIN_CELLS_PER_PERT = 15

bank = np.load(PRED_DIR / "signature_bank_k562gwps.npz", allow_pickle=True)
source_genes = bank["genes"].astype(str)
source_perts = bank["perts"].astype(str)
source_effects = bank["effects"]
source_gene_index = {g: i for i, g in enumerate(source_genes)}
source_pert_index = {p: i for i, p in enumerate(source_perts)}
print(f"K562_gwps bank: {len(source_perts)} perturbations, {len(source_genes)} genes", flush=True)

print("Loading RPE1 (backed)...", flush=True)
adata = ad.read_h5ad(RPE1_PATH, backed="r")
labels = adata.obs["gene"].astype(str).to_numpy()
rpe1_gene_symbols = adata.var["gene_name"].astype(str).to_numpy()
rpe1_gene_index = {g: i for i, g in enumerate(rpe1_gene_symbols)}
print(f"RPE1 shape: {adata.shape}", flush=True)


def mean_rows_backed(adata, indices, block=512):
    indices = np.sort(np.asarray(indices, dtype=np.int64))
    out = np.zeros(adata.n_vars, dtype=np.float64)
    if not len(indices):
        return out
    for i in range(0, len(indices), block):
        chunk = adata.X[indices[i:i + block], :]
        if sp.issparse(chunk):
            out += np.asarray(chunk.sum(axis=0)).ravel()
        else:
            out += np.asarray(chunk, dtype=np.float64).sum(axis=0)
    return out / len(indices)


# RPE1's own real control cells -> real basal expression stats (gate inputs) AND real baseline mean.
ctrl_idx = np.flatnonzero(labels == CONTROL_LABEL)
X_all = None  # loaded in chunks inside compute_context_gate_stats / mean_rows_backed, not materialized fully
print(f"RPE1 control cells: {len(ctrl_idx)}", flush=True)
rpe1_baseline = mean_rows_backed(adata, ctrl_idx)

print("Computing RPE1 context gate stats from its own real control cells...", flush=True)
# compute_context_gate_stats expects the control-cell SUBSET, not the whole matrix - build a
# lightweight wrapper that only reads control-cell rows, chunked.
class _ControlView:
    def __init__(self, adata, ctrl_idx):
        self.adata, self.ctrl_idx, self.shape = adata, ctrl_idx, (len(ctrl_idx), adata.n_vars)
    def __getitem__(self, key):
        if isinstance(key, slice):
            idx = self.ctrl_idx[key]
        else:
            idx = self.ctrl_idx[key]
        return self.adata.X[np.sort(idx), :]

gate_stats = compute_context_gate_stats(_ControlView(adata, ctrl_idx), block=512)

# RPE1's own real measured perturbation effects (log space, matching K562 bank's convention).
non_ctrl_labels = labels[labels != CONTROL_LABEL]
groups, counts = np.unique(non_ctrl_labels, return_counts=True)
groups, counts = groups[counts >= MIN_CELLS_PER_PERT], counts[counts >= MIN_CELLS_PER_PERT]
print(f"RPE1 perturbations with >= {MIN_CELLS_PER_PERT} cells: {len(groups)}", flush=True)

# Overlap: gene must (a) have a K562_gwps-trained signature for THAT SAME gene as a perturbation
# target, and (b) be a gene RPE1 itself measured as a downstream/expressed gene, and (c) RPE1 must
# have ALSO perturbed that same gene itself (so we have RPE1's own real ground-truth effect).
overlap_perts = [g for g in groups if g in source_pert_index and g in rpe1_gene_index]
print(f"Overlap (K562-trained AND RPE1-measured as a perturbation target): {len(overlap_perts)} genes", flush=True)

if len(overlap_perts) == 0:
    print("NO OVERLAP - cannot run cross-validation. Exiting.", flush=True)
    OUT_PATH.write_text(json.dumps({"status": "NO_OVERLAP", "overlap_count": 0}, indent=2))
    raise SystemExit(0)

y_true = []  # RPE1's own real effect (downstream gene = the perturbed gene itself, measured directly)
y_pred_by_mode = {m: [] for m in ["E01_none", "E02_binary_detection", "E03_combined", "E04_direction_aware"]}

rng_check = 0
for pert in overlap_perts:
    pert_idx = np.flatnonzero(labels == pert)
    rpe1_mean = mean_rows_backed(adata, pert_idx)
    j = rpe1_gene_index[pert]  # downstream gene = the perturbed gene itself (on-target knockdown)
    # NOTE: this file is "normalized" (confirmed z-score-like: 68% negative values, zero exact
    # zeros, range roughly -5 to +149) - NOT raw counts. The K562 bank's log-pseudocount-ratio
    # formula assumes non-negative raw-count-like values and is invalid here. Use a simple
    # difference in RPE1's own native units instead - valid regardless of the exact transform,
    # since it doesn't require non-negativity. Compared against K562 via correlation only (not
    # MSE, since the two are not on directly comparable absolute scales).
    rpe1_real_effect = float(rpe1_mean[j] - rpe1_baseline[j])
    y_true.append(rpe1_real_effect)

    k562_raw_effect_full = source_effects[source_pert_index[pert]]  # over K562_gwps's gene vocabulary
    k562_j_in_source = source_gene_index.get(pert, None)
    k562_raw_effect_j = float(k562_raw_effect_full[k562_j_in_source]) if k562_j_in_source is not None else 0.0

    stats_j = {"detection_fraction": gate_stats["detection_fraction"][j:j+1],
               "percentile": gate_stats["percentile"][j:j+1],
               "baseline_mean": gate_stats["baseline_mean"][j:j+1]}
    raw_effect_arr = np.array([k562_raw_effect_j])

    gate_none = compute_gate_weights(stats_j, GateConfig(mode="none"))[0]
    gate_binary = compute_gate_weights(stats_j, GateConfig(mode="binary_detection"))[0]
    gate_combined = compute_gate_weights(stats_j, GateConfig(mode="combined"))[0]
    gate_direction = compute_gate_weights_direction_aware(stats_j, raw_effect_arr, GateConfig(mode="combined"))[0]

    y_pred_by_mode["E01_none"].append(k562_raw_effect_j * gate_none * SHRINKAGE)
    y_pred_by_mode["E02_binary_detection"].append(k562_raw_effect_j * gate_binary * SHRINKAGE)
    y_pred_by_mode["E03_combined"].append(k562_raw_effect_j * gate_combined * SHRINKAGE)
    y_pred_by_mode["E04_direction_aware"].append(k562_raw_effect_j * gate_direction * SHRINKAGE)

adata.file.close()
y_true = np.array(y_true)

from scipy.stats import spearmanr

results = {"overlap_count": len(overlap_perts), "modes": {},
           "note": "RPE1 file is normalized (z-score-like, not raw counts) - K562's log-fold "
                   "effect and RPE1's real effect are NOT on directly comparable absolute "
                   "scales, so only correlation (direction/rank agreement) is reported, not MSE."}
print(f"\n=== Cross-context DIRECTION agreement: K562 -> RPE1 (n={len(overlap_perts)} genuinely held-out genes) ===", flush=True)
for mode, preds in y_pred_by_mode.items():
    preds = np.array(preds)
    pearson, _ = pearsonr(y_true, preds) if np.std(preds) > 0 and np.std(y_true) > 0 else (float("nan"), None)
    spearman, _ = spearmanr(y_true, preds) if np.std(preds) > 0 and np.std(y_true) > 0 else (float("nan"), None)
    sign_agreement = float(np.mean(np.sign(y_true) == np.sign(preds)))
    results["modes"][mode] = {"pearson": float(pearson), "spearman": float(spearman),
                               "sign_agreement": sign_agreement}
    print(f"  {mode:25s} Pearson={pearson:.4f}  Spearman={spearman:.4f}  sign_agreement={sign_agreement:.4f}", flush=True)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUT_PATH.write_text(json.dumps(results, indent=2))
print(f"\nSaved to {OUT_PATH}")
print("DONE")
