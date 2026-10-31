"""
E06: context-control-only representation (E03) vs context-control +
scBaseCount representation, evaluated against RPE1's own REAL measured
perturbation effects - genuine cross-cell-type accuracy, not a prediction-
difference diagnostic (same protocol as cross_context_validation_rpe1.py's
E01-E04 check). RPE1 was explicitly excluded from the E05 scBaseCount
reference pool (see data/manifests/scbasecount_selection.csv's leakage_check
column), so this validation is not circular.

Representations compared, each evaluated at lambda in {0.0, 0.3, 0.6, 1.0}:
  - E06_best_match: RPE1's own context stats blended with the scBaseCount
    reference group that best correlates with RPE1's real detection-fraction
    profile (data-driven match, no cell-type label used).
  - E06_mismatched: blended with the WORST-correlating reference group
    instead - a deliberately wrong representation.
  - E06_shuffled: blended with the best-matched group's stats after
    permuting its gene axis - destroys real gene-level correspondence while
    preserving the marginal distribution.
lambda=0.0 must recover E03 exactly for every variant (blend() returns the
context-only value when lam==0) - this is checked, not assumed.

If scBaseCount only helps via generic shrinkage (not real gene-level
information), the mismatched and shuffled controls should perform
comparably to the best-matched representation. If they perform
meaningfully worse, that's evidence the gene-level content is doing real
work.
"""
import json
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp
from scipy.stats import pearsonr, spearmanr

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))
from expression_gating import GateConfig, compute_context_gate_stats, compute_gate_weights
from scbasecount_blend import best_and_worst_matching_group, reference_gene_stats, blend

REPO_ROOT = Path(__file__).parent.parent
RPE1_PATH = REPO_ROOT / "data" / "rpe1_normalized_singlecell_01.h5ad"
PRED_DIR = REPO_ROOT / "predictions"
REF_PATH = PRED_DIR / "scbasecount_reference.npz"
OUT_PATH = REPO_ROOT / "vcc_agent_harness" / "experiments" / "E05_E06_metrics.json"

SEED = 2026
SHRINKAGE = 0.65
MIN_CELLS_PER_PERT = 15
CONTROL_LABEL = "non-targeting"
LAMBDAS = [0.0, 0.3, 0.6, 1.0]

bank = np.load(PRED_DIR / "signature_bank_k562gwps.npz", allow_pickle=True)
source_genes = bank["genes"].astype(str)
source_perts = bank["perts"].astype(str)
source_effects = bank["effects"]
source_gene_index = {g: i for i, g in enumerate(source_genes)}
source_pert_index = {p: i for i, p in enumerate(source_perts)}

ref = np.load(REF_PATH, allow_pickle=True)
print(f"scBaseCount reference: groups={ref['reference_groups'].tolist()}, "
      f"{len(ref['gene_symbols'])} genes", flush=True)

print("Loading RPE1 (backed)...", flush=True)
adata = ad.read_h5ad(RPE1_PATH, backed="r")
labels = adata.obs["gene"].astype(str).to_numpy()
rpe1_gene_symbols = adata.var["gene_name"].astype(str).to_numpy()
rpe1_gene_index = {g: i for i, g in enumerate(rpe1_gene_symbols)}


def mean_rows_backed(adata, indices, block=512):
    indices = np.sort(np.asarray(indices, dtype=np.int64))
    out = np.zeros(adata.n_vars, dtype=np.float64)
    if not len(indices):
        return out
    for i in range(0, len(indices), block):
        chunk = adata.X[indices[i:i + block], :]
        out += (np.asarray(chunk.sum(axis=0)).ravel() if sp.issparse(chunk)
                else np.asarray(chunk, dtype=np.float64).sum(axis=0))
    return out / len(indices)


class _ControlView:
    def __init__(self, adata, ctrl_idx):
        self.adata, self.ctrl_idx, self.shape = adata, ctrl_idx, (len(ctrl_idx), adata.n_vars)
    def __getitem__(self, key):
        idx = self.ctrl_idx[key] if isinstance(key, slice) else self.ctrl_idx[key]
        return self.adata.X[np.sort(idx), :]


ctrl_idx = np.flatnonzero(labels == CONTROL_LABEL)
print(f"RPE1 control cells: {len(ctrl_idx)}", flush=True)
rpe1_baseline = mean_rows_backed(adata, ctrl_idx)
print("Computing RPE1's own real context gate stats (whole transcriptome)...", flush=True)
rpe1_gate_stats = compute_context_gate_stats(_ControlView(adata, ctrl_idx), block=512)

print("Finding best/worst-matching scBaseCount reference group for RPE1 "
      "(data-driven, via detection-fraction correlation - no cell-type label used)...", flush=True)
best_group, worst_group, corrs, shared_genes = best_and_worst_matching_group(
    rpe1_gate_stats["detection_fraction"], rpe1_gene_symbols, ref)
print(f"  correlations: {json.dumps({k: round(v, 4) for k, v in corrs.items()}, indent=2)}", flush=True)
print(f"  best match: {best_group} (shared genes used: {len(shared_genes)})", flush=True)
print(f"  worst match (mismatched negative control): {worst_group}", flush=True)

non_ctrl_labels = labels[labels != CONTROL_LABEL]
groups_, counts_ = np.unique(non_ctrl_labels, return_counts=True)
groups_, counts_ = groups_[counts_ >= MIN_CELLS_PER_PERT], counts_[counts_ >= MIN_CELLS_PER_PERT]
overlap_perts = [g for g in groups_ if g in source_pert_index and g in rpe1_gene_index]
print(f"Overlap (K562-trained AND RPE1-measured as a perturbation target): {len(overlap_perts)} genes", flush=True)

variant_names = []
for variant, group in [("best_match", best_group), ("mismatched", worst_group), ("shuffled", best_group)]:
    for lam in LAMBDAS:
        variant_names.append(f"E06_{variant}_lam{lam}")

y_true = []
y_pred_by_variant = {name: [] for name in variant_names}
reference_fallback_count = 0

for pert in overlap_perts:
    pert_idx = np.flatnonzero(labels == pert)
    rpe1_mean = mean_rows_backed(adata, pert_idx)
    j = rpe1_gene_index[pert]
    rpe1_real_effect = float(rpe1_mean[j] - rpe1_baseline[j])
    y_true.append(rpe1_real_effect)

    k562_j = source_gene_index.get(pert, None)
    k562_effect = float(source_effects[source_pert_index[pert]][k562_j]) if k562_j is not None else 0.0

    context_det = float(rpe1_gate_stats["detection_fraction"][j])
    context_pct = float(rpe1_gate_stats["percentile"][j])
    gene_symbol = rpe1_gene_symbols[j]

    for variant, group in [("best_match", best_group), ("mismatched", worst_group), ("shuffled", best_group)]:
        shuffle_seed = SEED if variant == "shuffled" else None
        ref_stats = reference_gene_stats(ref, group, gene_symbol, shuffle_seed=shuffle_seed)
        if ref_stats is None:
            reference_fallback_count += 1
        for lam in LAMBDAS:
            det, pct = blend(context_det, context_pct, *(ref_stats if ref_stats else (None, None)), lam)
            stats_j = {"detection_fraction": np.array([det]), "percentile": np.array([pct]),
                       "baseline_mean": np.array([0.0])}
            gate = compute_gate_weights(stats_j, GateConfig(mode="combined"))[0]
            y_pred_by_variant[f"E06_{variant}_lam{lam}"].append(k562_effect * gate * SHRINKAGE)

adata.file.close()
y_true = np.array(y_true)

results = {
    "overlap_count": len(overlap_perts),
    "best_matching_group": best_group,
    "worst_matching_group": worst_group,
    "group_correlations": corrs,
    "reference_fallback_count": reference_fallback_count,
    "note": "RPE1 file is z-score-normalized, not raw counts - only correlation "
            "(direction/rank agreement) is reported, not MSE, as in the E01-E04 check. "
            "lam=0.0 is checked to exactly equal E03_combined from the E01-E04 run.",
    "variants": {},
}

print(f"\n=== E06: K562 -> RPE1 cross-context accuracy (n={len(overlap_perts)} held-out genes) ===", flush=True)
for name, preds in y_pred_by_variant.items():
    preds = np.array(preds)
    if np.std(preds) > 0 and np.std(y_true) > 0:
        pearson, _ = pearsonr(y_true, preds)
        spearman, _ = spearmanr(y_true, preds)
    else:
        pearson, spearman = float("nan"), float("nan")
    sign_agreement = float(np.mean(np.sign(y_true) == np.sign(preds)))
    results["variants"][name] = {"pearson": float(pearson), "spearman": float(spearman),
                                  "sign_agreement": sign_agreement}
    print(f"  {name:28s} Pearson={pearson:.4f}  Spearman={spearman:.4f}  sign_agreement={sign_agreement:.4f}", flush=True)

e03_pearson = results["variants"]["E06_best_match_lam0.0"]["pearson"]
e03_mismatched_pearson = results["variants"]["E06_mismatched_lam0.0"]["pearson"]
e03_shuffled_pearson = results["variants"]["E06_shuffled_lam0.0"]["pearson"]
consistent = np.allclose(e03_pearson, e03_mismatched_pearson) and np.allclose(e03_pearson, e03_shuffled_pearson)
print(f"\nlambda=0.0 sanity check (must all be equal to each other, and to the prior E03_combined "
      f"result of Pearson=0.5007): best_match={e03_pearson:.4f}, mismatched={e03_mismatched_pearson:.4f}, "
      f"shuffled={e03_shuffled_pearson:.4f}, all_equal={consistent}", flush=True)
results["lambda_zero_sanity_check_passed"] = bool(consistent)
results["lambda_zero_matches_prior_E03_pearson_0.5007"] = bool(np.isclose(e03_pearson, 0.5007, atol=1e-3))

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUT_PATH.write_text(json.dumps(results, indent=2))
print(f"\nSaved to {OUT_PATH}")
print("DONE")
