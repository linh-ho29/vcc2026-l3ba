"""
The decisive test: does v2 (fit on pooled 4-context Arc PIE data) actually
improve the REAL production pipeline, not just Arc's own framework?

Applies v2's per-gene weight to gate the PRODUCTION K562_gwps bank's
signature, evaluated against RPE1's REAL measured perturbation outcomes
(rpe1_normalized_singlecell_01.h5ad) on the identical 1,995-gene overlap
used throughout this project (E01-E06, E10) for direct comparability:
  E01 (ungated):        Pearson 0.481  (prior result)
  E03 (combined gate):   Pearson 0.501  (prior result, current champion)
  v1 (K562<->RPE1 only): underperformed naive blend on HepG2 (0.664 vs 0.729)

v2's target-context features are computed from RPE1's own raw bulk control
cells (rpe1_raw_bulk_01.h5ad), log1p-transformed - the same basis used for
v1, NOT the z-score-normalized singlecell file (that one is ground truth
only here, never a feature source - same scale discipline as the rest of
this project).
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

ROOT = Path(__file__).parent.parent
PRED_DIR = ROOT / "predictions"
RPE1_RAW_BULK = ROOT / "data" / "rpe1_raw_bulk_01.h5ad"
RPE1_SINGLECELL = ROOT / "data" / "rpe1_normalized_singlecell_01.h5ad"
V1_MODEL = PRED_DIR / "context_weight_model.npz"
V2_MODEL = PRED_DIR / "context_weight_model_v2.npz"
OUT_PATH = ROOT / "vcc_agent_harness" / "experiments" / "E09_v2_vs_production_rpe1.json"

SHRINKAGE = 0.65
MIN_CELLS_PER_PERT = 15
CONTROL_LABEL = "non-targeting"

bank = np.load(PRED_DIR / "signature_bank_k562gwps.npz", allow_pickle=True)
source_genes = bank["genes"].astype(str)
source_perts = bank["perts"].astype(str)
source_effects = bank["effects"]
source_gene_index = {g: i for i, g in enumerate(source_genes)}
source_pert_index = {p: i for i, p in enumerate(source_perts)}

# --- RPE1's own real control-cell stats, from the SAME raw-bulk basis v1/v2 were trained on ---
print("Loading RPE1 raw bulk (for target-context features, same basis as training)...", flush=True)
a_bulk = ad.read_h5ad(RPE1_RAW_BULK, backed="r")
bulk_genes = a_bulk.var["gene_name"].astype(str).to_numpy()
names = a_bulk.obs_names.tolist()
targets_bulk = np.array([n.split("_")[1] for n in names])
ncells_bulk = a_bulk.obs["num_cells_unfiltered"].to_numpy().astype(np.float64)
X_bulk = a_bulk.X[:, :]
a_bulk.file.close()
ctrl_mask = targets_bulk == CONTROL_LABEL
w_ctrl = ncells_bulk[ctrl_mask]
rpe1_control_mean = np.average(X_bulk[ctrl_mask], axis=0, weights=w_ctrl)
rpe1_bulk_gene_idx = {g: i for i, g in enumerate(bulk_genes)}
rpe1_pert_targets = set(np.unique(targets_bulk[targets_bulk != CONTROL_LABEL]))


def rpe1_target_features(gene_list):
    """baseline_expression (log1p), percentile, 4 pathway flags, is_perturbation_target -
    for a given gene_list (in source_genes order), using RPE1's own raw-bulk control mean."""
    n = len(gene_list)
    baseline_full = np.log1p(rpe1_control_mean)
    order = np.argsort(baseline_full)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(order))
    percentile_full = 100.0 * ranks / max(len(order) - 1, 1)

    baseline = np.full(n, np.nan)
    percentile = np.full(n, np.nan)
    for j, g in enumerate(gene_list):
        bi = rpe1_bulk_gene_idx.get(g)
        if bi is not None:
            baseline[j] = baseline_full[bi]
            percentile[j] = percentile_full[bi]
    pathway_ribo = np.array([g.startswith(("RPL", "RPS")) for g in gene_list], dtype=np.float64)
    pathway_mito = np.array([g.startswith("MT-") for g in gene_list], dtype=np.float64)
    pathway_histone = np.array([g.startswith(("HIST", "H1-", "H2A", "H2B", "H3-", "H4-")) for g in gene_list],
                                dtype=np.float64)
    pathway_hemo = np.array([g.startswith(("HBA", "HBB", "HBZ", "HBG", "HBD", "HBE")) for g in gene_list],
                             dtype=np.float64)
    is_pert_target = np.array([g in rpe1_pert_targets for g in gene_list], dtype=np.float64)
    return baseline, percentile, pathway_ribo, pathway_mito, pathway_histone, pathway_hemo, is_pert_target


def apply_model(model_path, gene_list, feature_order):
    m = np.load(model_path, allow_pickle=True)
    coef, feat_mean, feat_std = m["coef"], m["feat_mean"], m["feat_std"]
    feats = rpe1_target_features(gene_list)
    X = np.column_stack(feats)  # order matches feature_order by construction below
    valid = ~np.isnan(X).any(axis=1)
    Xb = np.hstack([(X - feat_mean) / feat_std, np.ones((X.shape[0], 1))])
    w = np.full(len(gene_list), np.nan)
    w[valid] = np.clip((Xb @ coef)[valid], 0.0, 1.0)
    return w


# v1's feature order: baseline_expression, detection_fraction, 4 pathway, source_target_similarity, is_pert_target
# v2's feature order: baseline_expression, percentile, 4 pathway, is_pert_target
# Different feature sets by design (v1 used detection_fraction from a different file + source_target_similarity;
# v2 simplified to percentile-only + no cross-source feature) - apply each with its OWN matching features.
def apply_v1(gene_list):
    m = np.load(V1_MODEL, allow_pickle=True)
    coef, feat_mean, feat_std = m["coef"], m["feat_mean"], m["feat_std"]
    baseline, percentile, p_ribo, p_mito, p_hist, p_hemo, is_pt = rpe1_target_features(gene_list)
    k562_bulk_gene_idx = source_gene_index
    src_tgt_sim = np.full(len(gene_list), np.nan)
    for j, g in enumerate(gene_list):
        ki = k562_bulk_gene_idx.get(g)
        if ki is not None and not np.isnan(baseline[j]):
            src_tgt_sim[j] = -abs(np.log1p(bank["control_mean"][ki]) - baseline[j])
    detection_frac = np.full(len(gene_list), np.nan)  # v1 used real detection_fraction; approximate via percentile>0 proxy unavailable here, use NaN->mean-imputed below
    X = np.column_stack([baseline, detection_frac, p_ribo, p_mito, p_hist, p_hemo, src_tgt_sim, is_pt])
    X = np.where(np.isnan(X), feat_mean, X)  # mean-impute unavailable features (detection_fraction) rather than drop
    valid = ~np.isnan(baseline)
    Xb = np.hstack([(X - feat_mean) / feat_std, np.ones((X.shape[0], 1))])
    w = np.full(len(gene_list), np.nan)
    w[valid] = np.clip((Xb @ coef)[valid], 0.0, 1.0)
    return w


def apply_v2(gene_list):
    m = np.load(V2_MODEL, allow_pickle=True)
    coef, feat_mean, feat_std = m["coef"], m["feat_mean"], m["feat_std"]
    baseline, percentile, p_ribo, p_mito, p_hist, p_hemo, is_pt = rpe1_target_features(gene_list)
    X = np.column_stack([baseline, percentile, p_ribo, p_mito, p_hist, p_hemo, is_pt])
    valid = ~np.isnan(baseline)
    Xb = np.hstack([(X - feat_mean) / feat_std, np.ones((X.shape[0], 1))])
    w = np.full(len(gene_list), np.nan)
    w[valid] = np.clip((Xb @ coef)[valid], 0.0, 1.0)
    return w


# --- RPE1's real ground truth (z-score file) + real control-cell gate stats for E01/E03 ---
print("Loading RPE1 normalized singlecell (ground truth + E03's own gate stats)...", flush=True)
adata = ad.read_h5ad(RPE1_SINGLECELL, backed="r")
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
rpe1_baseline = mean_rows_backed(adata, ctrl_idx)
rpe1_gate_stats = compute_context_gate_stats(_ControlView(adata, ctrl_idx), block=512)
gate_e03 = compute_gate_weights(rpe1_gate_stats, GateConfig(mode="combined"))

non_ctrl_labels = labels[labels != CONTROL_LABEL]
groups_, counts_ = np.unique(non_ctrl_labels, return_counts=True)
groups_, counts_ = groups_[counts_ >= MIN_CELLS_PER_PERT], counts_[counts_ >= MIN_CELLS_PER_PERT]
overlap_perts = [g for g in groups_ if g in source_pert_index and g in rpe1_gene_index]
print(f"Overlap (K562-trained AND RPE1-measured as a perturbation target): {len(overlap_perts)} genes", flush=True)

v1_weight = apply_v1(overlap_perts)
v2_weight = apply_v2(overlap_perts)

y_true, k562_effect_arr, gate_e03_arr, v1_w_arr, v2_w_arr = [], [], [], [], []
for i, pert in enumerate(overlap_perts):
    pert_idx = np.flatnonzero(labels == pert)
    rpe1_mean = mean_rows_backed(adata, pert_idx)
    j = rpe1_gene_index[pert]
    y_true.append(float(rpe1_mean[j] - rpe1_baseline[j]))
    k562_effect_arr.append(float(source_effects[source_pert_index[pert]][source_gene_index.get(pert, 0)])
                            if pert in source_gene_index else 0.0)
    gate_e03_arr.append(float(gate_e03[j]))
    v1_w_arr.append(v1_weight[i])
    v2_w_arr.append(v2_weight[i])
adata.file.close()

y_true = np.array(y_true)
k562_effect_arr = np.array(k562_effect_arr)
gate_e03_arr = np.array(gate_e03_arr)
v1_w_arr = np.nan_to_num(np.array(v1_w_arr), nan=np.nanmean(v1_w_arr))
v2_w_arr = np.nan_to_num(np.array(v2_w_arr), nan=np.nanmean(v2_w_arr))

variants = {
    "E01_ungated": np.ones(len(overlap_perts)),
    "E03_combined_gate_CHAMPION": gate_e03_arr,
    "v1_old_E09_weight": v1_w_arr,
    "v2_new_E09_weight": v2_w_arr,
}

print(f"\n=== Production K562_gwps -> real RPE1 ground truth (n={len(overlap_perts)}) ===", flush=True)
results = {}
for name, gate in variants.items():
    preds = k562_effect_arr * gate * SHRINKAGE
    pearson = pearsonr(y_true, preds)[0] if np.std(preds) > 0 else float("nan")
    spearman = spearmanr(y_true, preds)[0] if np.std(preds) > 0 else float("nan")
    sign_agree = float(np.mean(np.sign(y_true) == np.sign(preds)))
    results[name] = {"pearson": float(pearson), "spearman": float(spearman), "sign_agreement": sign_agree}
    print(f"  {name:28s} Pearson={pearson:.4f}  Spearman={spearman:.4f}  sign_agree={sign_agree:.4f}", flush=True)

champion_pearson = results["E03_combined_gate_CHAMPION"]["pearson"]
v2_beats_champion = results["v2_new_E09_weight"]["pearson"] > champion_pearson
print(f"\nv2 beats E03 champion ({champion_pearson:.4f}): {v2_beats_champion}", flush=True)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUT_PATH.write_text(json.dumps({
    "n": len(overlap_perts), "results": results, "v2_beats_champion": bool(v2_beats_champion),
    "note": "v1 applied with detection_fraction/source_target_similarity mean-imputed (not computable "
            "from the raw-bulk-only feature basis used here) - v1's own number should be read as "
            "approximate; v2 uses its full native feature set exactly as trained, no imputation needed.",
}, indent=2))
print(f"\nSaved to {OUT_PATH}")
print("DONE")
